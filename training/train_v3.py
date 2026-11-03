"""Faithful raw-input/BN-adapting original architecture, with sealed test evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import ClassicalCNN, QuantumHybrid, CLASSES
from train import ArchiveDataset, DEFAULT_WEIGHTS, ROOT, metrics

OUT = ROOT / 'models/candidates/cleaned-v3-seed42'
ARCHIVE = ROOT / 'training/data/cleaned-v3.zip'
PROTOCOL = {
    'seed': 42, 'quantum_seed': 43, 'image_size': 224, 'batch_size': 16,
    'preprocessing_version': 'resize224-rgb-totensor-v1',
    'cnn': {'original_epochs': 10, 'max_epochs': 20, 'optimizer': 'Adam', 'lr': .001, 'weight_decay': .0001},
    'quantum': {'original_epochs': 5, 'max_epochs': 30, 'optimizer': 'Adam', 'lr': .001, 'weight_decay': 0},
    'continuation_rule': 'Only continue beyond original10/5 budgets if best validation accuracy or macroF1 is below .85; stop on both>=.85, eight stale epochs, or fixed max budget.',
    'selection': 'Highest validation macroF1; accuracy breaks exact ties. Never use test predictions for checkpoint or budget selection.',
    'scheduler': 'ReduceLROnPlateau validation loss, factor .5, patience3',
    'loss': 'Unweighted cross entropy, matching original',
    'backbone': 'EfficientNetB0 ImageNet1K_V1 parameters frozen; model.train() adapts BatchNorm buffers and enables stochastic depth, matching original execution.',
    'quantum_architecture': 'Original1280->4 tanh encoder, fourXAngleEmbedding qubits, fourBasicEntanglerLayers, PauliZreadout,4->64ReLU->3head.',
    'determinism': 'Fixed Python/NumPy/PyTorch seeds and loader generator. MPS kernels may be nondeterministic across runs/hardware; no bitwise reproducibility claim.',
    'test_policy': 'Train/validation stage never predicts the locked test set. Separate evaluate stage runs once after selection; originals are diagnostic baselines with possible training overlap.',
}


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def image_loader(rows, batch=16, shuffle=False, seed_value=42):
    return DataLoader(ArchiveDataset(ARCHIVE, rows, False), batch_size=batch, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed_value))


def evaluate_cnn(model, loader, device):
    model.eval()
    labels, predicted, loss_sum, count = [], [], 0., 0
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            loss_sum += float(nn.functional.cross_entropy(logits, targets.to(device)).cpu()) * len(targets)
            labels.extend(targets.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
            count += len(targets)
    result = metrics(labels, predicted)
    result['loss'] = loss_sum / count
    return result


def train_cnn(model, manifest, device):
    train_loader = image_loader(manifest['splits']['train'], shuffle=True)
    val_loader = image_loader(manifest['splits']['validation'], batch=32)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
    history, best, best_state, stale = [], None, None, 0
    for epoch in range(1, PROTOCOL['cnn']['max_epochs'] + 1):
        start = time.monotonic()
        model.train()  # Deliberately preserves original train-mode BN adaptation.
        total_loss, count = 0., 0
        for index, (images, targets) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = nn.functional.cross_entropy(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(targets)
            count += len(targets)
            if index % 25 == 0:
                print(json.dumps({'event': 'training_batches', 'kind': 'classical', 'epoch': epoch, 'batches': index, 'total_batches': len(train_loader), 'seconds': time.monotonic()-start}), flush=True)
        result = evaluate_cnn(model, val_loader, device)
        scheduler.step(result['loss'])
        row = {'epoch': epoch, 'train_loss': total_loss/count, 'validation': result, 'seconds': time.monotonic()-start}
        history.append(row)
        score = (result['macro_f1'], result['accuracy'])
        if best is None or score > best:
            best, best_state, stale = score, snapshot(model), 0
            torch.save(best_state, OUT/'xai_model.pth')
        else:
            stale += 1
        (OUT/'classical_history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps({'event': 'epoch', 'kind': 'classical', **row}), flush=True)
        if epoch >= PROTOCOL['cnn']['original_epochs'] and ((best[0] >= .85 and best[1] >= .85) or stale >= 8):
            break
    model.load_state_dict(best_state)
    model.eval()
    selected = max(history, key=lambda row: (row['validation']['macro_f1'], row['validation']['accuracy']))
    return {'selected_epoch': selected['epoch'], 'epochs_run': len(history), 'validation': selected['validation'], 'original_budget_validation': max(history[:10], key=lambda row: row['validation']['macro_f1'])['validation']}


def extract_features(model, rows, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in image_loader(rows, batch=32):
            features.append(model(images.to(device), return_features=True).cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def quantum_metrics(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch) for batch in features.split(128)])
    result = metrics(labels.numpy(), logits.argmax(1).numpy())
    result['loss'] = float(nn.functional.cross_entropy(logits, labels))
    return result


def train_quantum(model, features, labels, val_features, val_labels):
    loader = DataLoader(TensorDataset(features, labels), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(43))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
    history, best, best_state, stale = [], None, None, 0
    for epoch in range(1, PROTOCOL['quantum']['max_epochs'] + 1):
        start = time.monotonic()
        model.train()
        total_loss = 0.
        for batch, targets in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
        result = quantum_metrics(model, val_features, val_labels)
        scheduler.step(result['loss'])
        row = {'epoch': epoch, 'train_loss': total_loss/len(labels), 'validation': result, 'seconds': time.monotonic()-start}
        history.append(row)
        score = (result['macro_f1'], result['accuracy'])
        if best is None or score > best:
            best, best_state, stale = score, snapshot(model), 0
            torch.save(best_state, OUT/'qxai_model.pth')
        else:
            stale += 1
        (OUT/'quantum_history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps({'event': 'epoch', 'kind': 'quantum', **row}), flush=True)
        if epoch >= PROTOCOL['quantum']['original_epochs'] and ((best[0] >= .85 and best[1] >= .85) or stale >= 8):
            break
    model.load_state_dict(best_state)
    model.eval()
    selected = max(history, key=lambda row: (row['validation']['macro_f1'], row['validation']['accuracy']))
    return {'selected_epoch': selected['epoch'], 'epochs_run': len(history), 'validation': selected['validation'], 'original_budget_validation': max(history[:5], key=lambda row: row['validation']['macro_f1'])['validation']}


def load_pair(directory, device):
    cnn = ClassicalCNN().to(device)
    cnn.load_state_dict(torch.load(directory/'xai_model.pth', map_location='cpu'), strict=True)
    cnn.eval()
    quantum = QuantumHybrid()
    quantum.load_state_dict(torch.load(directory/'qxai_model.pth', map_location='cpu'), strict=True)
    quantum.eval()
    return cnn, quantum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    manifest = json.loads((OUT/'split_manifest.json').read_text())
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise RuntimeError('Archive differs from locked split')
    protocol_path = OUT/'training_protocol.json'
    serialized = json.dumps(PROTOCOL, indent=2)+'\n'
    if protocol_path.exists() and protocol_path.read_text() != serialized:
        raise RuntimeError('Refusing to silently change prespecified protocol')
    protocol_path.write_text(serialized)
    seed(42)
    device = torch.device(('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device)
    start = time.monotonic()
    print(json.dumps({'event': 'started', 'stage': args.stage, 'device': str(device), 'test_name': manifest['test_name']}), flush=True)
    if args.stage == 'train':
        if (OUT/'selection.json').exists():
            raise RuntimeError('Selected candidate already exists; use a different experiment directory')
        cnn = ClassicalCNN(torch.load(DEFAULT_WEIGHTS, map_location='cpu')).to(device)
        classical = train_cnn(cnn, manifest, device)
        # Features are extracted only for training+validation; test remains sealed.
        train_features, train_labels = extract_features(cnn, manifest['splits']['train'], device)
        val_features, val_labels = extract_features(cnn, manifest['splits']['validation'], device)
        seed(43)
        quantum = QuantumHybrid()
        qinfo = train_quantum(quantum, train_features, train_labels, val_features, val_labels)
        selection = {'classical': classical, 'quantum': qinfo, 'runtime_seconds': time.monotonic()-start, 'device': str(device),
                     'test_evaluated': False, 'checkpoint_sha256': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['xai_model.pth', 'qxai_model.pth']}}
        (OUT/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
        print(json.dumps({'event': 'selection_locked', 'selection': selection}), flush=True)
    else:
        if (OUT/'evaluation.json').exists():
            raise RuntimeError('Test already evaluated; do not repeatedly probe it')
        selection = json.loads((OUT/'selection.json').read_text())
        for name, expected in selection['checkpoint_sha256'].items():
            if hashlib.sha256((OUT/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Candidate changed after selection was locked')
        cnn, quantum = load_pair(OUT, device)
        test_features, test_labels = extract_features(cnn, manifest['splits']['test'], device)
        with torch.no_grad():
            predictions = cnn.classifier(test_features.to(device)).argmax(1).cpu().numpy()
        candidate = {'classical': metrics(test_labels.numpy(), predictions), 'quantum': quantum_metrics(quantum, test_features, test_labels)}
        del cnn, quantum
        old_cnn, old_quantum = load_pair(ROOT/'backend', device)
        old_features, old_labels = extract_features(old_cnn, manifest['splits']['test'], device)
        with torch.no_grad():
            old_pred = old_cnn.classifier(old_features.to(device)).argmax(1).cpu().numpy()
        baseline = {'classical': metrics(old_labels.numpy(), old_pred), 'quantum': quantum_metrics(old_quantum, old_features, old_labels)}
        report = {
            'model_version': 'cleaned-v3-seed42', 'dataset_url': manifest['dataset_url'], 'dataset_version': manifest['dataset_version'],
            'archive_sha256': manifest['archive_sha256'], 'test_name': manifest['test_name'], 'test_protocol': manifest['split_policy'],
            'seed': 42, 'classes': CLASSES, 'candidate_test': candidate, 'old_checkpoint_test': baseline,
            'candidate_validation': {kind: selection[kind]['validation'] for kind in ['classical','quantum']},
            'training': selection, 'training_protocol': PROTOCOL, 'checkpoint_sha256': selection['checkpoint_sha256'],
            'baseline_checkpoint_sha256': {name: hashlib.sha256((ROOT/'backend'/name).read_bytes()).hexdigest() for name in ['xai_model.pth','qxai_model.pth']},
            'candidate_preprocessing': {'version': 'resize224-rgb-totensor-v1', 'rgb': True, 'resize': [224,224], 'interpolation': 'torchvision PIL bilinear', 'normalization': False},
            'test_counts': manifest['counts']['test'], 'source_overlap': manifest['source_overlap'],
            'target_80_each_model_met': all(result['accuracy'] >= .8 for result in candidate.values()),
            'limitations': ['Random image-level split; no patient IDs; patient-independent testing is not established.',
                           'New dataset/holdout is not directly comparable to v1 supplied Kaggle test results.',
                           'Previous project checkpoints used overlapping source images. Fresh v3 candidates exclude locked test images; old-checkpoint baseline may include training images.',
                           'Softmax scores are uncalibrated. No clinical validation or quantum advantage established.',
                           'Dataset license metadata: Unknown. Raw archives/images/caches excluded from release packaging.',
                           'MPS kernels are not guaranteed bitwise deterministic across platforms.'],
            'versions': {name: importlib.metadata.version(name) for name in ['torch','torchvision','PennyLane','numpy','Pillow']},
            'python': sys.version.split()[0], 'evaluation_device': str(device), 'evaluation_runtime_seconds': time.monotonic()-start,
        }
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'event': 'test_complete', 'candidate_test': candidate, 'target_met': report['target_80_each_model_met']}), flush=True)


if __name__ == '__main__':
    main()
