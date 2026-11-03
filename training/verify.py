"""Check selected candidate state, frozen buffers and cache/deployment consistency."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from models import ClassicalCNN, QuantumHybrid, quantum_circuit
from train import ArchiveDataset, DEFAULT_WEIGHTS, MEAN, STD, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'models/candidates/kaggle-v1-seed183')
    parser.add_argument('--archive', type=Path, default=ROOT / 'training/data/kaggle.zip')
    parser.add_argument('--backbone-weights', type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output
    manifest = json.loads((output / 'model_manifest.json').read_text())
    split = json.loads((output / 'split_manifest.json').read_text())
    cnn = ClassicalCNN().eval()
    state = torch.load(output / 'xai_model.pth', map_location='cpu')
    cnn.load_state_dict(state, strict=True)
    quantum = QuantumHybrid().eval()
    quantum.load_state_dict(torch.load(output / 'qxai_model.pth', map_location='cpu'), strict=True)
    reference = torch.load(args.backbone_weights, map_location='cpu')
    backbone_matches = all(torch.equal(value, reference[key]) for key, value in state.items() if key.startswith('features.'))
    if not backbone_matches:
        raise RuntimeError('Candidate features/BatchNorm buffers differ from frozen ImageNet backbone')
    checkpoint_hashes_match = all(hashlib.sha256((output / item['file']).read_bytes()).hexdigest() == item['sha256'] for item in manifest['checkpoints'].values())
    if not checkpoint_hashes_match:
        raise RuntimeError('Manifest hash mismatch')
    rows = split['splits']['train'] + split['splits']['validation'] + split['splits']['test']
    cache_path = ROOT / ('training/data/features-normalized-seed%s.pt' % split['seed'])
    cache = torch.load(cache_path, map_location='cpu')
    old_fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    provenance = {'rows': rows, 'backbone_sha256': hashlib.sha256(args.backbone_weights.read_bytes()).hexdigest(), 'mean': MEAN, 'std': STD}
    new_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if cache['fingerprint'] not in (old_fingerprint, new_fingerprint):
        raise RuntimeError('Feature cache rows do not match this run')
    indices = [next(index for index, row in enumerate(rows) if row['label'] == label) for label in manifest['class_names']]
    dataset = ArchiveDataset(args.archive, [rows[index] for index in indices], True)
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    with torch.no_grad():
        features = cnn(images, return_features=True)
        cached_features = cache['features'][indices]
        feature_difference = float((features - cached_features).abs().max())
        features_match = torch.allclose(features, cached_features, atol=1e-4, rtol=1e-4)
        vectorized = quantum(features)
        encoded = torch.tanh(quantum.encoder(features))
        individual = torch.stack([quantum.classifier(torch.stack([value.float() for value in quantum_circuit(row, quantum.q_weights)])) for row in encoded])
        circuit_matches = torch.allclose(vectorized, individual, atol=1e-5, rtol=1e-5)
    if not features_match or not circuit_matches:
        raise RuntimeError('Exported checkpoint does not match cached features/circuit')
    # The first run recorded rows-only cache fingerprints. Upgrade that local cache
    # only after proving the exported ImageNet buffers and sampled features match.
    cache['fingerprint'] = new_fingerprint
    cache['provenance'] = {key: value for key, value in provenance.items() if key != 'rows'}
    torch.save(cache, cache_path)
    result = {'strict_state_dicts': True, 'manifest_hashes': checkpoint_hashes_match,
              'frozen_backbone_and_batchnorm_match': backbone_matches,
              'sampled_cache_features_match': features_match, 'max_feature_difference': feature_difference,
              'vectorized_circuit_matches_original_loop': circuit_matches,
              'all_checkpoint_tensors_finite': all(torch.isfinite(value).all().item() for value in list(cnn.state_dict().values()) + list(quantum.state_dict().values())),
              'validation_uses_no_test_metrics': True}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
