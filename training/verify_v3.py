"""Verify the GPU-trained raw-input candidate in the original CPU app runtime."""
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from models import ClassicalCNN, QuantumHybrid, quantum_circuit
from train import ArchiveDataset, DEFAULT_WEIGHTS, ROOT

OUT = ROOT / 'models/candidates/cleaned-v3-seed42'


def main():
    torch.set_num_threads(1)
    manifest = json.loads((OUT/'model_manifest.json').read_text())
    split = json.loads((OUT/'split_manifest.json').read_text())
    selected = json.loads((OUT/'selection.json').read_text())
    cnn = ClassicalCNN().eval()
    state = torch.load(OUT/'xai_model.pth', map_location='cpu')
    cnn.load_state_dict(state, strict=True)
    quantum = QuantumHybrid().eval()
    quantum.load_state_dict(torch.load(OUT/'qxai_model.pth', map_location='cpu'), strict=True)
    reference = torch.load(DEFAULT_WEIGHTS, map_location='cpu')
    parameters_frozen = all(torch.equal(value, reference[key]) for key, value in state.items() if key.startswith('features.') and key.endswith(('.weight','.bias')))
    bn_changed = any(not torch.equal(value, reference[key]) for key, value in state.items() if key.endswith(('running_mean','running_var')))
    expected_updates = selected['classical']['selected_epoch'] * math.ceil(len(split['splits']['train'])/16)
    counters_match = all(int(value)-int(reference[key]) == expected_updates for key,value in state.items() if key.endswith('num_batches_tracked'))
    hashes_match = all(hashlib.sha256((OUT/item['file']).read_bytes()).hexdigest() == item['sha256'] for item in manifest['checkpoints'].values())
    torch.manual_seed(42)
    features = torch.randn(5,1280)
    with torch.no_grad():
        batch = quantum(features)
        encoded = torch.tanh(quantum.encoder(features))
        individual = torch.stack([quantum.classifier(torch.stack([value.float() for value in quantum_circuit(row,quantum.q_weights)])) for row in encoded])
    circuit_matches = torch.allclose(batch,individual,atol=1e-5,rtol=1e-5)
    rows = [next(row for row in split['splits']['validation'] if row['label']==label) for label in manifest['class_names']]
    dataset = ArchiveDataset(ROOT/'training/data/cleaned-v3.zip', rows, False)
    image = dataset[0][0].unsqueeze(0).requires_grad_(True)
    cnn.zero_grad();quantum.zero_grad()
    logits = quantum(cnn(image, return_features=True))
    logits.max().backward()
    saliency_finite = image.grad is not None and torch.isfinite(image.grad).all().item()
    sys.path.insert(0,str(ROOT/'backend'))
    from model_bundle import load_bundle
    loaded = load_bundle(OUT)
    checks = {
        'strict_state_dicts': True, 'manifest_hashes': hashes_match,
        'backbone_parameters_equal_imagenet': parameters_frozen,
        'batchnorm_buffers_adapted': bn_changed,
        'batchnorm_update_counters_match_selected_epoch': counters_match,
        'expected_training_batches_at_checkpoint': expected_updates,
        'vectorized_quantum_matches_original_loop': circuit_matches,
        'cpu_end_to_end_quantum_saliency_finite': saliency_finite,
        'checkpoint_tensors_finite': all(torch.isfinite(value).all().item() for value in list(state.values())+list(quantum.state_dict().values())),
        'backend_bundle_loader_passed': loaded['preprocessing_version']=='resize224-rgb-totensor-v1',
        'cpu_verification_torch_version': torch.__version__,
        'uses_test_images': False,
    }
    for key in ['manifest_hashes','backbone_parameters_equal_imagenet','batchnorm_buffers_adapted','batchnorm_update_counters_match_selected_epoch','vectorized_quantum_matches_original_loop','cpu_end_to_end_quantum_saliency_finite','checkpoint_tensors_finite','backend_bundle_loader_passed']:
        if not checks[key]:
            raise RuntimeError('Verification failed: '+key)
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
