"""Create a deployment manifest, model card and plots from a completed training run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'models/candidates/kaggle-v1-seed183')
    args = parser.parse_args()
    output = args.output
    report = json.loads((output / 'evaluation.json').read_text())
    report['versions'].update({name: importlib.metadata.version(name) for name in ('torchvision', 'PennyLane', 'Pillow', 'scipy', 'autoray')})
    report['versions']['python'] = sys.version.split()[0]
    provenance_path = ROOT / 'training/run_provenance.json'
    if provenance_path.exists():
        report['provenance'] = json.loads(provenance_path.read_text())
    split = json.loads((output / 'split_manifest.json').read_text())
    overlap_path = output / 'source_overlap.json'
    if overlap_path.exists():
        report['source_archive_overlap'] = json.loads(overlap_path.read_text())
    report['dataset_license'] = {
        'kaggle_metadata': 'Other (specified in description)',
        'upstream_metadata': 'Unknown',
        'redistribution_rights': 'Not established; dataset archives/images are excluded from release packaging.',
    }
    report['circuit'] = {
        'qubits': 4, 'layers': 4, 'encoding': 'AngleEmbedding rotation X (default)',
        'entangler': 'BasicEntanglerLayers default RX with CNOT ring',
        'readout': 'Four Pauli-Z expectations', 'device': 'default.qubit CPU simulator',
    }
    report['optimizer'] = {
        'algorithm': 'AdamW', 'classical_learning_rate': 0.001, 'quantum_learning_rate': 0.003,
        'weight_decay': 0.001, 'batch_size': 64, 'class_weights': 'inverse training class frequency',
        'gradient_norm_clip': 5, 'early_stopping_patience': 12,
        'scheduler': 'ReduceLROnPlateau validation macro F1, factor 0.5, patience 5',
    }
    report['split_summary'] = {key: value for key, value in split.items() if key not in ('splits', 'invalid_images')}
    report['validation_protocol'] = 'Only validation macro F1 is used for checkpoint selection. Final test evaluated after both selections.'
    report['comparison_interpretation'] = {
        'aggregate': 'Candidates improve overall accuracy and macro F1 compared with supplied original checkpoints on this test set.',
        'class_tradeoff': 'Candidate AD and PD recall are lower; CONTROL recall improves substantially. Original checkpoints are strongly biased toward AD on this test set.',
        'validation_test_gap': 'Candidate held-out test macro F1 is substantially lower than validation macro F1; overfitting and/or supplied-split distribution shift remains unresolved.',
        'promotion': 'Research candidates only. Do not claim reliable diagnosis or uniformly improved class performance; retain originals for explicit comparison.',
    }
    calibration_note = 'Softmax scores are uncalibrated model outputs; saliency maps do not establish a clinical explanation.'
    if calibration_note not in report['limitations']:
        report['limitations'].append(calibration_note)
    for group in ('candidate_test', 'old_checkpoint_test'):
        for result in report[group].values():
            for item in result['per_class'].values():
                n, p, z = item['support'], item['recall'], 1.959963984540054
                denominator = 1 + z*z/n
                center = (p + z*z/(2*n))/denominator
                half_width = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
                item['recall_wilson_95_interval'] = [max(0, center-half_width), min(1, center+half_width)]
    report['interval_limitation'] = 'Wilson intervals assume independent images; unknown subject clustering can make these intervals too narrow.'
    report['checkpoint_sha256'] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ('xai_model.pth', 'qxai_model.pth')
    }
    report['baseline_checkpoint_sha256'] = {
        name: hashlib.sha256((ROOT / 'backend' / name).read_bytes()).hexdigest()
        for name in ('xai_model.pth', 'qxai_model.pth')
    }
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    preprocessing = report['candidate_preprocessing']
    manifest = {
        'model_version': output.name,
        'preprocessing_version': preprocessing['version'],
        'class_names': report['classes'],
        'mean': preprocessing['normalize_mean'], 'std': preprocessing['normalize_std'], 'image_size': 224,
        'checkpoints': {
            'classical': {'file': 'xai_model.pth', 'sha256': report['checkpoint_sha256']['xai_model.pth']},
            'quantum': {'file': 'qxai_model.pth', 'sha256': report['checkpoint_sha256']['qxai_model.pth']},
        },
        'architecture': {'classical': 'efficientnet_b0_frozen_1280_256_3', 'quantum': 'encoder1280_4_tanh_angle_x_entangler4_z4_64_3'},
        'circuit': report['circuit'], 'dataset_url': report['dataset_url'],
        'archive_sha256': report['archive_sha256'], 'seed': report['seed'],
        'validation_selection': report['training'], 'held_out_metrics': report['candidate_test'],
        'limitations': report['limitations'], 'evaluation_file': 'evaluation.json', 'research_only': True,
    }
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    rows = []
    for group, title in [('old_checkpoint_test', 'Original'), ('candidate_test', 'Candidate')]:
        for kind, label in [('classical', 'XAI'), ('quantum', 'QXAI')]:
            result = report[group][kind]
            rows.append(f"| {title} {label} | {result['accuracy']:.1%} | {result['balanced_accuracy']:.1%} | {result['macro_f1']:.3f} | {result['per_class']['AD']['recall']:.1%} |")
    class_rows = []
    for kind, label in [('classical', 'XAI'), ('quantum', 'QXAI')]:
        for name in report['classes']:
            row = report['candidate_test'][kind]['per_class'][name]
            class_rows.append(f"| {label} | {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    training_rows = []
    for kind, label in [('classical', 'XAI'), ('quantum', 'QXAI')]:
        result = report['training'][kind]
        training_rows.append(f"- {label}: selected epoch {result['selected_epoch']} of {result['epochs_run']} completed; validation macro F1 {result['best_validation_macro_f1']:.4f}.")
    card = f"""# QXAI research candidates: {output.name}

These checkpoints were actually trained locally on the fresh Kaggle archive, using the architecture from Joshua Asirvatham's QXAI training project. They are image-classification research artifacts, not clinically validated diagnostic tools. There is no evidence of quantum advantage.

## Held-out image-level results

The same 1,362 retained supplied-test images are used for each row. Original weights use their own saved backbone and original unnormalized input; candidates use the frozen official ImageNet backbone and the normalized input recorded below. Original training membership is unknown and may overlap test images, so the original-checkpoint comparison is a diagnostic baseline.

| Model | Accuracy | Balanced accuracy | Macro F1 | AD recall |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

**Interpretation:** aggregate accuracy and macro F1 improve, but the candidates have lower AD and PD recall. The original models predict AD for most CONTROL images; their very high AD recall reflects this class bias. The new checkpoints are not uniformly better. Test macro F1 is substantially below validation macro F1, indicating unresolved overfitting and/or a difference between the supplied train and test distributions. Do not present these candidates as a fix for diagnostic accuracy.

| Candidate | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

Confusion matrices (rows true, columns predicted): `confusion_matrices.png`. Complete machine-readable metrics: `evaluation.json`.

## Data and split

Source: {report['dataset_url']}. Fresh official Kaggle public API download, dataset version 1; recorded archive SHA256 `{report['archive_sha256']}`.

Original archive: {split['source_images']:,} images. Retained: {split['retained_images']:,}. Train 5,486 (AD 2,177; CONTROL 2,548; PD 761), validation 968 (AD 384; CONTROL 450; PD 134), test 1,362 (AD 639; CONTROL 662; PD 61). The supplied test split is preserved. Canonical RGB 224×224 image hashes group exact matches; test membership wins cross-split duplicates. Removed {split['duplicate_images_removed']} duplicate copies, including {split['train_images_removed_for_test_overlap']} training copies overlapping test. Contradictory-label groups excluded: {split['contradictory_groups_excluded']}. Stratified validation fraction 15%, seed {report['seed']}.

No patient/subject identifiers are supplied. Patient independence is not established. Exact hashing does not detect near duplicates; acquisition differences and dataset construction can confound performance, especially PD versus AD/CONTROL. These results do not establish generalization to arbitrary uploads, another scanner, another dataset, or a clinical population. No private uploads were used for training or evaluation.

Kaggle lists the license as **Other (specified in description)**. Its description links to an upstream dataset whose license metadata is **Unknown**. No additional permission terms were found in those metadata. Redistribution rights are not established; raw images, archives and feature caches are excluded from release packaging.

## Training and selection

EfficientNet-B0 ImageNet1K_V1 parameters and BatchNorm buffers are frozen in evaluation mode. Both heads train on the same 1,280-dimensional cached features. AdamW, inverse-frequency class weights, batch 64, gradient norm clipping 5; initial learning rates 0.001 XAI and 0.003 QXAI; weight decay 0.001. Learning-rate scheduling and checkpoint selection use validation macro F1 only; early stopping patience 12. No test metrics select checkpoints.

{chr(10).join(training_rows)}

QXAI: trainable 1280→4 encoder with tanh; four simulated qubits; default X AngleEmbedding; four default BasicEntanglerLayers; Pauli-Z readouts; trainable 4→64→3 head. PennyLane batch broadcasting was checked against the original per-image circuit, including finite gradients. State-dict keys remain compatible with the original backend architecture.

## Deployment contract

Use **both** candidate weights together and select preprocessing `{preprocessing['version']}`: RGB, PIL bilinear resize to 224×224, ToTensor [0,1], mean `{preprocessing['normalize_mean']}`, standard deviation `{preprocessing['normalize_std']}`. No crop or aspect-ratio preservation. Class index order: `{report['classes']}`.

`model_manifest.json` records exact hashes. Never mix an old CNN backbone with the new QXAI head. Never normalize inputs for original weights. Existing `backend/` checkpoints have been preserved; promotion is controlled separately by the application.

Reproduction: `training/README.md`, `training/train.py`, saved `split_manifest.json`, and epoch history files. CPU only, no paid services. Total recorded training/evaluation runtime: {report['runtime_seconds'] / 60:.1f} minutes. Runtime package versions: `{report['versions']}`.
"""
    (output / 'MODEL_CARD.md').write_text(card)
    plot_matrices(output, report)
    print(json.dumps({'manifest': str(output / 'model_manifest.json'), 'candidate_test': report['candidate_test']}, indent=2))


def plot_matrices(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for index, (group, title) in enumerate([('old_checkpoint_test', 'Original'), ('candidate_test', 'Candidate')]):
        for column, (kind, label) in enumerate([('classical', 'XAI'), ('quantum', 'QXAI')]):
            ax = axes[index, column]
            matrix = np.asarray(report[group][kind]['confusion_matrix'])
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=662)
            for row in range(3):
                for col in range(3):
                    ax.text(col, row, str(matrix[row, col]), ha='center', va='center', color='white' if matrix[row, col] > 331 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=report['classes'], yticklabels=report['classes'], xlabel='Predicted class', ylabel='True class', title=f'{title} {label}')
    fig.suptitle('Supplied held-out test: image-level research results\nNo patient independence or clinical validation established')
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(output / 'confusion_matrices.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
