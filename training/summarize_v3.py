"""Publish the locked cleaned-v3 experiment's aggregate report and model contract."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/candidates/cleaned-v3-seed42'


def main():
    report = json.loads((OUT/'evaluation.json').read_text())
    split = json.loads((OUT/'split_manifest.json').read_text())
    manifest = {
        'model_version': report['model_version'], 'preprocessing_version': 'resize224-rgb-totensor-v1',
        'class_names': report['classes'], 'image_size': 224, 'mean': None, 'std': None,
        'checkpoints': {kind: {'file': filename, 'sha256': hashlib.sha256((OUT/filename).read_bytes()).hexdigest()}
                        for kind, filename in [('classical','xai_model.pth'),('quantum','qxai_model.pth')]},
        'test_name': report['test_name'], 'test_protocol': report['test_protocol'],
        'dataset_url': report['dataset_url'], 'dataset_version': report['dataset_version'],
        'archive_sha256': report['archive_sha256'], 'seed': report['seed'],
        'held_out_metrics': report['candidate_test'], 'target_80_each_model_met': report['target_80_each_model_met'],
        'backbone_training': report['training_protocol']['backbone'],
        'research_only': True, 'limitations': report['limitations'], 'evaluation_file': 'evaluation.json',
    }
    (OUT/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    class_rows = []
    for kind, label in [('classical','XAI'),('quantum','QXAI')]:
        result = report['candidate_test'][kind]
        selection = report['training'][kind]
        rows.append(f"| {label} | {result['accuracy']:.2%} | {result['balanced_accuracy']:.2%} | {result['macro_f1']:.4f} | {selection['validation']['accuracy']:.2%} | {selection['selected_epoch']}/{selection['epochs_run']} |")
        for name in report['classes']:
            value = result['per_class'][name]
            class_rows.append(f"| {label} | {name} | {value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} | {value['support']} |")
    outcome = 'Both candidates meet the requested 80% held-out image accuracy target.' if report['target_80_each_model_met'] else 'The requested 80% target is not met by both candidates; do not claim otherwise or repeatedly retrain against these test results.'
    card = f"""# Cleaned v3: original-architecture research candidates

{outcome} These are image-level research results, not patient-independent or clinically validated diagnoses.

## Locked held-out results

Test identifier: `{report['test_name']}`. The test has **407 images**: AD 200, CONTROL 155, PD 52. Test inference ran only after training and validation checkpoint selection were complete; it did not choose epochs or learning rates.

| Candidate | Test accuracy | Balanced accuracy | Macro F1 | Selected validation accuracy | Selected/completed epoch |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| Candidate | Class | Precision | Recall | F1 | Test support |
|---|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

This is **not the supplied-test benchmark used by candidate v1**. Its 72.9% XAI / 68.7% QXAI results remain preserved in `kaggle-v1-seed183/`. Changing dataset and holdout changes what the numbers measure; the two result pairs are not a paired improvement comparison. The original GitHub README's 89.46% / 86.52% used a random holdout from its 2,024-image archive, also a different exact test set. `evaluation.json` includes existing-checkpoint scores only as a diagnostic baseline: those weights have unknown original training membership and may already have trained on cleaned-v3 test images.

## Source and split

Dataset: {report['dataset_url']}; official Kaggle **version 12** (the dataset slug contains v3), last updated 2025-07-11. Fresh public API archive SHA256 `{report['archive_sha256']}`. The metadata license is **Unknown**; raw archives, images and feature caches stay outside release packaging.

Only `3_cls_2/train/` is used: AD 1,000 / CONTROL 767 / PD 254 = 2,021 images. The 1,767 `trainbinary/` copies are excluded. No supplied test folder exists. Canonical RGB bilinear 224×224 hashes group exact duplicates before stratified splitting. Found {split['duplicate_copies_removed']} duplicate copies and {split['contradictory_groups_excluded']} contradictory-label groups. NumPy seed 42, original class/path order, floor(70%) training / floor(10%) validation / remainder test yields:

- Training 1,413: AD 700, CONTROL 536, PD 177.
- Validation 201: AD 100, CONTROL 76, PD 25.
- Test 407: AD 200, CONTROL 155, PD 52.

The split manifest was saved before training; cross-partition hashes are disjoint. All 2,021 cleaned-v3 images also occur in the original GitHub archive. This run starts fresh from ImageNet, and its locked test is excluded from its training and selection. Earlier project models used overlapping source images (343 of this test's images were in v1 train and 64 in v1 validation), so this is **not a globally never-used project dataset**. No subject identifiers are provided; near duplicates and patient overlap cannot be ruled out. No private uploads were used.

## Architecture and faithful original behavior

The XAI and QXAI state-dict schemas and architecture match executable `QXAI-prj/main.py`. XAI: EfficientNet-B0 → global pooling → 1280 → 256 ReLU/dropout 0.5 → 3. QXAI uses the same 1280-feature backbone → linear 4 + tanh → four X AngleEmbedding qubits → four BasicEntanglerLayers → four Pauli-Z readouts → 4 → 64 ReLU → 3. The original README describes Ry but its executable circuit uses default X rotations; this run follows the code. PennyLane is a CPU quantum simulator, not quantum hardware, and does not establish quantum advantage.

ImageNet backbone **parameters are frozen**. During CNN training, `model.train()` updates BatchNorm running buffers and activates stochastic depth, matching the original code. The selected CNN is then put in eval mode and supplies fixed features for QXAI. This deliberately differs from v1's completely frozen eval-mode ImageNet backbone. No backbone convolution parameters were fine-tuned in this faithful run.

Separate train/validation dataset objects prevent the original shared-Subset transform bug. No image augmentation is enabled, matching what the original code actually executes after its transform assignments overwrite each other. Raw image input: RGB → PIL bilinear 224×224 → ToTensor [0,1], **no channel normalization**. Class order: AD, CONTROL, PD. Adam settings, batch 16, and original 10 CNN / 5 QML epoch budgets are recorded in `training_protocol.json`. The prespecified validation-only continuation permits at most 20 / 30 epochs if validation accuracy or macro F1 remains below 0.85, stopping after eight stale epochs. Actual completed/selected epochs are in the table. Highest validation macro F1 selects checkpoints; accuracy breaks ties. The original ineffective early-stopping helper is replaced by this saved protocol.

Training used the local `{report['training']['device']}` device for CNN computation, with fixed Python/NumPy/PyTorch seed 42 and QML seed 43; quantum simulation stays on CPU. MPS results may not be bitwise identical across runs/platforms. Recorded training runtime: {report['training']['runtime_seconds']/60:.1f} minutes; evaluation runtime: {report['evaluation_runtime_seconds']/60:.1f} minutes. No paid services were used.

## Deployment contract

Load **both** files in this directory using `model_manifest.json` and verify their SHA256 hashes. Set model directory explicitly. Select preprocessing `resize224-rgb-totensor-v1` with mean/std null; do not apply v1 candidate ImageNet normalization to these weights. Selected BatchNorm buffers are part of `xai_model.pth` and are essential to the QXAI feature distribution. Original backend weights are preserved separately. Softmax scores are uncalibrated; these aggregate results cannot establish reliable diagnosis for arbitrary uploads or a clinical population.

Reproduction: `training/audit_v3.py`, `training/train_v3.py --stage train`, then a single `--stage evaluate`; `training/PROTOCOL_AUDIT.md` explains differences from the first candidate and original claims.
"""
    (OUT/'MODEL_CARD.md').write_text(card)
    plot(report)
    print(json.dumps({'manifest': str(OUT/'model_manifest.json'), 'target_met': report['target_80_each_model_met']}, indent=2))


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1,2,figsize=(10,4.6))
    for ax, (kind,label) in zip(axes,[('classical','XAI'),('quantum','QXAI')]):
        matrix = np.asarray(report['candidate_test'][kind]['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=200)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>100 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=report['classes'],yticklabels=report['classes'],xlabel='Predicted class',ylabel='True class',title=label)
    fig.suptitle('Cleaned v3 version 12: locked 407-image holdout\nImage-level research; patient independence not established')
    fig.tight_layout(rect=[0,0,1,.9])
    fig.savefig(OUT/'confusion_matrices.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
