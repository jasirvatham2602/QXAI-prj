# QXAI research candidates: kaggle-v1-seed183

These checkpoints were actually trained locally on the fresh Kaggle archive, using the architecture from Joshua Asirvatham's QXAI training project. They are image-classification research artifacts, not clinically validated diagnostic tools. There is no evidence of quantum advantage.

## Held-out image-level results

The same 1,362 retained supplied-test images are used for each row. Original weights use their own saved backbone and original unnormalized input; candidates use the frozen official ImageNet backbone and the normalized input recorded below. Original training membership is unknown and may overlap test images, so the original-checkpoint comparison is a diagnostic baseline.

| Model | Accuracy | Balanced accuracy | Macro F1 | AD recall |
|---|---:|---:|---:|---:|
| Original XAI | 52.2% | 62.3% | 0.505 | 99.8% |
| Original QXAI | 53.6% | 63.2% | 0.529 | 98.7% |
| Candidate XAI | 72.9% | 64.7% | 0.689 | 66.7% |
| Candidate QXAI | 68.7% | 62.2% | 0.661 | 62.4% |

**Interpretation:** aggregate accuracy and macro F1 improve, but the candidates have lower AD and PD recall. The original models predict AD for most CONTROL images; their very high AD recall reflects this class bias. The new checkpoints are not uniformly better. Test macro F1 is substantially below validation macro F1, indicating unresolved overfitting and/or a difference between the supplied train and test distributions. Do not present these candidates as a fix for diagnostic accuracy.

| Candidate | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| XAI | AD | 0.782 | 0.667 | 0.720 | 639 |
| XAI | CONTROL | 0.687 | 0.814 | 0.745 | 662 |
| XAI | PD | 0.875 | 0.459 | 0.602 | 61 |
| QXAI | AD | 0.729 | 0.624 | 0.673 | 639 |
| QXAI | CONTROL | 0.651 | 0.767 | 0.705 | 662 |
| QXAI | PD | 0.829 | 0.475 | 0.604 | 61 |

Confusion matrices (rows true, columns predicted): `confusion_matrices.png`. Complete machine-readable metrics: `evaluation.json`.

## Data and split

Source: https://www.kaggle.com/datasets/farjanakabirsamanta/alzheimer-diseases-3-class. Fresh official Kaggle public API download, dataset version 1; recorded archive SHA256 `57a55eb74a4128e369b46c1fe1214bc35c379bb72ad619c53500c62ee43597e3`.

Original archive: 7,839 images. Retained: 7,816. Train 5,486 (AD 2,177; CONTROL 2,548; PD 761), validation 968 (AD 384; CONTROL 450; PD 134), test 1,362 (AD 639; CONTROL 662; PD 61). The supplied test split is preserved. Canonical RGB 224×224 image hashes group exact matches; test membership wins cross-split duplicates. Removed 23 duplicate copies, including 11 training copies overlapping test. Contradictory-label groups excluded: 0. Stratified validation fraction 15%, seed 183.

No patient/subject identifiers are supplied. Patient independence is not established. Exact hashing does not detect near duplicates; acquisition differences and dataset construction can confound performance, especially PD versus AD/CONTROL. These results do not establish generalization to arbitrary uploads, another scanner, another dataset, or a clinical population. No private uploads were used for training or evaluation.

Kaggle lists the license as **Other (specified in description)**. Its description links to an upstream dataset whose license metadata is **Unknown**. No additional permission terms were found in those metadata. Redistribution rights are not established; raw images, archives and feature caches are excluded from release packaging.

## Training and selection

EfficientNet-B0 ImageNet1K_V1 parameters and BatchNorm buffers are frozen in evaluation mode. Both heads train on the same 1,280-dimensional cached features. AdamW, inverse-frequency class weights, batch 64, gradient norm clipping 5; initial learning rates 0.001 XAI and 0.003 QXAI; weight decay 0.001. Learning-rate scheduling and checkpoint selection use validation macro F1 only; early stopping patience 12. No test metrics select checkpoints.

- XAI: selected epoch 79 of 80 completed; validation macro F1 0.9366.
- QXAI: selected epoch 33 of 45 completed; validation macro F1 0.8443.

QXAI: trainable 1280→4 encoder with tanh; four simulated qubits; default X AngleEmbedding; four default BasicEntanglerLayers; Pauli-Z readouts; trainable 4→64→3 head. PennyLane batch broadcasting was checked against the original per-image circuit, including finite gradients. State-dict keys remain compatible with the original backend architecture.

## Deployment contract

Use **both** candidate weights together and select preprocessing `resize224-rgb-imagenet-normalized-v2`: RGB, PIL bilinear resize to 224×224, ToTensor [0,1], mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`. No crop or aspect-ratio preservation. Class index order: `['AD', 'CONTROL', 'PD']`.

`model_manifest.json` records exact hashes. Never mix an old CNN backbone with the new QXAI head. Never normalize inputs for original weights. Existing `backend/` checkpoints have been preserved; promotion is controlled separately by the application.

Reproduction: `training/README.md`, `training/train.py`, saved `split_manifest.json`, and epoch history files. CPU only, no paid services. Total recorded training/evaluation runtime: 26.6 minutes. Runtime package versions: `{'torch': '1.13.0', 'numpy': '1.24.3', 'torchvision': '0.14.0', 'PennyLane': '0.38.0', 'Pillow': '11.3.0', 'scipy': '1.13.1', 'autoray': '0.7.0', 'python': '3.9.6'}`.
