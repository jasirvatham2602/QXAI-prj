# Cleaned v3: original-architecture research candidates

Both candidates meet the requested 80% held-out image accuracy target. These are image-level research results, not patient-independent or clinically validated diagnoses.

## Locked held-out results

Test identifier: `cleaned-v3-version12-grouped-random-image-holdout-seed42`. The test has **407 images**: AD 200, CONTROL 155, PD 52. Test inference ran only after training and validation checkpoint selection were complete; it did not choose epochs or learning rates.

| Candidate | Test accuracy | Balanced accuracy | Macro F1 | Selected validation accuracy | Selected/completed epoch |
|---|---:|---:|---:|---:|---:|
| XAI | 87.22% | 89.54% | 0.8987 | 88.06% | 9/10 |
| QXAI | 86.24% | 89.07% | 0.8922 | 89.05% | 5/5 |

| Candidate | Class | Precision | Recall | F1 | Test support |
|---|---|---:|---:|---:|---:|
| XAI | AD | 0.833 | 0.925 | 0.877 | 200 |
| XAI | CONTROL | 0.887 | 0.761 | 0.819 | 155 |
| XAI | PD | 1.000 | 1.000 | 1.000 | 52 |
| QXAI | AD | 0.843 | 0.885 | 0.863 | 200 |
| QXAI | CONTROL | 0.841 | 0.787 | 0.813 | 155 |
| QXAI | PD | 1.000 | 1.000 | 1.000 | 52 |

This is **not the supplied-test benchmark used by candidate v1**. Its 72.9% XAI / 68.7% QXAI results remain preserved in `kaggle-v1-seed183/`. Changing dataset and holdout changes what the numbers measure; the two result pairs are not a paired improvement comparison. The original GitHub README's 89.46% / 86.52% used a random holdout from its 2,024-image archive, also a different exact test set. `evaluation.json` includes existing-checkpoint scores only as a diagnostic baseline: those weights have unknown original training membership and may already have trained on cleaned-v3 test images.

## Source and split

Dataset: https://www.kaggle.com/datasets/jasirvatham2602/classifying-ad-and-pd-v3; official Kaggle **version 12** (the dataset slug contains v3), last updated 2025-07-11. Fresh public API archive SHA256 `ffe22a860b3592d1a59bd0ffadcb2c38f6a026345b16a28ea0204d0729e036fb`. The metadata license is **Unknown**; raw archives, images and feature caches stay outside release packaging.

Only `3_cls_2/train/` is used: AD 1,000 / CONTROL 767 / PD 254 = 2,021 images. The 1,767 `trainbinary/` copies are excluded. No supplied test folder exists. Canonical RGB bilinear 224×224 hashes group exact duplicates before stratified splitting. Found 0 duplicate copies and 0 contradictory-label groups. NumPy seed 42, original class/path order, floor(70%) training / floor(10%) validation / remainder test yields:

- Training 1,413: AD 700, CONTROL 536, PD 177.
- Validation 201: AD 100, CONTROL 76, PD 25.
- Test 407: AD 200, CONTROL 155, PD 52.

The split manifest was saved before training; cross-partition hashes are disjoint. All 2,021 cleaned-v3 images also occur in the original GitHub archive. This run starts fresh from ImageNet, and its locked test is excluded from its training and selection. Earlier project models used overlapping source images (343 of this test's images were in v1 train and 64 in v1 validation), so this is **not a globally never-used project dataset**. No subject identifiers are provided; near duplicates and patient overlap cannot be ruled out. No private uploads were used.

## Architecture and faithful original behavior

The XAI and QXAI state-dict schemas and architecture match executable `QXAI-prj/main.py`. XAI: EfficientNet-B0 → global pooling → 1280 → 256 ReLU/dropout 0.5 → 3. QXAI uses the same 1280-feature backbone → linear 4 + tanh → four X AngleEmbedding qubits → four BasicEntanglerLayers → four Pauli-Z readouts → 4 → 64 ReLU → 3. The original README describes Ry but its executable circuit uses default X rotations; this run follows the code. PennyLane is a CPU quantum simulator, not quantum hardware, and does not establish quantum advantage.

ImageNet backbone **parameters are frozen**. During CNN training, `model.train()` updates BatchNorm running buffers and activates stochastic depth, matching the original code. The selected CNN is then put in eval mode and supplies fixed features for QXAI. This deliberately differs from v1's completely frozen eval-mode ImageNet backbone. No backbone convolution parameters were fine-tuned in this faithful run.

Separate train/validation dataset objects prevent the original shared-Subset transform bug. No image augmentation is enabled, matching what the original code actually executes after its transform assignments overwrite each other. Raw image input: RGB → PIL bilinear 224×224 → ToTensor [0,1], **no channel normalization**. Class order: AD, CONTROL, PD. Adam settings, batch 16, and original 10 CNN / 5 QML epoch budgets are recorded in `training_protocol.json`. The prespecified validation-only continuation permits at most 20 / 30 epochs if validation accuracy or macro F1 remains below 0.85, stopping after eight stale epochs. Actual completed/selected epochs are in the table. Highest validation macro F1 selects checkpoints; accuracy breaks ties. The original ineffective early-stopping helper is replaced by this saved protocol.

Training used the local `mps` device for CNN computation, with fixed Python/NumPy/PyTorch seed 42 and QML seed 43; quantum simulation stays on CPU. MPS results may not be bitwise identical across runs/platforms. Recorded training runtime: 6.6 minutes; evaluation runtime: 3.6 minutes. No paid services were used.

## Deployment contract

Load **both** files in this directory using `model_manifest.json` and verify their SHA256 hashes. Set model directory explicitly. Select preprocessing `resize224-rgb-totensor-v1` with mean/std null; do not apply v1 candidate ImageNet normalization to these weights. Selected BatchNorm buffers are part of `xai_model.pth` and are essential to the QXAI feature distribution. Original backend weights are preserved separately. Softmax scores are uncalibrated; these aggregate results cannot establish reliable diagnosis for arbitrary uploads or a clinical population.

Reproduction: `training/audit_v3.py`, `training/train_v3.py --stage train`, then a single `--stage evaluate`; `training/PROTOCOL_AUDIT.md` explains differences from the first candidate and original claims.
