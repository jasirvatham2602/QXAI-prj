# Original QXAI protocol and the first candidate gap

Reference: `jasirvatham2602/QXAI-prj`, commit `275564472359ff282c95a4de450e38fc729d7c57`; original `main.py` SHA256 `c255a90183f58e2b8d9a502662cb320619c41a326270546437203a22e9776ba2`.

The original README reports **89.46% XAI and 86.52% QXAI** on its own 20% random image holdout. The original archive contains 2,024 actual images: AD 1,003, CONTROL 767, PD 254. The source code stratifies each class with NumPy seed 42, takes floor(70%) for training, floor(10%) for validation, and the remainder for testing: **1,415 / 201 / 408**. The reported percentages correspond to 365/408 and 353/408 correct predictions. This is a different benchmark from the supplied Kaggle test used in the first candidate run.

The fresh full Kaggle archive used by candidate v1 has 7,839 images and its own train/test folders. Only **three** retained supplied-test images overlap the original GitHub archive by canonical 224×224 RGB hash; the other original images map into supplied training. V1 therefore did not reproduce the original data curation or evaluation protocol. Its 72.9%/68.7% supplied-test accuracy does not invalidate the original 89.46%/86.52% random-holdout claims, and the two pairs must not be placed in one comparative improvement table.

## Methodological differences

| Aspect | Original source | Candidate v1 |
|---|---|---|
| Data | Manually filtered 2,024-image archive | Full 7,839-image Kaggle archive |
| Split | Random image-level 70/10/20, seed 42 | Supplied Kaggle test; stratified validation from remaining training, seed 183 |
| Image input | RGB resize224 + ToTensor, no normalization | Same resize, then ImageNet channel normalization |
| Backbone weights | ImageNet parameters frozen | ImageNet parameters frozen |
| Backbone buffers/mode | `model.train()` updates BatchNorm running statistics and enables stochastic depth during CNN training | Parameters **and** BatchNorm buffers frozen in eval mode; cached fixed features |
| CNN optimization | Adam lr .001, weight decay .0001, batch16, ten epochs | AdamW lr .001, weight decay .001, weighted CE, batch64, up to80 epochs |
| QML optimization | Adam lr .001, unweighted CE, five epochs | AdamW lr .003, weighted CE, up to70 epochs |
| Selection | Final epoch; ineffective stopping helper | Best validation macro F1, patience12 |

Freezing `requires_grad` does **not** freeze BatchNorm buffers. The original train-mode feature distribution differs from its eval-mode distribution and differs from v1's fixed ImageNet features. Both original QML and CNN architectures were retained in v1; however, the full experimental procedure was materially different.

Original augmentation lines 71–80 assign an augmented transform through `train_ds.dataset`, then assign `base_transform` through validation/test. All three `Subset` objects reference the same `full_dataset`, so the latter assignments also reset training transforms. The nominal flips/rotation/color jitter are **not actually active** in that source. The early-stopping counter in lines 255–260 resets on every function call and cannot reach its tolerance. The README describes Ry encoding, while executable `AngleEmbedding` and `BasicEntanglerLayers` retain their default X rotations; candidates preserve executable behavior.

## Cleaned v3 replacement

The user subsequently supplied `jasirvatham2602/classifying-ad-and-pd-v3`. The slug says v3, but official Kaggle metadata currently identifies **dataset version 12**, last updated 2025-07-11. The zip has a three-class `3_cls_2/train/` dataset with 2,021 images (AD1,000 / CONTROL767 / PD254) and a separate `trainbinary/` copy of the AD/CONTROL subset. Binary copies must be excluded rather than counted as additional training images. There is no supplied test directory.

The replacement experiment locks grouped 70/10/20 image partitions with NumPy seed42 before training, retains original architecture, raw preprocessing, train-mode BN adaptation, Adam settings and ten/five epoch budgets. It intentionally saves best validation checkpoints and persists split histories. Test selection is never driven by the 80% target. This is a new cleaned-data random-holdout benchmark; it is not comparable to v1's supplied Kaggle test. Previous project models saw overlapping source images, but fresh v3 models start from ImageNet and exclude their locked holdout from training/selection. Without subject IDs, neither benchmark establishes patient-independent or clinical generalization. Fine-tuning, if tested, is a separate validation-only architecture-preserving ablation, not a means of repeatedly probing the test until a target is met.
