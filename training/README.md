# Reproducible QXAI training

This pipeline actually trains both compatible classifier heads on the public Kaggle archive. It preserves the original checkpoints in `backend/` and writes separate candidate checkpoints, a manifest, and held-out evaluation reports under `models/candidates/kaggle-v1-seed183/`. Neither model is clinically validated.

## Data and rights

- Dataset: <https://www.kaggle.com/datasets/farjanakabirsamanta/alzheimer-diseases-3-class>
- Official download: <https://www.kaggle.com/api/v1/datasets/download/farjanakabirsamanta/alzheimer-diseases-3-class>
- Recorded archive SHA256: `57a55eb74a4128e369b46c1fe1214bc35c379bb72ad619c53500c62ee43597e3`
- Kaggle metadata lists **Other (specified in description)**. Its description points to `gokulramasamy/alzheimer-parkinson-disease`, whose metadata lists **Unknown** and provides no additional terms. The dataset is publicly downloadable; unrestricted redistribution rights are **not established**. Review source terms before distributing data or using it beyond this local research experiment.

Training images, downloaded archives, metadata caches, and extracted features stay in ignored `training/data/`. Do not include them in a site or source release.

## Run

Use a separate Python 3.9 environment; do not alter the application's runtime environment. Install `training/requirements.txt`. Place the Kaggle zip at `training/data/kaggle.zip` without extracting it.

The frozen backbone uses torchvision's official `EfficientNet_B0_Weights.IMAGENET1K_V1` checkpoint, `efficientnet_b0_rwightman-7f5810bc.pth`. Torchvision can download this from <https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth>. Keep it in the normal Torch hub checkpoint cache, or pass `--backbone-weights /absolute/path/to/checkpoint.pth`.

From the repository root:

```sh
MPLBACKEND=Agg python training/train.py --seed 183
MPLBACKEND=Agg python training/summarize.py
MPLBACKEND=Agg python training/verify.py
```

Use `--archive` and `--output` to select different local paths. `--classical-epochs` defaults to 80 and `--quantum-epochs` to 70; early stopping ends a run after 12 epochs without a validation macro-F1 improvement. Running a different experiment requires a separate output directory. Feature extraction is cached against the exact manifest, backbone hash, and preprocessing. The initial CPU run used four threads, no GPU, and no paid services.

## Split and evaluation

The supplied Kaggle test set remains the held-out test. Before splitting, the pipeline hashes decoded RGB images resized to the actual 224×224 model input. It groups exact matches, removes duplicate copies, and excludes any groups with contradictory labels. For train/test duplicates, the supplied test image wins and its training copy is removed. It then takes a stratified 15% validation sample from the remaining supplied training groups, using seed 183. Saved manifests identify every retained archive path, class, and group hash. Metadata files contain source image filenames, not user-uploaded information.

The recorded split has 5,486 train, 968 validation, and 1,362 test images. Twenty-three duplicate copies were removed, including 11 supplied training images overlapping the test set. No contradictory-label groups were found. Exact hashes do not remove near duplicates or establish subject independence. No patient IDs were supplied, so these are **image-level research results**, with possible patient overlap and acquisition confounds. In particular, PD image acquisition may differ from AD/CONTROL.

Checkpoint selection uses **validation macro F1 only**. Inverse-frequency class weights address training imbalance. The final held-out report provides confusion matrices, accuracy, balanced accuracy, macro F1, and precision/recall/F1/support for each class. AD recall is reported explicitly rather than hidden in aggregate accuracy.

The old checkpoints are evaluated on the same retained test images with their original **unnormalized** Resize/ToTensor preprocessing and their own saved backbone buffers. Their exact original training membership was not saved, and the old training source overlaps this archive. Treat that comparison as a diagnostic baseline, not an independent generalization estimate. No private uploads are used by this pipeline.

## Architecture and preprocessing

The state-dict schemas match the original deployment architecture:

- XAI: EfficientNet-B0 → global average pooling → 1280→256→3 classifier, ReLU and dropout 0.5.
- QXAI: the same 1280-dimensional CNN features → linear 4 + tanh → four simulated qubits → four BasicEntanglerLayers → four Pauli-Z expectations → 4→64→3 classifier.
- PennyLane `AngleEmbedding` retains the original default **X rotation**; `BasicEntanglerLayers` also retains defaults. The implementation broadcasts the quantum batch but was checked against per-image circuit evaluation, including finite gradients. This is a classical CPU quantum simulation; these results do not demonstrate quantum advantage.

The CNN's ImageNet parameters **and BatchNorm buffers** remain frozen in evaluation mode during feature extraction. Both heads train on those same cached features. No test-derived normalization, feature tuning, or checkpoint selection occurs.

Candidate preprocessing version: `resize224-rgb-imagenet-normalized-v2`.

1. Decode and convert to RGB.
2. Resize to 224×224 using torchvision/PIL bilinear interpolation (no crop, no preserved aspect ratio).
3. Convert to float tensor in [0,1].
4. Normalize channels using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

The deployment must select this preprocessing together with **both** candidate checkpoints. Do not use the old CNN features with the new quantum head, or apply candidate normalization to old weights. The candidate `model_manifest.json` records filenames and SHA256 hashes for explicit loading; parent application code controls promotion. Candidate results remain research-only and do not establish reliable diagnosis for arbitrary PNG files or real clinical scans.
