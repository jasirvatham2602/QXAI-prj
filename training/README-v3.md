# Cleaned v3 experiment

This experiment uses the user's replacement [Classifying AD and PD v3 dataset](https://www.kaggle.com/datasets/jasirvatham2602/classifying-ad-and-pd-v3). Its current Kaggle version is **12**, not 3. The fresh archive SHA256 is `ffe22a860b3592d1a59bd0ffadcb2c38f6a026345b16a28ea0204d0729e036fb`.

Only the three-class `3_cls_2/train/` directory is used. The separate `trainbinary/` copies are excluded. The 2,021 intended images have no exact input-hash duplicates or contradictory-label groups. A stratified 70/10/20 split, NumPy seed 42 and original path/class ordering, is saved before training: 1,413 training, 201 validation, 407 test images. No patient IDs are available; this is an image-level research benchmark. The prior project has used overlapping source images, so it is not a globally untouched project dataset.

The original executable architecture, raw Resize/ToTensor input, train-mode BatchNorm adaptation, unweighted cross entropy, Adam settings and 10 CNN / 5 QML epoch budgets are retained. The saved protocol intentionally adds validation checkpoint selection and fixes the ineffective original stopping helper. A prespecified validation-only continuation can extend to at most 20 / 30 epochs, with patience 8, if validation accuracy or macro F1 remains below 0.85. No held-out predictions choose training settings. See [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md) for exact differences from the original repository and first candidate run.

Use a separate Python 3.9 environment with `requirements-v3.txt`. The recorded run uses torch 2.5.1 / torchvision 0.20.1 on the local Apple GPU; PennyLane quantum simulation stays on CPU. The older torch 1.13 GPU runtime crashed before completing an epoch, so it was preserved for CPU app compatibility checking and not changed. Fixed seeds improve reproducibility, but MPS kernels are not guaranteed bitwise deterministic across systems.

Place the official zip in ignored `training/data/cleaned-v3.zip`. Do not include it, extracted golden images, or feature caches in a release package. Kaggle license metadata says **Unknown**; redistribution rights are not established.

```sh
python training/audit_v3.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python training/train_v3.py --stage train --device mps
```

Training writes `selection.json` with hashes and validation results. The test set is not predicted during this stage. After selection is complete, run **one** CPU evaluation, matching the deployment target:

```sh
python training/train_v3.py --stage evaluate --device cpu
MPLBACKEND=Agg python training/summarize_v3.py
python training/validation_golden_v3.py --stage capture
```

The evaluation stage refuses to overwrite existing test results or evaluate changed checkpoints. The 80% target is a requested outcome, not permission to repeatedly inspect the test set until a run passes. Earlier supplied-Kaggle-test results remain preserved in `models/candidates/kaggle-v1-seed183/`; they are a different benchmark.

Using the original CPU app environment, verify model state, frozen parameters, adapted BatchNorm counters, the deployment loader, quantum-loop equivalence, actual backend probabilities and image gradients:

```sh
python training/verify_v3.py
MPLBACKEND=Agg python training/validation_golden_v3.py --stage verify
```

The golden checks use six fixed validation cases captured after selection. They do not choose weights. Their source images remain in ignored `training/data/v3-golden/`; aggregate expected predictions and hashes are in the candidate directory. Existing backend checkpoints are never overwritten. Load both new files together using `model_manifest.json`, raw preprocessing `resize224-rgb-totensor-v1`, null mean/std and class order `AD, CONTROL, PD`.

Outputs are under `models/candidates/cleaned-v3-seed42/`: split/protocol locks, selected weights, epoch histories, evaluation, manifest, model card, confusion matrices, golden predictions and compatibility verification. These research results do not establish reliable clinical diagnosis, calibrated probabilities or quantum advantage.
