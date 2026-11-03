"""Lock the cleaned-v3 image-level split before model training or test evaluation."""
import collections
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['AD', 'CONTROL', 'PD']
OUT = ROOT / 'models/candidates/cleaned-v3-seed42'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'training/data/cleaned-v3.zip'
    rows, excluded_binary = [], 0
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts = Path(name).parts
            if '__MACOSX' in parts or Path(name).name.startswith('._') or Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            if 'trainbinary' in parts:
                excluded_binary += 1
                continue
            if len(parts) < 4 or parts[-3] != 'train' or parts[-2] not in CLASSES:
                raise ValueError('Unexpected image path: ' + name)
            with Image.open(io.BytesIO(source.read(name))) as image:
                rgb = image.convert('RGB')
                digest = hashlib.sha256(rgb.resize((224, 224), Image.Resampling.BILINEAR).tobytes()).hexdigest()
                rows.append({'path': name, 'label': parts[-2], 'target': CLASSES.index(parts[-2]), 'group': digest, 'size': list(rgb.size)})
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    conflicts = {key for key, members in groups.items() if len({row['label'] for row in members}) > 1}
    retained = [members[0] for key, members in groups.items() if key not in conflicts]
    splits = {'train': [], 'validation': [], 'test': []}
    rng = np.random.RandomState(42)
    for label in CLASSES:
        members = sorted([row for row in retained if row['label'] == label], key=lambda row: row['path'])
        indices = np.arange(len(members))
        rng.shuffle(indices)
        ntrain, nval = int(.7 * len(members)), int(.1 * len(members))
        for name, selected in [('train', indices[:ntrain]), ('validation', indices[ntrain:ntrain+nval]), ('test', indices[ntrain+nval:])]:
            splits[name] += [members[index] for index in selected]
    manifest = {
        'dataset_url': 'https://www.kaggle.com/datasets/jasirvatham2602/classifying-ad-and-pd-v3',
        'dataset_version': 12, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'seed': 42, 'test_name': 'cleaned-v3-version12-grouped-random-image-holdout-seed42',
        'split_policy': 'No supplied test split. Stratified 70/10/20 split of unique input-hash groups, NumPy RandomState(42), original class/path order. Duplicate copies are not reweighted.',
        'source_three_class_images': len(rows), 'source_counts': dict(collections.Counter(row['label'] for row in rows)),
        'excluded_binary_copies': excluded_binary, 'duplicate_copies_removed': sum(len(members)-1 for key, members in groups.items() if key not in conflicts),
        'contradictory_groups_excluded': len(conflicts), 'retained_images': len(retained),
        'patient_independence': 'Not established: no subject IDs. Previous project models used overlapping source images, but this fresh run starts from ImageNet and excludes its locked test images from training/selection.',
        'counts': {name: dict(collections.Counter(row['label'] for row in selected)) for name, selected in splits.items()},
        'splits': splits,
    }
    for left in splits:
        for right in splits:
            if left != right:
                assert not ({row['group'] for row in splits[left]} & {row['group'] for row in splits[right]})
    old = json.loads((ROOT / 'models/candidates/kaggle-v1-seed183/split_manifest.json').read_text())
    old_groups = {row['group']: split for split, samples in old['splits'].items() for row in samples}
    with zipfile.ZipFile(ROOT.parent / 'QXAI-training-review/data.zip') as source:
        original = set()
        for name in source.namelist():
            if '__MACOSX' in Path(name).parts or Path(name).name.startswith('._') or Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            with Image.open(io.BytesIO(source.read(name))) as image:
                original.add(hashlib.sha256(image.convert('RGB').resize((224,224), Image.Resampling.BILINEAR).tobytes()).hexdigest())
    manifest['source_overlap'] = {
        'original_github_archive_unique_images': len(original),
        'v3_images_present_in_github_archive': sum(row['group'] in original for row in retained),
        'v3_images_by_previous_kaggle_v1_partition': dict(collections.Counter(old_groups.get(row['group'], 'absent') for row in retained)),
        'v3_test_images_by_previous_kaggle_v1_partition': dict(collections.Counter(old_groups.get(row['group'], 'absent') for row in splits['test'])),
    }
    path = OUT / 'split_manifest.json'
    serialized = json.dumps(manifest, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise RuntimeError('Refusing to change locked v3 split')
    path.write_text(serialized)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'splits'}, indent=2))


if __name__ == '__main__':
    main()
