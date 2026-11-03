"""Validate a model bundle and its preprocessing without importing the ML runtime."""
import hashlib
import json
import os
from pathlib import Path

CLASSES = ['AD', 'CONTROL', 'PD']

def load_bundle(directory=None):
    directory = Path(directory or os.getenv('QXAI_MODEL_DIR') or Path(__file__).parent).resolve()
    manifest_path = directory / 'model_manifest.json'
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        if data['class_names'] != CLASSES or data['image_size'] != 224:
            raise ValueError('Unsupported class order or image size')
        contracts = {'resize224-rgb-imagenet-normalized-v2':([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                     'resize224-rgb-totensor-v1':(None,None)}
        expected = contracts.get(data['preprocessing_version'])
        if expected is None:
            raise ValueError('Unsupported model preprocessing')
        if (data['mean'],data['std']) != expected:
            raise ValueError('Unexpected normalization')
        paths = {}
        for kind in ('classical','quantum'):
            item = data['checkpoints'][kind]
            if Path(item['file']).name != item['file'] or not item['file'].endswith('.pth'):
                raise ValueError('Invalid checkpoint filename')
            path = directory / item['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Model checksum mismatch')
            paths[kind] = path
        data['paths'] = paths
        return data
    if os.getenv('QXAI_MODEL_DIR') or directory != Path(__file__).parent.resolve():
        raise ValueError('Explicit model directory requires a manifest')
    paths = {kind:directory/name for kind,name in [('classical','xai_model.pth'),('quantum','qxai_model.pth')]}
    version = hashlib.sha256(b''.join(path.read_bytes() for path in paths.values())).hexdigest()[:16]
    return {'model_version':version,'preprocessing_version':'resize224-rgb-totensor-v1',
            'class_names':CLASSES,'image_size':224,'mean':None,'std':None,'paths':paths}
