"""Capture validation-only CPU predictions; check the actual original app runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'models/candidates/cleaned-v3-seed42'


def capture():
    from models import ClassicalCNN, QuantumHybrid
    from train import ArchiveDataset
    split = json.loads((OUT/'split_manifest.json').read_text())
    selection = json.loads((OUT/'selection.json').read_text())
    for name, expected in selection['checkpoint_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == expected
    rows = [row for label in ['AD','CONTROL','PD'] for row in [r for r in split['splits']['validation'] if r['label']==label][:2]]
    dataset = ArchiveDataset(ROOT/'training/data/cleaned-v3.zip',rows,False)
    cnn, quantum = ClassicalCNN().eval(), QuantumHybrid().eval()
    cnn.load_state_dict(torch.load(OUT/'xai_model.pth',map_location='cpu'),strict=True)
    quantum.load_state_dict(torch.load(OUT/'qxai_model.pth',map_location='cpu'),strict=True)
    images = torch.stack([dataset[index][0] for index in range(len(rows))])
    with torch.no_grad():
        features = cnn(images,return_features=True)
        classical_logits = cnn.classifier(features)
        quantum_logits = quantum(features)
        classical_probability = classical_logits.softmax(1)
        quantum_probability = quantum_logits.softmax(1)
    folder = ROOT/'training/data/v3-golden'
    folder.mkdir(parents=True,exist_ok=True)
    examples = []
    with zipfile.ZipFile(ROOT/'training/data/cleaned-v3.zip') as archive:
        for index,row in enumerate(rows):
            data = archive.read(row['path'])
            local = folder/(f"{index+1:02}_{row['label']}"+Path(row['path']).suffix.lower())
            local.write_bytes(data)
            examples.append({'source_path':row['path'],'source_label':row['label'],'local_path':str(local.relative_to(ROOT)),
                             'image_sha256':hashlib.sha256(data).hexdigest(),'classical_logits':classical_logits[index].tolist(),
                             'quantum_logits':quantum_logits[index].tolist(),'classical_probabilities':classical_probability[index].tolist(),
                             'quantum_probabilities':quantum_probability[index].tolist()})
    report = {'purpose':'Validation-only deployment compatibility cases. Captured after checkpoint selection; never used to choose weights.',
              'torch_version':torch.__version__,'device':'cpu','uses_test_images':False,'classes':['AD','CONTROL','PD'],
              'checkpoint_sha256':selection['checkpoint_sha256'],'cases':examples,'probability_atol':.0001}
    (OUT/'validation_golden.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'captured_validation_cases':len(examples),'torch':torch.__version__}))


def verify():
    from PIL import Image
    os.environ['QXAI_MODEL_DIR'] = str(OUT)
    sys.path.insert(0,str(ROOT/'backend'))
    from inference import xai_model,qxai_model,base_transform,classical_saliency,quantum_image_saliency
    golden = json.loads((OUT/'validation_golden.json').read_text())
    images = []
    for row in golden['cases']:
        path = ROOT/row['local_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256']
        with Image.open(path) as image:
            images.append(base_transform(image.convert('RGB')))
    batch = torch.stack(images)
    with torch.no_grad():
        features = xai_model(batch,return_features=True)
        c = xai_model.classifier(features).softmax(1)
        q = qxai_model(features).softmax(1)
    expected_c = torch.tensor([row['classical_probabilities'] for row in golden['cases']])
    expected_q = torch.tensor([row['quantum_probabilities'] for row in golden['cases']])
    atol = golden['probability_atol']
    probabilities_match = torch.allclose(c,expected_c,atol=atol,rtol=atol) and torch.allclose(q,expected_q,atol=atol,rtol=atol)
    predictions_match = torch.equal(c.argmax(1),expected_c.argmax(1)) and torch.equal(q.argmax(1),expected_q.argmax(1))
    c_sal = classical_saliency(xai_model,batch[:1],int(c[0].argmax()))
    q_sal = quantum_image_saliency(xai_model,qxai_model,batch[:1],int(q[0].argmax()))
    result = {'torch_version':torch.__version__,'uses_test_images':False,'actual_backend_loaded':True,'validation_cases':len(images),
              'probabilities_match':bool(probabilities_match),'predictions_match':bool(predictions_match),
              'max_classical_probability_difference':float((c-expected_c).abs().max()),
              'max_quantum_probability_difference':float((q-expected_q).abs().max()),
              'classical_saliency_finite_nonzero':bool(torch.isfinite(c_sal).all() and c_sal.abs().max()>0),
              'quantum_saliency_finite_nonzero':bool(torch.isfinite(q_sal).all() and q_sal.abs().max()>0)}
    if not all(result[key] for key in ['probabilities_match','predictions_match','classical_saliency_finite_nonzero','quantum_saliency_finite_nonzero']):
        raise RuntimeError('Original app runtime compatibility check failed: '+json.dumps(result))
    (OUT/'golden_backend_compatibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['capture','verify'],required=True)
    args = parser.parse_args()
    capture() if args.stage=='capture' else verify()
