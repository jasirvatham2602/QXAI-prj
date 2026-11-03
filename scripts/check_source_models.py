"""Diagnostic only: evaluate saved weights on labeled source images, not a held-out benchmark."""
import argparse
import collections
import io
import json
from pathlib import Path
import random
import sys
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--per-class',type=int,default=12)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import inference as model
from PIL import Image

report={'purpose':'Source-data diagnostic sample; not a held-out accuracy estimate','samples_per_class':args.per_class,
        'class_order':model.class_names,'models':{kind:{} for kind in ('classical','quantum')}}
with zipfile.ZipFile(args.archive) as archive:
    for label in model.class_names:
        names=sorted(name for name in archive.namelist()
                     if name.split('/')[-2]==label and not name.startswith('__MACOSX/')
                     and not Path(name).name.startswith('._')
                     and name.lower().endswith(('.png','.jpg','.jpeg')))
        chosen=random.Random(183).sample(names,min(args.per_class,len(names)))
        tensors=[]
        for name in chosen:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                tensors.append(model.base_transform(image.convert('RGB')))
        batch=model.torch.stack(tensors).to(model.device)
        with model.torch.no_grad():
            classical=model.xai_model(batch).argmax(1).tolist()
            quantum=model.qxai_model(model.xai_model(batch,return_features=True)).argmax(1).tolist()
        for kind,predictions in [('classical',classical),('quantum',quantum)]:
            counts=collections.Counter(model.class_names[index] for index in predictions)
            report['models'][kind][label]={'correct':counts.get(label,0),'total':len(predictions),'predictions':dict(counts)}
        print('Checked source class '+label,flush=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
