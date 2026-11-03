"""Preserve original weights with an explicit, hash-checked legacy input contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args()
if args.destination.exists():raise SystemExit('Choose a new original-bundle directory')
args.destination.mkdir(parents=True)
paths={kind:root/'backend'/name for kind,name in [('classical','xai_model.pth'),('quantum','qxai_model.pth')]}
manifest={'model_version':hashlib.sha256(b''.join(p.read_bytes() for p in paths.values())).hexdigest()[:16],
 'preprocessing_version':'resize224-rgb-totensor-v1','class_names':['AD','CONTROL','PD'],'image_size':224,'mean':None,'std':None,
 'checkpoints':{kind:{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for kind,path in paths.items()},'research_only':True}
for path in paths.values():shutil.copyfile(path,args.destination/path.name)
(args.destination/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared the preserved original model bundle with explicit preprocessing and hashes.')
