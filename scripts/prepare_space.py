"""Stage only the reviewed runtime files for the existing Hugging Face Space."""
import argparse
import json
from pathlib import Path
import shutil
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'backend'))
from model_bundle import load_bundle
parser=argparse.ArgumentParser()
parser.add_argument('destination',type=Path)
parser.add_argument('--models',type=Path,required=True)
args=parser.parse_args()
bundle=load_bundle(args.models)
if args.destination.exists():raise SystemExit('Choose a new staging directory')
args.destination.mkdir(parents=True)
files=['main.py','service.py','settings.py','storage.py','admin_access.py','model_bundle.py','inference.py','worker.py','run_inference.py','start_space.py','requirements-space.txt']
for name in files:shutil.copyfile(root/'backend'/name,args.destination/name)
shutil.copyfile(root/'backend/Dockerfile.space',args.destination/'Dockerfile')
for name in ['model_manifest.json','xai_model.pth','qxai_model.pth']:shutil.copyfile(args.models/name,args.destination/name)
# The runtime directory is deliberately separate from training caches and all .env files.
(args.destination/'README.md').write_text('''---
title: QXAI Research API
emoji: 🔬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Authenticated classical and quantum MRI research workspace
---

Research use only. Private histories and scans require a verified QXAI login.
API and one worker share the existing CPU Basic Space. It may sleep while inactive.
No training dataset, uploaded scans, private backup, or local credential files are included in this repository.
''')
print(json.dumps({'staged_runtime_files':len(list(args.destination.iterdir())),'model_version':bundle['model_version'],'destination':str(args.destination)}))
