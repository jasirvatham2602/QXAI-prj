"""Update only the named, existing CPU Basic QXAI Space using a reviewed staging bundle."""
import argparse
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'backend'))
import shlex
def dotenv_values(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#') or '=' not in line:continue
        name,value=line.split('=',1)
        parts=shlex.split(value,comments=True)
        result[name.strip()]=parts[0] if parts else ''
    return result
from huggingface_hub import HfApi
from model_bundle import load_bundle

REPO='jasirvatham2602/qxai-backend-hugging-face'
parser=argparse.ArgumentParser()
parser.add_argument('staging',type=Path)
parser.add_argument('--apply',action='store_true',help='Update the existing Space; never request paid hardware')
args=parser.parse_args()
bundle=load_bundle(args.staging)
expected={'main.py','service.py','settings.py','storage.py','admin_access.py','model_bundle.py','inference.py','worker.py','run_inference.py','start_space.py','requirements-space.txt','Dockerfile','README.md','model_manifest.json','xai_model.pth','qxai_model.pth'}
if {p.name for p in args.staging.iterdir()}!=expected or any(not p.is_file() or p.is_symlink() for p in args.staging.iterdir()):raise ValueError('Unexpected runtime files')
api=HfApi();info=api.space_info(REPO)
runtime=api.get_space_runtime(REPO)
if info.sdk!='docker' or str(runtime.hardware) not in ('cpu-basic','SpaceHardware.CPU_BASIC'):raise ValueError('Expected the existing free CPU Basic Docker Space')
if not args.apply:
    print('Validated existing free Space and runtime bundle. No remote changes made.');raise SystemExit()
settings={**dotenv_values(root/'backend/.env.local'),**dotenv_values(root/'backend/.env.worker.local')}
public={'SUPABASE_URL':settings['SUPABASE_URL'],'SUPABASE_PUBLISHABLE_KEY':settings.get('SUPABASE_PUBLISHABLE_KEY') or settings['SUPABASE_ANON_KEY'],
        'FRONTEND_ORIGINS':'https://qxai-fullstack-prj.netlify.app,http://127.0.0.1:5174,http://localhost:5174',
        'QXAI_ENV':'production','QXAI_MODEL_DIR':'/app'}
secret=settings.get('SUPABASE_SECRET_KEY') or settings['SUPABASE_SERVICE_ROLE_KEY']
# Add only to the existing backend's server-side secret store, never to source files.
api.add_space_secret(REPO,key='SUPABASE_SECRET_KEY',value=secret)
for name,value in public.items():api.add_space_variable(REPO,key=name,value=value)
commit=api.upload_folder(repo_id=REPO,repo_type='space',folder_path=args.staging,commit_message='Deploy authenticated research workspace and evaluated '+bundle['model_version'])
print('Updated existing CPU Basic Space. Commit:',commit.oid)
