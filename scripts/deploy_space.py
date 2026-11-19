"""Update only the named, existing CPU Basic QXAI Space using a reviewed staging bundle."""
import argparse
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'backend'))
import shlex
def dotenv_values(path):
    result={}
    path=Path(path)
    if not path.is_file():return result
    for line in path.read_text().splitlines():
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
expected={'main.py','service.py','settings.py','storage.py','admin_access.py','model_bundle.py','inference.py','worker.py','run_inference.py','start_space.py','guide_rag.py','requirements-space.txt','Dockerfile','README.md','model_manifest.json','xai_model.pth','qxai_model.pth'}
if {p.name for p in args.staging.iterdir()}!=expected or any(not p.is_file() or p.is_symlink() for p in args.staging.iterdir()):raise ValueError('Unexpected runtime files')
api=HfApi();info=api.space_info(REPO)
runtime=api.get_space_runtime(REPO)
if info.sdk!='docker' or str(runtime.hardware) not in ('cpu-basic','SpaceHardware.CPU_BASIC'):raise ValueError('Expected the existing free CPU Basic Docker Space')
if not args.apply:
    print('Validated existing free Space and runtime bundle. No remote changes made.');raise SystemExit()
settings={**dotenv_values(root/'backend/.env.local'),**dotenv_values(root/'backend/.env.worker.local')}
public={'FRONTEND_ORIGINS':'https://blue-qxai-prj.netlify.app,http://127.0.0.1:5174,http://localhost:5174',
        'QXAI_ENV':'production','QXAI_MODEL_DIR':'/app'}
supabase_url=settings.get('SUPABASE_URL','').strip()
supabase_key=(settings.get('SUPABASE_PUBLISHABLE_KEY') or settings.get('SUPABASE_ANON_KEY') or '').strip()
if bool(supabase_url)!=bool(supabase_key):raise ValueError('Configure both the Supabase URL and publishable key, or omit both to preserve the existing Space settings')
if supabase_url:
    public['SUPABASE_URL']=supabase_url
    public['SUPABASE_PUBLISHABLE_KEY']=supabase_key
secret=(settings.get('SUPABASE_SECRET_KEY') or settings.get('SUPABASE_SERVICE_ROLE_KEY') or '').strip()
llm_key=settings.get('QXAI_LLM_API_KEY','').strip()
llm_model=settings.get('QXAI_LLM_MODEL','').strip()
if bool(llm_key)!=bool(llm_model):raise ValueError('Configure both QXAI_LLM_API_KEY and QXAI_LLM_MODEL before deployment')
if llm_key:
    public['QXAI_LLM_MODEL']=llm_model
    if settings.get('QXAI_LLM_BASE_URL','').strip():public['QXAI_LLM_BASE_URL']=settings['QXAI_LLM_BASE_URL'].strip()
# Add only to the existing backend's server-side secret store, never to source files.
if secret:api.add_space_secret(REPO,key='SUPABASE_SECRET_KEY',value=secret)
if llm_key:api.add_space_secret(REPO,key='QXAI_LLM_API_KEY',value=llm_key)
for name,value in public.items():api.add_space_variable(REPO,key=name,value=value)
commit=api.upload_folder(repo_id=REPO,repo_type='space',folder_path=args.staging,commit_message='Deploy authenticated research workspace and evaluated '+bundle['model_version'])
print('Updated existing CPU Basic Space. Commit:',commit.oid)
