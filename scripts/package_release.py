"""Create a portable source/build package using an allowlist; never include local secrets."""
import argparse
from pathlib import Path
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('output',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
if args.output.suffix!='.zip':raise SystemExit('Output must end in .zip')
args.output.parent.mkdir(parents=True,exist_ok=True)
folders=['src','public','backend','supabase','scripts','tests','.github','dist','training','models']
files=['package.json','package-lock.json','vite.config.js','eslint.config.js','index.html','.env.example','.gitignore','DEPLOYMENT.md','RELEASE_CHECKLIST.md','README.md','netlify.toml']
paths=[root/name for name in files]
for folder in folders:
    paths.extend((root/folder).rglob('*'))
with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as output:
    for path in sorted(set(paths)):
        relative=path.relative_to(root)
        if not path.is_file() or path.is_symlink():continue
        if any(part in ('venv','node_modules','__pycache__','.pytest_cache','data','cache','.cache','.qxai-private') for part in relative.parts):continue
        if path.name.startswith('.env') and path.name!='.env.example':continue
        if path.suffix in ('.pyc','.db','.log','.npz','.npy'):continue
        if 'training' in relative.parts and path.suffix not in ('.py','.md','.json','.txt'):continue
        output.write(path,relative.as_posix())
print('Release package created; local configuration and runtime data excluded.')
