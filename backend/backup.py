"""Encrypted local backup of completed/failed research analyses; no credential export."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

from cryptography.fernet import Fernet
import httpx
from settings import worker_settings
from storage import analysis_prefix, object_path

MAX_BACKUP_BYTES=256*1024*1024

def private_write(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as output: output.write(data)

def key_for(path, create=False):
    path=Path(path)
    if not path.exists():
        if not create: raise ValueError('Backup key does not exist')
        private_write(path,Fernet.generate_key())
    if path.stat().st_mode & 0o077: raise ValueError('Backup key must be owner-readable only (chmod 600)')
    return path.read_bytes()

def encode_archive(rows, blobs, key):
    files={'analyses.json':json.dumps(rows,sort_keys=True).encode(),**blobs}
    inventory={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'records':len(rows),'files':inventory,
              'scope':'terminal analyses and referenced images; excludes auth accounts, secrets, queued/processing jobs, and stale attempt outputs'}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
        archive.writestr('manifest.json',json.dumps(manifest))
    return Fernet(key).encrypt(buffer.getvalue())

def decode_archive(data,key):
    raw=Fernet(key).decrypt(data)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate backup members')
        if any(Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name for name in names):raise ValueError('Unsafe backup paths')
        manifest=json.loads(archive.read('manifest.json'))
        if manifest['schema_version']!=1 or set(names)!={'manifest.json',*manifest['files']}:raise ValueError('Invalid inventory')
        files={name:archive.read(name) for name in manifest['files']}
        if sum(map(len,files.values()))>MAX_BACKUP_BYTES:raise ValueError('Backup exceeds local recovery size limit')
        if any(hashlib.sha256(data).hexdigest()!=manifest['files'][name] for name,data in files.items()):raise ValueError('Backup integrity mismatch')
        rows=json.loads(files['analyses.json'])
        if len(rows)!=manifest['records']:raise ValueError('Record count mismatch')
        expected={'analyses.json'}
        for row in rows:
            if row['status'] not in ('complete','failed'):raise ValueError('Non-terminal backup row')
            expected.add('images/'+object_path(row['owner_id'],row['id'],'input'))
            if row['status']=='complete':expected.add('images/'+object_path(row['owner_id'],row['id'],'saliency',row.get('output_path')))
        if set(files)!=expected:raise ValueError('Backup is missing referenced images')
        return manifest,files

def create_backup(client,key):
    rows=[];offset=0
    while True:
        response=client.get('/rest/v1/analyses',params={'select':'*','status':'in.(complete,failed)','order':'id.asc','limit':100,'offset':offset})
        response.raise_for_status();batch=response.json();rows.extend(batch)
        if len(batch)<100:break
        offset+=len(batch)
    blobs={};size=0
    for row in rows:
        analysis_prefix(row['owner_id'],row['id'])
        for kind in (['input','saliency'] if row['status']=='complete' else ['input']):
            path=object_path(row['owner_id'],row['id'],kind,row.get('output_path'))
            response=client.get('/storage/v1/object/authenticated/analyses/'+path);response.raise_for_status()
            size+=len(response.content)
            if size>MAX_BACKUP_BYTES:raise ValueError('Backup exceeds 256 MB; use a streaming backup before proceeding')
            blobs['images/'+path]=response.content
    return encode_archive(rows,blobs,key)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['create','verify','recover-local'])
    parser.add_argument('archive',type=Path)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args(argv)
    key=key_for(args.key_file,create=args.operation=='create')
    if args.operation=='create':
        url,headers=worker_settings()
        with httpx.Client(base_url=url,headers=headers,timeout=30) as client:data=create_backup(client,key)
        manifest,_=decode_archive(data,key)
        private_write(args.archive,data)
    else:
        manifest,files=decode_archive(args.archive.read_bytes(),key)
        if args.operation=='recover-local':
            if not args.destination or args.destination.exists():raise ValueError('Choose a new private recovery directory')
            args.destination.mkdir(parents=True,mode=0o700)
            for name,data in files.items():private_write(args.destination/name,data)
            private_write(args.destination/'manifest.json',json.dumps(manifest).encode())
    print(json.dumps({'operation':args.operation,'verified_records':manifest['records'],'verified_files':len(manifest['files'])}))

if __name__=='__main__':main()
