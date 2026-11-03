"""Inspect expired analyses by default. --apply explicitly enables permanent deletion."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import httpx
from settings import worker_settings
from storage import analysis_prefix


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Permanently delete expired records and images')
    args=parser.parse_args(argv)
    days=int(os.getenv('RETENTION_DAYS','30'))
    if days<1:raise ValueError('RETENTION_DAYS must be positive')
    url,headers=worker_settings()
    cutoff=(datetime.now(timezone.utc)-timedelta(days=days)).isoformat()
    inspected=0;offset=0
    with httpx.Client(base_url=url,headers=headers,timeout=30) as client:
        while True:
            response=client.get('/rest/v1/analyses',params={'select':'id,owner_id','status':'in.(complete,failed)',
                'created_at':'lt.'+cutoff,'order':'created_at.asc,id.asc','limit':'100','offset':str(offset)})
            response.raise_for_status();rows=response.json()
            if not rows:break
            for row in rows:
                inspected+=1
                if not args.apply:continue
                prefix=analysis_prefix(row['owner_id'],row['id'])
                listing=client.post('/storage/v1/object/list/analyses',json={'prefix':prefix+'/','limit':100})
                listing.raise_for_status()
                paths=[prefix+'/'+obj['name'] for obj in listing.json() if obj.get('id') and '/' not in obj['name'] and '..' not in obj['name']]
                if paths:
                    removed=client.request('DELETE','/storage/v1/object/analyses',json={'prefixes':paths});removed.raise_for_status()
                deleted=client.delete('/rest/v1/analyses',params={'id':'eq.'+row['id']});deleted.raise_for_status()
            if not args.apply:offset+=len(rows)
    print(json.dumps({'mode':'apply' if args.apply else 'dry_run','retention_days':days,'expired_records':inspected}))

if __name__=='__main__':main()
