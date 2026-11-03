"""Monitor readiness without credentials, allowing a bounded free-host cold start."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('base_url')
parser.add_argument('--wait-seconds',type=int,default=0)
args=parser.parse_args()
if not 0<=args.wait_seconds<=180:parser.error('wait-seconds must be between 0 and 180')
started=time.monotonic();deadline=started+args.wait_seconds;attempts=0
while True:
    attempts+=1
    try:
        with urllib.request.urlopen(args.base_url.rstrip('/')+'/health/ready',timeout=15) as response:
            ready=response.status==200
    except (urllib.error.URLError,TimeoutError):ready=False
    if ready or time.monotonic()>=deadline:break
    time.sleep(min(5,max(0,deadline-time.monotonic())))
print(json.dumps({'ready':ready,'attempts':attempts,'elapsed_seconds':round(time.monotonic()-started,2)}))
sys.exit(0 if ready else 1)
