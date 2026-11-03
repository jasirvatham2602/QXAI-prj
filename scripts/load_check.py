"""Small, non-mutating API latency check; never submits scans or reads histories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
import urllib.request
parser=argparse.ArgumentParser()
parser.add_argument('base_url')
args=parser.parse_args()
def check(_):
    started=time.monotonic()
    with urllib.request.urlopen(args.base_url.rstrip('/')+'/health/live',timeout=20) as response:
        if response.status!=200:raise RuntimeError('API liveness check failed')
    return time.monotonic()-started
with ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(check,range(12)))
print(json.dumps({'requests':len(values),'concurrency':2,'median_ms':round(statistics.median(values)*1000,2),'max_ms':round(max(values)*1000,2)}))
