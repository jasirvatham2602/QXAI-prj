"""Local configuration and Supabase key handling; never log key values."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name('.env.local'))

def worker_settings():
    load_dotenv(Path(__file__).with_name('.env.worker.local'))
    url = os.environ['SUPABASE_URL'].rstrip('/')
    key = os.getenv('SUPABASE_SECRET_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    if not key:
        raise RuntimeError('Set SUPABASE_SECRET_KEY in the worker environment')
    headers = {'apikey': key}
    # New secret keys are not JWTs and belong only on the apikey header.
    if not key.startswith('sb_secret_'):
        headers['Authorization'] = 'Bearer ' + key
    return url, headers
