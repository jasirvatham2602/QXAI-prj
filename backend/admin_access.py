"""Admin access: verified Supabase admin JWTs, plus explicitly enabled loopback-only development login."""
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from settings import worker_settings

COOKIE='qxai_local_admin'
_sessions={}
_attempts=defaultdict(deque)


def password_hash(password):
    salt=secrets.token_bytes(16)
    digest=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,600000)
    return 'pbkdf2-sha256$'+salt.hex()+'$'+digest.hex()


def password_matches(password,stored):
    try:
        algorithm,salt,digest=stored.split('$')
        if algorithm!='pbkdf2-sha256':return False
        actual=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),600000)
        return hmac.compare_digest(actual,bytes.fromhex(digest))
    except (ValueError,TypeError):return False


def loopback(host):
    if host=='localhost':return True
    try:return ipaddress.ip_address(host).is_loopback
    except ValueError:return False


def local_enabled(request: Request):
    if os.getenv('QXAI_ENABLE_LOCAL_ADMIN')!='true' or os.getenv('QXAI_ENV','production')!='development':return False
    origin=urlsplit(request.headers.get('origin',''))
    return bool(request.client and loopback(request.client.host) and loopback(request.url.hostname or '')
                and origin.scheme=='http' and loopback(origin.hostname or '')
                and origin.port in (5173,5174))


def local_session(request: Request):
    if not local_enabled(request):return False
    token=request.cookies.get(COOKIE,'')
    key=hashlib.sha256(token.encode()).hexdigest()
    expires=_sessions.get(key,0)
    if expires<time.time():
        _sessions.pop(key,None);return False
    return True


def login(request,username,password):
    if not local_enabled(request):raise HTTPException(403,'Local admin login is disabled here')
    now=time.time();history=_attempts[request.client.host]
    while history and history[0]<now-600:history.popleft()
    if len(history)>=5:raise HTTPException(429,'Too many attempts. Try again in ten minutes.')
    history.append(now)
    if not (hmac.compare_digest(username,os.getenv('QXAI_LOCAL_ADMIN_USERNAME','admin')) and
            password_matches(password,os.getenv('QXAI_LOCAL_ADMIN_PASSWORD_HASH',''))):
        raise HTTPException(401,'Incorrect admin credentials')
    history.clear()
    # Discard expired sessions so the process does not retain an unbounded token list.
    for key,expires in list(_sessions.items()):
        if expires<now:_sessions.pop(key,None)
    token=secrets.token_urlsafe(32)
    _sessions[hashlib.sha256(token.encode()).hexdigest()]=now+3600
    return token


def logout(request):
    token=request.cookies.get(COOKIE,'')
    _sessions.pop(hashlib.sha256(token.encode()).hexdigest(),None)


def local_service_headers():
    # Called only after a validated local admin session. Production uses scoped admin JWTs.
    return worker_settings()
