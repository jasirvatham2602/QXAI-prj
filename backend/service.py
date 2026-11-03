"""Authenticated research API; Supabase enforces ownership on every data request."""
import io
from functools import lru_cache
import json
import logging
import os
import uuid
import unicodedata
import warnings
from datetime import datetime, timezone
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import settings
from model_bundle import load_bundle
from storage import analysis_prefix, object_path

log = logging.getLogger('qxai')
URL = os.getenv('SUPABASE_URL', '').rstrip('/')
KEY = os.getenv('SUPABASE_PUBLISHABLE_KEY') or os.getenv('SUPABASE_ANON_KEY', '')
BUCKET = 'analyses'
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
bearer = HTTPBearer()

async def request(method, path, token=None, **kwargs):
    if not URL or not KEY:
        raise HTTPException(503, 'Storage and authentication are not configured')
    headers = {'apikey': KEY, **kwargs.pop('headers', {})}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            result = await client.request(method, URL + path, headers=headers, **kwargs)
    except httpx.RequestError:
        raise HTTPException(503, 'Data service temporarily unavailable')
    if result.status_code >= 400:
        status = result.status_code
        messages = {401: 'Sign in again', 403: 'Access denied', 404: 'Not found',
                    409: 'Request conflicts with the current analysis state',
                    429: 'Too many active analyses. Try again later.'}
        log.warning(json.dumps({'event': 'provider_error', 'status': status}))
        raise HTTPException(status if status in messages else 503,
                            messages.get(status, 'Data service temporarily unavailable'))
    return result

async def identity(auth: HTTPAuthorizationCredentials = Depends(bearer)):
    if not URL or not KEY:
        raise HTTPException(503, 'Authentication is not configured')
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            result = await client.get(URL + '/auth/v1/user', headers={
                'apikey': KEY, 'Authorization': 'Bearer ' + auth.credentials})
    except httpx.RequestError:
        raise HTTPException(503, 'Authentication service temporarily unavailable')
    if result.status_code >= 500:
        raise HTTPException(503, 'Authentication service temporarily unavailable')
    if result.status_code != 200:
        raise HTTPException(401, 'Sign in again')
    return result.json()['id'], auth.credentials

app = FastAPI(title='QXAI Research API')
app.add_middleware(CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv('FRONTEND_ORIGINS', 'http://localhost:5173').split(',')],
    allow_credentials=True, allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Authorization', 'Content-Type'])

@app.middleware('http')
async def prevent_private_cache(request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response

@lru_cache(maxsize=1)
def public_model_metadata():
    bundle = load_bundle()
    return {key:bundle[key] for key in ('model_version','preprocessing_version','class_names')}

@app.get('/models')
def model_metadata():
    return {**public_model_metadata(), 'research_only':True}

@app.get('/health/live')
def live():
    return {'status': 'alive'}

@app.get('/health/ready')
async def ready():
    response = await request('POST', '/rest/v1/rpc/worker_is_ready', json={})
    if response.json() is not True:
        raise HTTPException(503, 'Model worker is starting or unavailable')
    return {'status': 'ready', 'models': 'worker self-test passed'}

@app.get('/analyses')
async def history(
    limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0),
    start: Optional[datetime] = None, end: Optional[datetime] = None,
    user=Depends(identity),
):
    def utc(value):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if start and end and utc(start) >= utc(end):
        raise HTTPException(422, 'Start date must be before the end date')
    params = [('select', '*'), ('order', 'created_at.desc,id.desc'), ('limit', str(limit)), ('offset', str(offset))]
    if start:
        params.append(('created_at', 'gte.' + utc(start).isoformat()))
    if end:
        params.append(('created_at', 'lt.' + utc(end).isoformat()))
    return (await request('GET', '/rest/v1/analyses', user[1], params=params)).json()

async def owned(analysis_id, user):
    try:
        analysis_id = str(uuid.UUID(analysis_id))
    except ValueError:
        raise HTTPException(404, 'Analysis not found')
    rows = (await request('GET', '/rest/v1/analyses', user[1],
                          params={'id': 'eq.' + analysis_id, 'select': '*'})).json()
    if not rows:
        raise HTTPException(404, 'Analysis not found')
    return rows[0]

@app.get('/analyses/{analysis_id}')
async def detail(analysis_id: str, user=Depends(identity)):
    return await owned(analysis_id, user)

def upload_display_name(filename):
    name = unicodedata.normalize('NFC', (filename or 'upload.png').replace(chr(92), '/').rsplit('/', 1)[-1])
    name = ''.join(char for char in name if not unicodedata.category(char).startswith('C')).strip()
    return name[:255] or 'upload.png'

@app.post('/analyses', status_code=202)
async def create(image: UploadFile = File(...), user=Depends(identity)):
    original_filename = upload_display_name(image.filename)
    try:
        raw = await image.read(MAX_BYTES + 1)
    finally:
        await image.close()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Maximum upload size is 8 MB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                if img.format not in ('PNG', 'JPEG') or min(img.size) < 32 or img.width * img.height > MAX_PIXELS:
                    raise ValueError()
                img.load()
                output = io.BytesIO()
                img.convert('RGB').save(output, format='PNG')
                data = output.getvalue()
                if len(data) > MAX_BYTES:
                    raise ValueError()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, 'Use a valid PNG or JPEG, at least 32 pixels per side and at most 16 megapixels')
    analysis_id = str(uuid.uuid4())
    path = f'{user[0]}/{analysis_id}/input.png'
    await request('POST', '/storage/v1/object/' + BUCKET + '/' + path, user[1],
                  content=data, headers={'Content-Type': 'image/png'})
    try:
        row = (await request('POST', '/rest/v1/rpc/enqueue_named_analysis', user[1],
                             json={'new_id': analysis_id, 'upload_name': original_filename})).json()
    except HTTPException:
        # Remove only the object created by this request; preserve the original error.
        try:
            await request('DELETE', '/storage/v1/object/' + BUCKET, user[1], json={'prefixes': [path]})
        except HTTPException:
            log.warning(json.dumps({'event': 'upload_cleanup_required', 'analysis_id': analysis_id}))
        raise
    return row

@app.get('/analyses/{analysis_id}/images/{kind}')
async def image(analysis_id: str, kind: str, user=Depends(identity)):
    row = await owned(analysis_id, user)
    if kind not in ('input', 'saliency') or (kind == 'saliency' and row['status'] != 'complete'):
        raise HTTPException(404)
    try:
        path = object_path(user[0], row['id'], kind, row.get('output_path'))
    except ValueError:
        raise HTTPException(404)
    result = await request('GET', '/storage/v1/object/authenticated/' + BUCKET + '/' + path, user[1])
    return Response(result.content, media_type='image/png', headers={'Cache-Control': 'no-store'})

@app.delete('/analyses/{analysis_id}', status_code=204)
async def delete(analysis_id: str, user=Depends(identity)):
    row = await owned(analysis_id, user)
    if row['status'] in ('queued', 'processing'):
        raise HTTPException(409, 'Wait for processing to finish before deleting')
    prefix = analysis_prefix(user[0], row['id'])
    # List all attempt outputs so retries do not leave orphaned images.
    objects = (await request('POST', '/storage/v1/object/list/' + BUCKET, user[1],
                             json={'prefix': prefix + '/', 'limit': 100})).json()
    paths = [prefix + '/' + obj['name'] for obj in objects
             if obj.get('id') and '/' not in obj['name'] and '..' not in obj['name']]
    if paths:
        await request('DELETE', '/storage/v1/object/' + BUCKET, user[1], json={'prefixes': paths})
    await request('DELETE', '/rest/v1/analyses', user[1], params={'id': 'eq.' + row['id']})

@app.get('/operations')
async def operations(user=Depends(identity)):
    return (await request('POST', '/rest/v1/rpc/analysis_metrics', user[1], json={})).json()

# Admin scan review. Local credentials never authorize access on a production server.
from fastapi import Request
from pydantic import BaseModel, Field
import admin_access
admin_bearer = HTTPBearer(auto_error=False)

class AdminLogin(BaseModel):
    username: str = Field(min_length=1,max_length=100)
    password: str = Field(min_length=1,max_length=200)

async def admin_principal(request: Request,credentials=Depends(admin_bearer)):
    if admin_access.local_session(request):return ('local',None)
    if credentials:
        user=await identity(credentials)
        return ('supabase',user[1])
    raise HTTPException(401,'Admin sign-in required')

async def admin_request(method,path,principal,**kwargs):
    if principal[0]!='local':return await request(method,path,principal[1],**kwargs)
    url,headers=admin_access.local_service_headers()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            result=await client.request(method,url+path,headers=headers,**kwargs)
    except httpx.RequestError:raise HTTPException(503,'Admin data service temporarily unavailable')
    if result.status_code>=400:
        raise HTTPException(403 if result.status_code in (401,403) else 503,'Admin data request failed')
    return result

@app.get('/admin/session')
async def admin_session(request: Request):
    return {'local_login_enabled':admin_access.local_enabled(request),'authenticated':admin_access.local_session(request)}

@app.post('/admin/login')
async def admin_login(body: AdminLogin,request: Request,response: Response):
    token=admin_access.login(request,body.username,body.password)
    response.set_cookie(admin_access.COOKIE,token,max_age=3600,httponly=True,samesite='strict',secure=False,path='/admin')
    return {'authenticated':True,'local_only':True}

@app.post('/admin/logout',status_code=204)
async def admin_logout(request: Request,response: Response):
    admin_access.logout(request)
    response.delete_cookie(admin_access.COOKIE,path='/admin')

@app.get('/admin/analyses')
async def admin_history(limit: int=Query(50,ge=1,le=100),offset: int=Query(0,ge=0),principal=Depends(admin_principal)):
    return (await admin_request('POST','/rest/v1/rpc/admin_analysis_history',principal,json={'page_limit':limit,'page_offset':offset})).json()

async def admin_owned(analysis_id,principal):
    try:analysis_id=str(uuid.UUID(analysis_id))
    except ValueError:raise HTTPException(404,'Analysis not found')
    row=(await admin_request('POST','/rest/v1/rpc/admin_analysis_detail',principal,json={'analysis_id':analysis_id})).json()
    if not row:raise HTTPException(404,'Analysis not found')
    return row

@app.get('/admin/analyses/{analysis_id}')
async def admin_detail(analysis_id: str,principal=Depends(admin_principal)):
    return await admin_owned(analysis_id,principal)

@app.get('/admin/analyses/{analysis_id}/images/{kind}')
async def admin_image(analysis_id: str,kind: str,principal=Depends(admin_principal)):
    row=await admin_owned(analysis_id,principal)
    if kind not in ('input','saliency') or (kind=='saliency' and row['status']!='complete'):raise HTTPException(404)
    try:path=object_path(row['owner_id'],row['id'],kind,row.get('output_path'))
    except ValueError:raise HTTPException(404)
    image=(await admin_request('GET','/storage/v1/object/authenticated/analyses/'+path,principal)).content
    return Response(image,media_type='image/png',headers={'Cache-Control':'no-store'})
