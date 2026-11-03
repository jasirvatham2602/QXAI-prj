import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import service

@pytest.fixture
def client():
    service.app.dependency_overrides[service.identity] = lambda: ('owner','token')
    with TestClient(service.app) as c: yield c
    service.app.dependency_overrides.clear()

def test_auth_required():
    with TestClient(service.app) as c:
        assert c.get('/analyses').status_code in (401,403)

def test_bad_image(client):
    assert client.post('/analyses',files={'image':('bad.png',b'not an image','image/png')}).status_code==422

def test_size_limit(client):
    assert client.post('/analyses',files={'image':('big.png',b'x'*(service.MAX_BYTES+1),'image/png')}).status_code==413

def test_missing_or_foreign_record(client,monkeypatch):
    async def fake(*a,**kw):
        class R:
            def json(self): return []
        return R()
    monkeypatch.setattr(service,'request',fake)
    assert client.get('/analyses/00000000-0000-0000-0000-000000000001').status_code==404
    assert client.get('/analyses/not-a-uuid').status_code==404

def test_active_delete_rejected(client,monkeypatch):
    async def fake(*a): return {'status':'processing'}
    monkeypatch.setattr(service,'owned',fake)
    assert client.delete('/analyses/id').status_code==409

def test_upload_uses_generated_path(client,monkeypatch):
    calls=[]
    async def fake(method,path,token,**kwargs):
        calls.append((method,path,kwargs))
        class R:
            def json(self): return {'id':'new','status':'queued'}
        return R()
    monkeypatch.setattr(service,'request',fake)
    b=io.BytesIO();Image.new('RGB',(32,32)).save(b,format='PNG')
    r=client.post('/analyses',files={'image':('../../private.png',b.getvalue(),'image/png')})
    assert r.status_code==202
    assert '/owner/' in calls[0][1] and '..' not in calls[0][1]

def test_public_static_route_removed(client):
    assert client.get('/saliency_maps/example.png').status_code==404

def test_live_without_configuration(client):
    assert client.get('/health/live').json()['status']=='alive'


def test_upload_filename_metadata_is_sanitized_without_changing_storage_path(client,monkeypatch):
    calls=[]
    async def fake(method,path,token,**kwargs):
        calls.append((method,path,kwargs))
        class R:
            def json(self):return {'id':'new','original_filename':kwargs.get('json',{}).get('upload_name')}
        return R()
    monkeypatch.setattr(service,'request',fake)
    data=io.BytesIO();Image.new('RGB',(32,32)).save(data,format='PNG')
    result=client.post('/analyses',files={'image':('../../PD_sample.png',data.getvalue(),'image/png')})
    assert result.status_code==202
    assert result.json()['original_filename']=='PD_sample.png'
    assert calls[1][1]=='/rest/v1/rpc/enqueue_named_analysis'
    assert 'PD_sample.png' not in calls[0][1]


def test_filename_display_handles_windows_paths_and_controls():
    assert service.upload_display_name('C:\\images\\AD\n_sample.png')=='AD_sample.png'
    assert len(service.upload_display_name('a'*300+'.png'))==255
    assert service.upload_display_name('')=='upload.png'
