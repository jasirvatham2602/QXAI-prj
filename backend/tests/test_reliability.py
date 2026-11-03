import asyncio
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).parents[1]))
import httpx
import pytest
from fastapi.testclient import TestClient
import service
import worker
from storage import object_path


def error(status):
    return httpx.HTTPStatusError('test',request=httpx.Request('GET','https://example.invalid'),response=httpx.Response(status))

@pytest.mark.parametrize('exception,expected',[(error(503),True),(error(429),True),(error(403),False),(ValueError('invalid'),False),(subprocess.TimeoutExpired('model',600),False)])
def test_retry_classification(exception,expected):
    assert worker.retryable(exception) is expected


def test_retry_limit_and_backoff():
    now=datetime(2026,11,3,tzinfo=timezone.utc)
    first=worker.failure_update({'attempts':1},error(503),now)
    assert first['status']=='queued' and first['next_attempt_at']=='2026-11-03T00:00:30+00:00'
    assert worker.failure_update({'attempts':3},error(503),now)['status']=='failed'
    assert worker.failure_update({'attempts':1},ValueError(),now)['status']=='failed'


def test_old_attempt_cannot_finalize_reclaimed_job():
    client=httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(lambda request: httpx.Response(200,json=[])))
    w=worker.Worker(client,worker_id='00000000-0000-0000-0000-000000000001')
    with pytest.raises(worker.LeaseLost):
        w.update_job({'id':'test','attempts':1},{'status':'complete'})
    client.close()


def test_inference_subprocess_has_a_hard_deadline(tmp_path):
    runner=Mock(side_effect=subprocess.TimeoutExpired('model',600))
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_model(tmp_path/'input.png',str(tmp_path),runner=runner)
    assert runner.call_args.kwargs['timeout']==600


def test_output_path_cannot_escape_owned_analysis():
    owner='00000000-0000-0000-0000-000000000001'
    aid='00000000-0000-0000-0000-000000000002'
    assert object_path(owner,aid,'saliency',f'{owner}/{aid}/saliency-attempt.png').endswith('saliency-attempt.png')
    for path in ('another-owner/file.png',f'{owner}/{aid}/../other.png',f'{owner}/{aid}/nested/file.png'):
        with pytest.raises(ValueError):object_path(owner,aid,'saliency',path)


def test_readiness_fails_when_worker_is_not_ready(monkeypatch):
    async def fake(*args,**kwargs):return httpx.Response(200,json=False)
    monkeypatch.setattr(service,'request',fake)
    with TestClient(service.app) as c:
        assert c.get('/health/ready').status_code==503


def test_readiness_requires_successful_worker_check(monkeypatch):
    async def fake(*args,**kwargs):return httpx.Response(200,json=True)
    monkeypatch.setattr(service,'request',fake)
    with TestClient(service.app) as c:
        assert c.get('/health/ready').status_code==200


def test_history_validates_dates_and_bounds(monkeypatch):
    calls=[]
    async def fake(*args,**kwargs):
        calls.append(kwargs)
        return httpx.Response(200,json=[])
    service.app.dependency_overrides[service.identity]=lambda:('owner','token')
    monkeypatch.setattr(service,'request',fake)
    try:
        with TestClient(service.app) as c:
            assert c.get('/analyses?start=2026-11-04&end=2026-11-03').status_code==422
            assert c.get('/analyses?limit=1000').status_code==422
            assert c.get('/analyses?start=2026-11-03&end=2026-11-04&offset=50').status_code==200
            params=calls[-1]['params']
            assert ('offset','50') in params
            assert ('created_at','gte.2026-11-03T00:00:00+00:00') in params
    finally:service.app.dependency_overrides.clear()


def test_transient_provider_failure_is_safe_503(monkeypatch):
    original=httpx.AsyncClient
    def transport(request):raise httpx.ConnectError('sensitive upstream detail',request=request)
    monkeypatch.setattr(service.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(transport)))
    with pytest.raises(service.HTTPException) as info:
        asyncio.run(service.request('GET','/rest/v1/analyses','test'))
    assert info.value.status_code==503
    assert 'sensitive' not in info.value.detail


def test_successful_job_records_attempt_and_cleans_temporary_files(monkeypatch):
    requests=[];directories=[]
    def transport(request):
        requests.append(request)
        if request.method=='PATCH':return httpx.Response(200,json=[{'status':'complete'}])
        return httpx.Response(200,content=b'sample')
    def model(source,directory):
        directories.append(directory)
        Path(directory,'saliency.png').write_bytes(b'png')
        return {'classical':{'label':'PD'},'quantum':{'label':'PD'}},{'classical_seconds':1.0}
    monkeypatch.setattr(worker,'run_model',model)
    with httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(transport)) as client:
        w=worker.Worker(client,worker_id='00000000-0000-0000-0000-000000000003')
        w.process({'id':'00000000-0000-0000-0000-000000000002','owner_id':'00000000-0000-0000-0000-000000000001','attempts':2})
    patches=[request for request in requests if request.method=='PATCH']
    import json
    payload=json.loads(patches[0].content)
    assert payload['status']=='complete' and payload['stage_timings']['classical_seconds']==1.0
    assert 'saliency-2-' in payload['output_path']
    assert patches[0].url.params['attempts']=='eq.2'
    assert not Path(directories[0]).exists()


def test_temporary_storage_outage_requeues_current_attempt():
    import json
    changes=[]
    def transport(request):
        if request.method=='GET':raise httpx.ConnectError('network unavailable',request=request)
        changes.append(json.loads(request.content))
        return httpx.Response(200,json=[{'status':'queued'}])
    with httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(transport)) as client:
        w=worker.Worker(client)
        w.process({'id':'test','owner_id':'owner','attempts':1})
    assert changes[0]['status']=='queued' and changes[0]['failure_category']=='data_service_unavailable'


def test_retention_is_read_only_by_default(monkeypatch,capsys):
    import retention
    requests=[]
    def transport(request):
        requests.append(request)
        if request.url.params.get('offset')=='0':return httpx.Response(200,json=[{'id':'expired','owner_id':'owner'}])
        return httpx.Response(200,json=[])
    original=httpx.Client
    monkeypatch.setattr(retention,'worker_settings',lambda:('https://example.invalid',{}))
    monkeypatch.setattr(retention.httpx,'Client',lambda **kw:original(base_url='https://example.invalid',transport=httpx.MockTransport(transport)))
    retention.main([])
    assert all(request.method=='GET' for request in requests)
    assert '"mode": "dry_run"' in capsys.readouterr().out
