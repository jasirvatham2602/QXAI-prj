import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest
from fastapi.testclient import TestClient
import admin_access
import service

ORIGIN={'Origin':'http://127.0.0.1:5174'}

@pytest.fixture
def local_client(monkeypatch):
    monkeypatch.setenv('QXAI_ENV','development')
    monkeypatch.setenv('QXAI_ENABLE_LOCAL_ADMIN','true')
    monkeypatch.setenv('QXAI_LOCAL_ADMIN_USERNAME','admin')
    monkeypatch.setenv('QXAI_LOCAL_ADMIN_PASSWORD_HASH',admin_access.password_hash('password'))
    admin_access._sessions.clear();admin_access._attempts.clear()
    with TestClient(service.app,base_url='http://127.0.0.1:7860',client=('127.0.0.1',50000)) as client:
        yield client
    service.app.dependency_overrides.clear()


def test_local_password_login_is_disabled_in_production(local_client,monkeypatch):
    monkeypatch.setenv('QXAI_ENV','production')
    assert local_client.post('/admin/login',headers=ORIGIN,json={'username':'admin','password':'password'}).status_code==403


def test_wrong_password_and_untrusted_origin_fail(local_client):
    assert local_client.post('/admin/login',headers=ORIGIN,json={'username':'admin','password':'wrong'}).status_code==401
    assert local_client.post('/admin/login',headers={'Origin':'https://untrusted.example'},json={'username':'admin','password':'password'}).status_code==403


def test_local_login_creates_http_only_session(local_client):
    result=local_client.post('/admin/login',headers=ORIGIN,json={'username':'admin','password':'password'})
    assert result.status_code==200
    cookie=result.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'samesite=strict' in cookie
    assert local_client.get('/admin/session',headers=ORIGIN).json()['authenticated'] is True
    local_client.post('/admin/logout',headers=ORIGIN)
    assert local_client.get('/admin/session',headers=ORIGIN).json()['authenticated'] is False


def test_admin_history_requires_session(local_client):
    assert local_client.get('/admin/analyses',headers=ORIGIN).status_code==401


def test_admin_login_rate_limit(local_client):
    for _ in range(5):
        assert local_client.post('/admin/login',headers=ORIGIN,json={'username':'admin','password':'incorrect'}).status_code==401
    assert local_client.post('/admin/login',headers=ORIGIN,json={'username':'admin','password':'password'}).status_code==429


def test_public_demo_login_requires_enabled_flag_and_allowed_origin(local_client,monkeypatch):
    monkeypatch.setenv('QXAI_ENV','production')
    monkeypatch.setenv('QXAI_ENABLE_DEMO_ADMIN','true')
    monkeypatch.setenv('FRONTEND_ORIGINS','https://blue-qxai-prj.netlify.app')
    monkeypatch.setenv('QXAI_DEMO_ADMIN_PASSWORD_HASH',admin_access.password_hash('password'))
    headers={'Origin':'https://blue-qxai-prj.netlify.app'}
    result=local_client.post('/admin/login',headers=headers,json={'username':'admin','password':'password'})
    assert result.status_code==200
    assert 'samesite=none' in result.headers['set-cookie'].lower()
    assert 'secure' in result.headers['set-cookie'].lower()
    assert local_client.get('/admin/session',headers=headers).json()['authenticated'] is True
    assert local_client.post('/admin/login',headers={'Origin':'https://attacker.example'},json={'username':'admin','password':'password'}).status_code==403
