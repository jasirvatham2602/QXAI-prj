import pytest
import settings


def test_modern_secret_is_not_used_as_bearer(monkeypatch):
    monkeypatch.setattr(settings, 'load_dotenv', lambda *args: None)
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co/')
    monkeypatch.setenv('SUPABASE_SECRET_KEY', 'sb_secret_test_only')
    url, headers = settings.worker_settings()
    assert url == 'https://example.supabase.co'
    assert headers == {'apikey': 'sb_secret_test_only'}


def test_legacy_key_compatibility(monkeypatch):
    monkeypatch.setattr(settings, 'load_dotenv', lambda *args: None)
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.delenv('SUPABASE_SECRET_KEY', raising=False)
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY', 'test-only-legacy-key')
    _, headers = settings.worker_settings()
    assert headers['Authorization'] == 'Bearer test-only-legacy-key'


def test_worker_requires_secret(monkeypatch):
    monkeypatch.setattr(settings, 'load_dotenv', lambda *args: None)
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.delenv('SUPABASE_SECRET_KEY', raising=False)
    monkeypatch.delenv('SUPABASE_SERVICE_ROLE_KEY', raising=False)
    with pytest.raises(RuntimeError, match='SUPABASE_SECRET_KEY'):
        settings.worker_settings()
