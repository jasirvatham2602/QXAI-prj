"""Bounded job processing, transient retries, and independent model-readiness heartbeats."""
import json
import logging
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from settings import worker_settings
from model_bundle import load_bundle

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
log = logging.getLogger('qxai.worker')
TIMEOUT_SECONDS = 600  # Shorter than the database's 15-minute abandoned-job threshold.
MAX_ATTEMPTS = 3

class LeaseLost(Exception):
    """The database no longer assigns this attempt to this worker."""

def retryable(error):
    return isinstance(error, httpx.RequestError) or (
        isinstance(error, httpx.HTTPStatusError) and
        (error.response.status_code in (408, 429) or error.response.status_code >= 500)
    )

def failure_update(job, error, now=None):
    now = now or datetime.now(timezone.utc)
    can_retry = retryable(error) and job['attempts'] < MAX_ATTEMPTS
    category = ('processing_timeout' if isinstance(error, subprocess.TimeoutExpired)
                else 'data_service_unavailable' if retryable(error) else 'processing_failed')
    return {
        'status': 'queued' if can_retry else 'failed', 'failure_category': category,
        'next_attempt_at': (now + timedelta(seconds=30 * 2 ** max(0, job['attempts'] - 1))).isoformat(),
        'completed_at': None if can_retry else now.isoformat(),
        'worker_id': None,
    }

def run_model(source, directory, runner=subprocess.run):
    runner([sys.executable, str(Path(__file__).with_name('run_inference.py')), str(source), directory],
           check=True, timeout=TIMEOUT_SECONDS, capture_output=True)
    result = json.loads((Path(directory) / 'result.json').read_text())
    timings = json.loads((Path(directory) / 'timings.json').read_text())
    return result, timings

class Worker:
    def __init__(self, client, worker_id=None):
        self.client = client
        self.id = worker_id or str(uuid.uuid4())
        self.ready = False
        self.stop = threading.Event()
        bundle = load_bundle()
        self.version = bundle['model_version']
        self.preprocessing_version = bundle['preprocessing_version']

    def call(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response

    def update_job(self, job, payload):
        response = self.call('PATCH', '/rest/v1/analyses', params={
            'id': 'eq.' + job['id'], 'status': 'eq.processing',
            'worker_id': 'eq.' + self.id, 'attempts': 'eq.' + str(job['attempts']),
        }, headers={'Prefer': 'return=representation'}, json=payload)
        if not response.json():
            raise LeaseLost()

    def heartbeat(self):
        scale = 1 if sys.platform == 'darwin' else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self.call('POST', '/rest/v1/worker_health',
                  headers={'Prefer': 'resolution=merge-duplicates'}, json={
                      'worker_id': self.id, 'seen_at': datetime.now(timezone.utc).isoformat(),
                      'model_ready': self.ready, 'model_version': self.version,
                      'memory_bytes': int(peak * scale),
                  })

    def heartbeat_loop(self):
        while not self.stop.is_set():
            try:
                self.heartbeat()
            except httpx.HTTPError:
                log.warning('worker_heartbeat_unavailable')
            self.stop.wait(15)
        self.ready = False
        try:
            self.heartbeat()
        except httpx.HTTPError:
            pass

    def process(self, job):
        started = time.monotonic()
        prefix = f"{job['owner_id']}/{job['id']}"
        output_path = f"{prefix}/saliency-{job['attempts']}-{self.id}.png"
        try:
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'input.png'
                source.write_bytes(self.call('GET', '/storage/v1/object/authenticated/analyses/' + prefix + '/input.png').content)
                result, timings = run_model(source, directory)
                self.call('POST', '/storage/v1/object/analyses/' + output_path,
                          content=(Path(directory) / 'saliency.png').read_bytes(),
                          headers={'Content-Type': 'image/png', 'x-upsert': 'true'})
            duration = time.monotonic() - started
            self.update_job(job, {'status': 'complete', 'result': result,
                'duration_seconds': duration, 'model_version': self.version,
                'preprocessing_version': self.preprocessing_version, 'stage_timings': timings,
                'output_path': output_path, 'failure_category': None,
                'completed_at': datetime.now(timezone.utc).isoformat()})
            log.info(json.dumps({'event': 'analysis_complete', 'analysis_id': job['id'],
                                 'attempt': job['attempts'], 'duration_seconds': duration}))
        except LeaseLost:
            log.warning(json.dumps({'event': 'analysis_attempt_superseded', 'analysis_id': job['id']}))
        except Exception as error:
            try:
                self.update_job(job, failure_update(job, error))
            except (httpx.HTTPError, LeaseLost):
                # Recovery is performed in Postgres after the abandoned-job deadline.
                log.warning('job_state_recovery_deferred')
            log.warning(json.dumps({'event': 'analysis_attempt_failed', 'analysis_id': job['id'],
                                    'attempt': job['attempts'], 'retryable': retryable(error)}))

    def run(self):
        thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        thread.start()
        try:
            # Actually load the saved weights and run both models before reporting ready.
            subprocess.run([sys.executable, str(Path(__file__).with_name('run_inference.py')), '--check'],
                           check=True, timeout=TIMEOUT_SECONDS, capture_output=True)
            self.ready = True
            try:
                self.heartbeat()
            except httpx.HTTPError:
                log.warning('worker_initial_heartbeat_unavailable')
            recovered_at = 0.0
            while not self.stop.is_set():
                try:
                    if time.monotonic() - recovered_at >= 60:
                        self.call('POST', '/rest/v1/rpc/recover_stale_analyses', json={})
                        recovered_at = time.monotonic()
                    jobs = self.call('POST', '/rest/v1/rpc/claim_analysis', json={'worker_token': self.id}).json()
                    if jobs:
                        self.process(jobs[0])
                    else:
                        self.stop.wait(5)
                except httpx.HTTPError:
                    log.warning('worker_data_service_unavailable')
                    self.stop.wait(5)
        finally:
            self.stop.set()
            thread.join(timeout=35)


def run():
    url, headers = worker_settings()
    with httpx.Client(base_url=url, headers=headers, timeout=30) as client:
        worker = Worker(client)
        signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
        signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
        worker.run()

if __name__ == '__main__':
    run()
