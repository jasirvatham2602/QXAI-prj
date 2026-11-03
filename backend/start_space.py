"""Supervise API + one worker together on the existing free CPU Space."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).parent

def main():
    children=[]
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        for child in children:
            if child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        children.append(subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','0.0.0.0','--port',os.getenv('PORT','7860'),'--no-access-log'],cwd=ROOT))
        children.append(subprocess.Popen([sys.executable,'worker.py'],cwd=ROOT))
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('A required QXAI service stopped')
            time.sleep(1)
    finally:
        stop()
        deadline=time.monotonic()+40
        for child in children:
            try: child.wait(timeout=max(0.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill();child.wait()

if __name__=='__main__': main()
