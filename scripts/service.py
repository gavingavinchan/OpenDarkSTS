"""Start the single local server as a detached process."""
import json
import os
import subprocess
import sys
import time
import secrets
import signal
from urllib.request import urlopen

from .workspace import WORKSPACE_DIR, atomic_json, contained, exclusive_lock
from . import process_support
from .process_support import detached_options
from .training_pipeline_lib import process_is_alive, read_process_start_ticks


def serve(port=8766):
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    root = contained(WORKSPACE_DIR / "reports" / "server", WORKSPACE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "service.json"
    with exclusive_lock(root / ".launch.lock"):
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if process_is_alive(record.get("pid"), record.get("start_ticks")):
                return {**record, "already_running": True}
        log_path = contained(root / "server.log", root)
        token = secrets.token_hex(16)
        env = {**os.environ, "OPENDARKSTS_SERVER_INSTANCE": token}
        with log_path.open("a", encoding="utf-8") as output:
            child = subprocess.Popen([sys.executable, "-m", "scripts.training_pipeline_ui", "--port", str(port)],
                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env=env, **detached_options())
        record = {"pid": child.pid, "start_ticks": read_process_start_ticks(child.pid),
                  "url": f"http://127.0.0.1:{port}", "log": str(log_path), "workspace": str(WORKSPACE_DIR)}
        for _ in range(50):
            if child.poll() is not None:
                raise RuntimeError(f"Server could not start. Check {log_path}")
            try:
                with urlopen(record["url"] + "/api/operations/health", timeout=.2) as response:
                    if response.status == 200 and json.loads(response.read())["instance"] == token:
                        # Give an occupied port time to fail in the newly launched child.
                        time.sleep(.1)
                        if child.poll() is None:
                            atomic_json(record_path, record)
                            return record
            except OSError:
                pass
            time.sleep(.1)
        child.terminate()
        child.wait(timeout=5)
        raise RuntimeError(f"Server did not become ready. Check {log_path}")


def stop(*, grace_seconds=5):
    """Stop only this workspace's recorded server, guarding against PID reuse."""
    root = contained(WORKSPACE_DIR / "reports" / "server", WORKSPACE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    record_path = contained(root / "service.json", root)
    with exclusive_lock(root / ".launch.lock"):
        if not record_path.is_file():
            return {"stopped": False, "reason": "No server recorded for this workspace"}
        record = json.loads(record_path.read_text())
        pid, identity = record.get("pid"), record.get("start_ticks")
        if (type(pid) is not int or pid <= 0 or pid == os.getpid()
                or type(identity) is not int or identity <= 0
                or record.get("workspace") != str(WORKSPACE_DIR)):
            raise ValueError("Invalid server process identity; no process was stopped")
        if not process_is_alive(pid, identity):
            record_path.unlink()
            return {"stopped": False, "reason": "Recorded server is no longer running"}
        if read_process_start_ticks(pid) != identity:
            raise ValueError("Server process identity cannot be verified; no process was stopped")
        if process_support.IS_WINDOWS:
            process_support.stop_windows_tree(pid, identity, grace_seconds)
        else:
            if os.getpgid(pid) != pid:
                raise ValueError("Recorded server is not a detached process group; no process was stopped")
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline and process_is_alive(pid, identity):
                time.sleep(.1)
            if process_is_alive(pid, identity):
                os.killpg(pid, signal.SIGKILL)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and process_is_alive(pid, identity):
                    time.sleep(.1)
        if process_is_alive(pid, identity):
            raise RuntimeError("Server could not be stopped; its process record was retained")
        record_path.unlink()
        return {"stopped": True, "pid": pid, "workspace": str(WORKSPACE_DIR),
                "note": "Independent background jobs continue; use jobs to inspect them"}
