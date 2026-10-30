import os

import pytest

from scripts import process_support as support, training_pipeline_lib as training


def test_windows_identity_check_does_not_probe_with_os_kill(monkeypatch):
    class Process:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return 123.5
        def is_running(self): return True
        def status(self): return 'running'
    monkeypatch.setattr(support, 'IS_WINDOWS', True)
    monkeypatch.setattr(support.psutil, 'Process', Process)
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('Windows status must not call os.kill'))
    assert training.read_process_start_ticks(123) == 123500000
    assert training.process_is_alive(123, 123500000)
    assert not training.process_is_alive(123, 123500001)


def test_windows_stop_checks_identity_and_stops_descendants(monkeypatch):
    stopped = []
    class Process:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return 100.
        def is_running(self): return True
        def status(self): return 'running'
        def children(self, recursive): return [Process(321)]
        def terminate(self): stopped.append(self.pid)
    monkeypatch.setattr(support.psutil, 'Process', Process)
    monkeypatch.setattr(support.psutil, 'wait_procs', lambda processes, timeout: (processes, []))
    assert support.stop_windows_tree(123, 100000000, .1) is False
    assert stopped == [321, 123]
    stopped.clear()
    assert support.stop_windows_tree(123, 100000001, .1) is False
    assert stopped == []
    with pytest.raises(ValueError):
        support.stop_windows_tree(os.getpid(), 100000000, .1)


def test_launch_lock_prevents_racing_training_submissions(tmp_path, monkeypatch):
    from test_training_run_launch import build_ready_run
    from scripts.workspace import exclusive_lock
    import hashlib
    selection, config, run = build_ready_run(tmp_path)
    token = hashlib.sha256(str(run).encode()).hexdigest()
    with exclusive_lock(training.WORKSPACE_DIR / 'runs' / '.launch_locks' / token):
        with pytest.raises(training.TrainingRunActiveError, match='launch is in progress'):
            training.start_training_run(selection, config)
