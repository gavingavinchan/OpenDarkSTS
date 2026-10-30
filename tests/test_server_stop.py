"""Portable shutdown and process identity checks for the workspace GUI server."""
import os
import subprocess
import sys

import pytest

from scripts import service, workspace


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'WORKSPACE_DIR', tmp_path)
    return tmp_path


def server_record(root, **updates):
    path = root / 'reports' / 'server' / 'service.json'
    record = {'pid': 12345, 'start_ticks': 42, 'workspace': str(root), **updates}
    workspace.atomic_json(path, record)
    return path



@pytest.mark.parametrize('pid,identity,root_ok', [(0, 42, True), (os.getpid(), 42, True), (12345, None, True), (12345, 42, False)])
def test_stop_refuses_invalid_process_identity(isolated, monkeypatch, pid, identity, root_ok):
    server_record(isolated, pid=pid, start_ticks=identity, workspace=str(isolated) if root_ok else '/other')
    monkeypatch.setattr(service, 'process_is_alive', lambda *a: pytest.fail('Invalid identity must not be probed'))
    with pytest.raises(ValueError, match='identity'):
        service.stop()



def test_stop_ignores_stale_reused_pid(isolated, monkeypatch):
    path = server_record(isolated)
    monkeypatch.setattr(service, 'process_is_alive', lambda pid, ticks: False)
    assert not service.stop()['stopped']
    assert not path.exists()
    assert not service.stop()['stopped']



def test_stop_uses_windows_process_tree_abstraction(isolated, monkeypatch):
    path = server_record(isolated)
    alive, stopped = iter([True, False]), []
    monkeypatch.setattr(service.process_support, 'IS_WINDOWS', True)
    monkeypatch.setattr(service, 'process_is_alive', lambda *a: next(alive))
    monkeypatch.setattr(service, 'read_process_start_ticks', lambda pid: 42)
    monkeypatch.setattr(service.process_support, 'stop_windows_tree', lambda *args: stopped.append(args))
    assert service.stop(grace_seconds=.1)['stopped']
    assert stopped == [(12345, 42, .1)]
    assert not path.exists()



@pytest.mark.skipif(os.name == 'nt', reason='Linux process-group integration; Windows path tested above')
def test_stop_terminates_detached_linux_process(isolated):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        ticks = service.read_process_start_ticks(process.pid)
        server_record(isolated, pid=process.pid, start_ticks=ticks)
        assert service.stop(grace_seconds=.2)['stopped']
        process.wait(timeout=3)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)



def test_command_help_is_readable():
    result = subprocess.run([sys.executable, '-m', 'opendarksts', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '==SUPPRESS==' not in result.stdout
    assert 'Stop the recorded GUI server' in result.stdout
    assert 'Review and export detection results' in result.stdout

