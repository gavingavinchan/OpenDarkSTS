"""Portable detached process launch and Windows process-tree management."""
from __future__ import annotations

import os
import subprocess

import psutil

IS_WINDOWS = os.name == "nt"


def detached_options():
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {"start_new_session": True}


def windows_start_ticks(pid):
    try:
        return int(psutil.Process(pid).create_time() * 1_000_000)
    except (psutil.Error, ValueError, OverflowError):
        return None


def windows_process_alive(pid, start_ticks=None):
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        return start_ticks is None or int(process.create_time() * 1_000_000) == start_ticks
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Conservatively retain a live status when a process cannot be inspected.
        return True


def stop_windows_tree(pid, start_ticks, grace_seconds):
    if pid <= 0 or pid == os.getpid() or start_ticks is None:
        raise ValueError("A valid process identity is required to stop a Windows job")
    if not windows_process_alive(pid, start_ticks):
        return False
    parent = psutil.Process(pid)
    if int(parent.create_time() * 1_000_000) != start_ticks:
        raise ValueError("Process identity changed")
    processes = parent.children(recursive=True) + [parent]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
    escalated = bool(alive)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    if alive:
        raise RuntimeError("Some job processes could not be stopped")
    return escalated
