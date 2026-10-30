# Windows validation

Training and operational workers launch through the active Python environment.
Windows workers use a new process group; process status uses psutil creation times
instead of `os.kill(pid, 0)`, and stopping training terminates its descendants before
its parent. Linux retains its process-group signalling and `/proc` identity checks.
Submission locks use `msvcrt` on Windows and `flock` on Linux.

The launch, relaunch, failed-run, stop and CLI tests use Python executables and run
on both platforms. Windows-specific identity and descendant handling also have
mocked tests on Linux. See the [psutil API](https://psutil.io/api/) for process
identity and process-tree operations.

Actual Windows execution has not been performed in the development machine's Linux
session. Before release, require the Windows CI job plus a manual Detect run with
the bundled example: initialize, serve, run the example, inspect CSV/JSON/previews,
restart the server and verify the job remains visible. Check training start/stop on
a short fixture, including a workspace path containing spaces. Record the OS,
Python/package versions, commands and outcome in the release review report.
