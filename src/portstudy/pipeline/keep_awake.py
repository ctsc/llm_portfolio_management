"""Keep the Windows desktop awake while the weekly run works (Phase 3A).

Under WSL the run starts a small `powershell.exe` helper that calls
`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` and keeps calling it until it is
stopped; `run_weekend.py` stops it on every exit path. Outside WSL nothing is started.

# UNVERIFIED: needs confirmation on Carter's desktop (manualwork.md).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

_LOG = logging.getLogger("portstudy.keep_awake")

POWERSHELL_SCRIPT = (
    '$sig = \'[DllImport("kernel32.dll")] public static extern uint '
    "SetThreadExecutionState(uint esFlags);'; "
    "$k = Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -PassThru; "
    "while ($true) { [void]$k::SetThreadExecutionState(0x80000001); Start-Sleep -Seconds 50 }"
)


def is_wsl(proc_version: Path = Path("/proc/version")) -> bool:
    try:
        return "microsoft" in proc_version.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def start_keep_awake() -> subprocess.Popen[bytes] | None:
    """Start the helper under WSL; None when not under WSL or powershell.exe is missing."""
    if not is_wsl():
        return None
    exe = shutil.which("powershell.exe")
    if exe is None:
        _LOG.warning("keep_awake_unavailable", extra={"extra_fields": {"reason": "no powershell"}})
        return None
    proc = subprocess.Popen(
        [exe, "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_SCRIPT],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _LOG.info("keep_awake_started", extra={"extra_fields": {"pid": proc.pid}})
    return proc


def stop_keep_awake(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _LOG.warning("keep_awake_kill", extra={"extra_fields": {"pid": proc.pid}})
        proc.kill()
