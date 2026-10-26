"""Sandbox: path guard, scrubbed environment, command blocklist, and a process runner.

What it protects against (and what it does NOT):
- safe_path() keeps the FILE TOOLS inside the run folder.
- run_python() / run_command() run code in a child process with a time limit, no stdin, a
  scrubbed environment (no API keys) and a bounded output buffer; the whole process tree is
  killed on timeout or Stop.
- blocked_reason() is a SECOND layer for shell commands; the real protection for run_shell
  is that the user must click Approve for every command.
- It is NOT a hard sandbox: the child runs as the same Windows user, so code it runs could
  still open files outside the folder. Say so in the report (Docker would be the next step).
"""
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psutil

DEFAULT_TIMEOUT = 30          # seconds for run_python
SHELL_TIMEOUT = 120           # seconds for run_shell (pip install can take a while)
MAX_CAPTURE = 10_000          # characters of output we keep (the END of the output)
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")

# (pattern, reason). Matched case-insensitively anywhere in the command line.
BLOCKLIST = [
    (r"\brm\s+(-\S+\s+)*-\S*[rf]", "rm with -r or -f"),
    (r"\b(del|erase)\s+(/\S+\s+)*/[sq]\b", "del with /s or /q"),
    (r"\b(rmdir|rd)\s+(/\S+\s+)*/s\b", "rmdir /s"),
    (r"\bformat\s+[a-z]:", "formatting a drive"),
    (r"\b(shutdown|restart-computer|stop-computer)\b", "shutdown or restart"),
    (r"\breg(\.exe)?\s+(add|delete|import|load|unload|save|restore)\b", "registry change"),
    (r"\bdiskpart\b", "diskpart"),
    (r"\bpowershell(\.exe)?\b.*\s-(e|enc|encodedcommand)\b", "encoded PowerShell"),
    (r"\binvoke-expression\b|\biex\b", "invoke-expression"),
    (r"\b(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b.*\|\s*(sh|bash|zsh|iex|powershell|pwsh|python)\b",
     "download piped into a shell"),
    (r"\bnet\s+(user|localgroup)\b", "user account change"),
    (r"\bsudo\b", "sudo"),
    (r"\bmkfs\b|\bdd\s+if=", "raw disk write"),
]


def blocked_reason(command: str) -> str:
    """Empty string if the command passes the safety list, else why it is blocked."""
    for pattern, why in BLOCKLIST:
        if re.search(pattern, command or "", re.IGNORECASE):
            return why
    return ""


# ------------------------------------------------------------------ path guard
def safe_path(root, rel) -> Path:
    """Return the absolute path of `rel` inside `root`, or raise PermissionError.

    The final check is done on the RESOLVED path, so '..', absolute paths, drive letters
    and symlinks that point outside the folder are all refused."""
    if not isinstance(rel, str) or not rel.strip():
        raise PermissionError("path is empty")
    if "\x00" in rel:
        raise PermissionError("path contains a null character")
    if ":" in rel:                                  # drive letters (C:\..) and NTFS streams
        raise PermissionError(f"path escapes workspace: {rel}")
    base = Path(root).resolve()
    p = (base / rel).resolve()
    if not p.is_relative_to(base):
        raise PermissionError(f"path escapes workspace: {rel}")
    return p


# ------------------------------------------------------------------ environment
def scrubbed_env(drop_names: Iterable[str] = ()) -> dict:
    """Copy of os.environ without secrets: names ending in _KEY, _TOKEN, _SECRET or
    _PASSWORD, plus every name in `drop_names` (we pass the names found in .env)."""
    drop = {n.upper() for n in drop_names}
    return {k: v for k, v in os.environ.items()
            if k.upper() not in drop and not k.upper().endswith(SECRET_SUFFIXES)}


def shell_env(drop_names: Iterable[str] = ()) -> dict:
    """Scrubbed environment for shell commands: the folder of the running Python (the venv's
    Scripts folder on Windows) comes first in PATH so `pip` is the project's pip, and pip
    prints no progress bars."""
    env = scrubbed_env(drop_names)
    key = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[key] = os.path.dirname(sys.executable) + os.pathsep + env.get(key, "")
    env.update({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_PROGRESS_BAR": "off", "PYTHONUTF8": "1"})
    return env


# ------------------------------------------------------------------ process tree
def kill_tree(pid: int) -> None:
    """Kill a process and all its children (children first). Never raises."""
    try:
        parent = psutil.Process(pid)
        victims = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for proc in victims:
        try:
            proc.kill()
        except psutil.Error:
            pass


# ------------------------------------------------------------------ runner
@dataclass
class RunResult:
    output: str                  # stdout and stderr together; the last MAX_CAPTURE characters
    total_chars: int             # how much the program really printed
    exit_code: Optional[int]     # None when it was killed
    timed_out: bool
    stopped: bool
    seconds: float
    truncated: bool = False      # True when the program printed more than MAX_CAPTURE characters

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.stopped


_LINE_ENDS = re.compile(r"\r*\n|\r")


def normalise_newlines(s: str) -> str:
    r"""Every line ending becomes a single \n.

    Any run of \r followed by \n counts as ONE line end (not only \r\n). On Windows a child
    that prints text containing \r\n through Python's text-mode stdout sends \r\r\n; replacing
    only \r\n would leave a \r behind that turns into a blank line. A lone \r (pip progress
    lines, spinners) also becomes a line break."""
    return _LINE_ENDS.sub("\n", s)


class _Capture(threading.Thread):
    """Reads the child's output so the pipe never blocks, keeping only the last chars."""

    def __init__(self, stream, limit: int):
        super().__init__(daemon=True)
        self.stream, self.limit = stream, limit
        self.buf = bytearray()
        self.total = 0
        self.truncated = False

    def run(self) -> None:
        keep = self.limit * 4                       # bytes; a UTF-8 character is at most 4
        try:
            for chunk in iter(lambda: self.stream.read(4096), b""):
                self.total += len(chunk)
                self.buf += chunk
                if len(self.buf) > keep * 2:
                    del self.buf[:-keep]
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        s = bytes(self.buf).decode("utf-8", errors="replace")
        s = normalise_newlines(s)                            # Windows line ends, pip progress lines
        self.truncated = len(s) > self.limit
        return s[-self.limit:]


def run_command(root, cmd, timeout: float, stop_event=None, env: Optional[dict] = None,
                shell: bool = False, hide_path: bool = True) -> RunResult:
    """Run `cmd` (a list, or a string when shell=True) with cwd=root. Blocks until it ends,
    the timeout passes, or stop_event is set; then kills the whole process tree."""
    root = Path(root).resolve()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)          # Windows only; 0 elsewhere
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=root, shell=shell, env=env if env is not None else scrubbed_env(),
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, creationflags=flags)
    cap = _Capture(p.stdout, MAX_CAPTURE)
    cap.start()
    timed_out = stopped = False
    while p.poll() is None:
        if stop_event is not None and stop_event.is_set():
            stopped = True
        elif time.time() - t0 > timeout:
            timed_out = True
        if stopped or timed_out:
            kill_tree(p.pid)
            break
        time.sleep(0.05)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    cap.join(timeout=2)
    out = cap.text()
    if hide_path:                                               # shorter, and no personal path in logs
        for r in {str(root), str(root).replace("\\", "/")}:
            out = out.replace(r, "<workspace>")
    code = None if (timed_out or stopped) else p.returncode
    return RunResult(out, cap.total, code, timed_out, stopped, round(time.time() - t0, 2),
                     truncated=cap.truncated)


def run_python(root, script: str, timeout: float = DEFAULT_TIMEOUT, stop_event=None,
               env: Optional[dict] = None, hide_path: bool = True) -> RunResult:
    """Run the workspace file `script` with the current Python.

    Flags: -E ignores PYTHON* variables, -s skips user site-packages, -X utf8 makes the
    output UTF-8 on Windows, -u shows output at once (so nothing is lost when we kill it).
    We do NOT use -I: isolated mode would stop the script from importing the other files
    in its own folder, which every 'tests.py imports solution.py' task needs."""
    root = Path(root).resolve()
    path = safe_path(root, script)
    if not path.is_file():
        raise FileNotFoundError(f"no such file in the workspace: {script}")
    cmd = [sys.executable, "-E", "-s", "-X", "utf8", "-u", str(path)]
    return run_command(root, cmd, timeout, stop_event, env, shell=False, hide_path=hide_path)