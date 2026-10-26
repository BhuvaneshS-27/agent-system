"""File tools and run_python for ONE run folder (workspace/run_<id>/).

Every method returns a ToolResult(ok, text). `text` is exactly what the model gets as its
observation. Expected problems (bad path, missing file, syntax error, traceback) are
returned as text, never raised, so the agent loop can show them and the model can fix them.
"""
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from agents.coding.sandbox import (DEFAULT_TIMEOUT, SHELL_TIMEOUT, RunResult, blocked_reason, run_command,
                                   safe_path, scrubbed_env, shell_env)
from agents.coding.sandbox import run_python as sandbox_run_python
from core.settings import ROOT
from research.errors import WebStopped
from research.fetch import format_page
from research.search import format_results

OBS_LIMIT = 2000            # characters of any observation given to the model
READ_LIMIT = 4000           # read_file may show a little more
MAX_WRITE = 200_000         # refuse bigger files
MAX_LIST = 200
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


@dataclass
class ToolResult:
    ok: bool
    text: str


def clip(text: str, limit: int = OBS_LIMIT, keep: str = "end") -> str:
    """Shorten text for the model. keep='end' suits program output (the traceback is last)."""
    if len(text) <= limit:
        return text
    cut = len(text) - limit
    if keep == "end":
        return f"[{cut} earlier characters omitted]\n{text[-limit:]}"
    return f"{text[:limit]}\n[{cut} more characters omitted]"


def syntax_error_text(name: str, e: Exception) -> str:
    """One readable block: message, line number, the offending line and a caret."""
    if not isinstance(e, SyntaxError):                    # e.g. a null byte in the source
        return f"{type(e).__name__}: {e}"
    lines = [f"{type(e).__name__}: {e.msg} ({name}, line {e.lineno})"]
    if e.text:
        lines.append("    " + e.text.rstrip("\r\n"))
        if e.offset:
            lines.append("    " + " " * (e.offset - 1) + "^")
    return "\n".join(lines)


class Toolbox:
    def __init__(self, root, bus=None, tags=None, stop_event=None, timeout: float = DEFAULT_TIMEOUT,
                 drop_env=(), approvals=None, job=None, shell_timeout: float = SHELL_TIMEOUT, web=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.bus = bus
        self.tags = dict(tags or {})                  # job, agent, step: the loop updates "step"
        self.stop_event = stop_event
        self.timeout = timeout
        self.drop_env = tuple(drop_env)
        self.approvals = approvals                    # Approvals object; None = run_shell is unavailable
        self.job = job
        self.shell_timeout = shell_timeout
        self.web = web                                # WebTools object; None = web tools are unavailable

    @classmethod
    def for_job(cls, settings, bus, job, **kw):
        """A fresh folder workspace/run_<job id>/ and a toolbox bound to the job."""
        return cls(settings.workspace_dir / f"run_{job.id}", bus,
                   tags={"job": job.id, "agent": job.agent}, stop_event=job.stop_event,
                   drop_env=dotenv_values(ROOT / ".env").keys(), job=job, **kw)

    def _emit(self, kind: str, **data) -> None:
        if self.bus:
            self.bus.emit(kind, **{**self.tags, **data})

    def _path(self, tool: str, rel):
        """safe_path, or None after logging a 'blocked' event."""
        try:
            return safe_path(self.root, rel)
        except PermissionError as e:
            self._emit("blocked", tool=tool, path=str(rel)[:200], reason=str(e))
            return None

    def _refused(self, rel) -> ToolResult:
        return ToolResult(False, f"ERROR: path not allowed: {str(rel)[:100]!r}. "
                                 "Use a relative path inside the workspace, like 'solution.py'.")

    # ---------------------------------------------------------------- files
    def list_files(self, path: str = ".") -> ToolResult:
        p = self._path("list_files", path)
        if p is None:
            return self._refused(path)
        if not p.exists():
            return ToolResult(False, f"ERROR: no such folder: {path}")
        if p.is_file():
            return ToolResult(True, f"{path} (file, {p.stat().st_size} bytes)")
        rows, more = [], False
        for f in sorted(p.rglob("*")):
            rel = f.relative_to(self.root)
            if SKIP_DIRS & set(rel.parts):
                continue
            if len(rows) >= MAX_LIST:
                more = True
                break
            rows.append(f"{rel.as_posix()}/" if f.is_dir() else f"{rel.as_posix()}  ({f.stat().st_size} bytes)")
        if not rows:
            return ToolResult(True, "(the folder is empty)")
        return ToolResult(True, clip("\n".join(rows) + ("\n... more files not shown" if more else ""),
                                     keep="start"))

    def read_file(self, path: str) -> ToolResult:
        p = self._path("read_file", path)
        if p is None:
            return self._refused(path)
        if not p.exists():
            return ToolResult(False, f"ERROR: no such file: {path}")
        if p.is_dir():
            return ToolResult(False, f"ERROR: {path} is a folder; use list_files")
        data = p.read_bytes()
        if b"\x00" in data[:2000]:
            return ToolResult(False, f"ERROR: {path} looks binary ({len(data)} bytes); cannot show it")
        text = data.decode("utf-8", errors="replace")
        return ToolResult(True, clip(text, READ_LIMIT, keep="start") if text else "(the file is empty)")

    def write_file(self, path: str, content: str) -> ToolResult:
        """Writes the file. For every .py file the code is also compiled (never run): if it
        has a syntax error the file is still saved, but the observation is an ERROR that
        names the line, so the model fixes it before running anything."""
        p = self._path("write_file", path)
        if p is None:
            return self._refused(path)
        if not isinstance(content, str):
            return ToolResult(False, "ERROR: content must be text")
        if len(content) > MAX_WRITE:
            return ToolResult(False, f"ERROR: file too large ({len(content)} characters, limit {MAX_WRITE})")
        if p.is_dir():
            return ToolResult(False, f"ERROR: {path} is a folder")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
        rel = p.relative_to(self.root).as_posix()
        size = len(content.encode("utf-8"))
        syntax = ""
        if p.suffix.lower() == ".py":
            try:
                compile(content, rel, "exec", dont_inherit=True)
            except (SyntaxError, ValueError, RecursionError) as e:
                syntax = syntax_error_text(rel, e)
        self._emit("file_written", path=rel, bytes=size, syntax_ok=not syntax,
                   syntax_error=syntax, content=content[:20000])
        if syntax:
            return ToolResult(False, f"ERROR: wrote {rel} ({size} bytes) but it has a syntax error:\n"
                                     f"{syntax}\nFix it and write the file again before running it.")
        return ToolResult(True, f"OK: wrote {rel} ({size} bytes)" + (", syntax OK" if p.suffix.lower() == ".py" else ""))

    # ---------------------------------------------------------------- python
    def run_python(self, code: str = "", path: str = "") -> ToolResult:
        """Run `code` (saved as _run.py first, so tracebacks show real line numbers) or an
        existing workspace file `path`. 30 s limit, no stdin, keys removed from the environment."""
        if path:
            p = self._path("run_python", path)
            if p is None:
                return self._refused(path)
            script = p.relative_to(self.root).as_posix()
            if not p.is_file():
                return ToolResult(False, f"ERROR: no such file: {path}")
        elif code and isinstance(code, str):
            script = "_run.py"
            (self.root / script).write_text(code, encoding="utf-8", newline="\n")
        else:
            return ToolResult(False, "ERROR: run_python needs either 'code' or 'path'")

        self._emit("terminal_start", cmd=f"python {script}", timeout=self.timeout, folder=self.root.name)
        res = sandbox_run_python(self.root, script, self.timeout, self.stop_event,
                                 env=scrubbed_env(self.drop_env))
        self._emit("terminal", cmd=f"python {script}", output=res.output, total_chars=res.total_chars, truncated=res.truncated,
                   exit_code=res.exit_code, timed_out=res.timed_out, stopped=res.stopped,
                   seconds=res.seconds, ok=res.ok)
        return ToolResult(res.ok, observation(res, self.timeout))

    # ---------------------------------------------------------------- shell
    def run_shell(self, command: str) -> ToolResult:
        """Three gates, in this order: (1) the blocklist refuses obviously destructive commands
        and logs a 'blocked' event; (2) the user must click Approve (the agent thread waits);
        (3) the command runs in the run folder with a scrubbed environment and a time limit."""
        step = self.tags.get("step", 0)
        why = blocked_reason(command)
        if why:
            self._emit("blocked", tool="run_shell", command=command[:300], reason=f"safety list: {why}")
            return ToolResult(False, f"BLOCKED by the safety list ({why}). The command was not run and "
                                     "was not even shown to the user. Choose a safer way.")
        if self.approvals is None or self.job is None:
            return ToolResult(False, "ERROR: run_shell is not available in this session.")

        decision = self.approvals.ask(self.job, step, command)
        if decision == "stopped":
            return ToolResult(False, "STOPPED: the user pressed Stop while the command waited for approval.")
        if decision != "approved":
            how = "did not answer in time" if decision == "timeout" else "pressed Deny"
            return ToolResult(False, f"DENIED: the user {how}; the command was not run. "
                                     "Do not ask for the same command again. Find another way, or "
                                     "finish and explain what could not be done.")

        shown = f"{command[:200]}"
        self._emit("terminal_start", cmd=shown, timeout=self.shell_timeout, folder=self.root.name)
        res = run_command(self.root, command, self.shell_timeout, self.stop_event,
                          env=shell_env(self.drop_env), shell=True)
        self._emit("terminal", cmd=shown, output=res.output, total_chars=res.total_chars,
                   truncated=res.truncated, exit_code=res.exit_code, timed_out=res.timed_out,
                   stopped=res.stopped, seconds=res.seconds, ok=res.ok)
        return ToolResult(res.ok, observation(res, self.shell_timeout))

    # ---------------------------------------------------------------- web
    def _web_note(self, out, what: str) -> None:
        """Remember in the job record which provider answered (shown later as a status chip)."""
        if self.job is not None:
            self.job.attempts += max(out.attempts, 0)
            if out.ok:
                self.job.provider_used = f"{what}: {out.provider}"

    def web_search(self, query: str) -> ToolResult:
        """Search chain Tavily -> Exa -> Serper -> ddgs, with cache. Returns 5 results as text."""
        if self.web is None:
            return ToolResult(False, "ERROR: web tools are not available in this session.")
        try:
            out = self.web.search(query, max_results=5, tags=self.tags, stop_event=self.stop_event)
        except WebStopped:
            return ToolResult(False, "STOPPED: the user pressed Stop during the web search.")
        self._web_note(out, "search")
        if not out.ok:
            return ToolResult(False, f"ERROR: web search failed ({out.error}). Try other words, or "
                                     "finish and explain that the web is not reachable.")
        return ToolResult(True, format_results(out))

    def fetch_page(self, url: str, start: int = 0) -> ToolResult:
        """Page chain trafilatura -> Jina Reader -> Tavily Extract, with cache. Returns one slice of the text."""
        if self.web is None:
            return ToolResult(False, "ERROR: web tools are not available in this session.")
        try:
            out = self.web.fetch(url, tags=self.tags, stop_event=self.stop_event)
        except WebStopped:
            return ToolResult(False, "STOPPED: the user pressed Stop while the page was loading.")
        self._web_note(out, "fetch")
        if not out.ok:
            return ToolResult(False, f"ERROR: could not read the page ({out.error}). Try another URL.")
        return ToolResult(True, format_page(out, start))


def observation(res: RunResult, timeout: float) -> str:
    """The text the model sees after a run: what happened, then the end of the output."""
    if res.timed_out:
        head = f"TIMEOUT: killed after {timeout:g} s (infinite loop, or waiting for input?)."
    elif res.stopped:
        head = "STOPPED: the user pressed Stop; the process was killed."
    elif res.exit_code == 0:
        head = "EXIT 0 (success)."
    else:
        head = f"EXIT {res.exit_code} (error). Read the traceback below and fix the cause."
    body = clip(res.output.strip(), OBS_LIMIT) if res.output.strip() else "(no output)"
    return f"{head}\n{body}"