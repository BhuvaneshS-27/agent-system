"""Tests for the sandbox and the file tools. They run real child processes (a few seconds)."""
import os
import sys
import threading
import time

import psutil
import pytest

from agents.coding.sandbox import MAX_CAPTURE, safe_path, scrubbed_env
from agents.coding.tools import OBS_LIMIT, Toolbox
from core.bus import EventBus


@pytest.fixture()
def tb(tmp_path):
    bus = EventBus(tmp_path / "events.jsonl")
    box = Toolbox(tmp_path / "ws", bus, tags={"job": "t1"}, timeout=3)
    box.bus_ = bus
    return box


def kinds(box):
    return [e["kind"] for e in box.bus_.read_since(0)[0]]


# ------------------------------------------------------------------ safe_path
@pytest.mark.parametrize("bad", ["../x.txt", "a/../../x.txt", "/etc/passwd", "C:\\Windows\\win.ini",
                                 "C:win.ini", "", "   ", "a\x00b", "file.txt:stream"])
def test_safe_path_refuses_escapes(tmp_path, bad):
    with pytest.raises(PermissionError):
        safe_path(tmp_path, bad)


def test_safe_path_allows_inside_paths(tmp_path):
    root = tmp_path.resolve()
    assert safe_path(tmp_path, "a.py") == root / "a.py"
    assert safe_path(tmp_path, "sub/b.py") == root / "sub" / "b.py"
    assert safe_path(tmp_path, "sub/../c.py") == root / "c.py"      # '..' that stays inside is fine
    assert safe_path(tmp_path, ".") == root


def test_safe_path_refuses_symlink_that_points_outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    ws = tmp_path / "ws"
    ws.mkdir()
    try:
        (ws / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this system cannot create symlinks")
    with pytest.raises(PermissionError):
        safe_path(ws, "link/secret.txt")


def test_tools_refuse_escape_and_log_a_blocked_event(tb, tmp_path):
    r = tb.write_file("../evil.txt", "x")
    assert not r.ok and "not allowed" in r.text
    assert not (tmp_path / "evil.txt").exists()
    assert not tb.read_file("../../etc/passwd").ok
    assert not tb.list_files("..").ok
    assert not tb.run_python(path="../x.py").ok
    assert kinds(tb).count("blocked") == 4


# ------------------------------------------------------------------ file tools
def test_write_read_list_roundtrip(tb):
    assert tb.write_file("notes/a.txt", "hello").ok                  # creates the sub folder
    assert tb.read_file("notes/a.txt").text == "hello"
    listing = tb.list_files(".").text
    assert "notes/a.txt  (5 bytes)" in listing and "notes/" in listing
    assert not tb.read_file("missing.txt").ok
    assert not tb.read_file("notes").ok                              # a folder, not a file
    assert tb.list_files("empty_never_made").ok is False


def test_syntax_check_on_py_files(tb):
    bad = tb.write_file("bad.py", "def f(:\n    pass\n")
    assert not bad.ok and "syntax error" in bad.text and "line 1" in bad.text
    assert (tb.root / "bad.py").exists()                            # saved anyway, so it can be fixed
    ev = [e for e in tb.bus_.read_since(0)[0] if e["kind"] == "file_written"][-1]
    assert ev["syntax_ok"] is False and ev["path"] == "bad.py" and ev["job"] == "t1"

    indent = tb.write_file("ind.py", "def f():\nreturn 1\n")
    assert not indent.ok and "IndentationError" in indent.text

    good = tb.write_file("good.py", "def f():\n    return 1\n")
    assert good.ok and "syntax OK" in good.text
    assert tb.write_file("notes.txt", "def f(:").ok                   # only .py files are checked


def test_write_file_limits(tb):
    assert not tb.write_file("big.txt", "x" * 300_000).ok
    assert not tb.write_file(".", "x").ok                            # the workspace folder itself


# ------------------------------------------------------------------ run_python
def test_stdout_and_exit_code(tb):
    r = tb.run_python(code="print('hi', 2 + 3)")
    assert r.ok and "EXIT 0" in r.text and "hi 5" in r.text
    term = [e for e in tb.bus_.read_since(0)[0] if e["kind"] == "terminal"][0]
    assert term["exit_code"] == 0 and "hi 5" in term["output"]


def test_traceback_is_returned_and_hides_the_machine_path(tb):
    r = tb.run_python(code="x = 1\nprint(1 / 0)\n")
    assert not r.ok and "EXIT 1" in r.text
    assert "ZeroDivisionError" in r.text and "line 2" in r.text
    assert str(tb.root) not in r.text and "<workspace>" in r.text


def test_script_can_import_other_files_in_the_workspace(tb):
    tb.write_file("solution.py", "def add(a, b):\n    return a + b\n")
    tb.write_file("tests.py", "from solution import add\nassert add(2, 3) == 5\nprint('tests ok')\n")
    r = tb.run_python(path="tests.py")                               # why we do not use python -I
    assert r.ok and "tests ok" in r.text


def test_unicode_output(tb):
    r = tb.run_python(code="print('caf\\u00e9 \\u2713')")
    assert r.ok and "café ✓" in r.text


def test_input_does_not_hang(tb):
    t0 = time.time()
    r = tb.run_python(code="input('name? ')")
    assert not r.ok and "EOFError" in r.text and time.time() - t0 < 3


def test_infinite_loop_is_killed_after_the_timeout(tb):
    tb.timeout = 1
    t0 = time.time()
    r = tb.run_python(code="print('start', flush=True)\nwhile True:\n    pass\n")
    assert time.time() - t0 < 4
    assert not r.ok and r.text.startswith("TIMEOUT") and "start" in r.text   # output before the kill is kept
    term = [e for e in tb.bus_.read_since(0)[0] if e["kind"] == "terminal"][0]
    assert term["timed_out"] is True and term["exit_code"] is None


def test_the_whole_process_tree_is_killed(tb):
    tb.timeout = 1
    code = ("import subprocess, sys, time\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "open('child.pid', 'w').write(str(c.pid))\n"
            "time.sleep(60)\n")
    assert tb.run_python(code=code).text.startswith("TIMEOUT")
    pid = int((tb.root / "child.pid").read_text())
    deadline = time.time() + 5
    while time.time() < deadline:                                    # allow a moment for the OS to clean up
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild process is still running")


def test_stop_event_kills_the_run_quickly(tb):
    ev = threading.Event()
    tb.stop_event = ev
    tb.timeout = 30
    threading.Timer(0.7, ev.set).start()
    t0 = time.time()
    r = tb.run_python(code="import time\ntime.sleep(60)")
    assert time.time() - t0 < 2.5
    assert not r.ok and r.text.startswith("STOPPED")


def test_huge_output_is_bounded(tb):
    code = "import sys\nfor _ in range(3000):\n    sys.stdout.write('x' * 1000 + '\\n')\n"
    r = tb.run_python(code=code)
    assert r.ok and len(r.text) < OBS_LIMIT + 200 and "omitted" in r.text
    term = [e for e in tb.bus_.read_since(0)[0] if e["kind"] == "terminal"][0]
    assert len(term["output"]) <= MAX_CAPTURE and term["total_chars"] > 2_000_000


def test_keys_are_not_visible_to_the_child_process(tb, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "key-value-123")
    monkeypatch.setenv("FAKE_WORKER_TOKEN", "token-value-456")
    monkeypatch.setenv("SEC_USER_AGENT", "Team me@example.com")      # no secret suffix, but it is in .env
    monkeypatch.setenv("HARMLESS_NAME", "visible-ok")
    tb.drop_env = ("SEC_USER_AGENT",)
    code = ("import os\n"
            "for k, v in sorted(os.environ.items()):\n"
            "    if k.startswith(('FAKE', 'SEC_', 'HARMLESS')):\n"
            "        print(k + '=' + v)\n"
            "print('has_path', any(k.upper() == 'PATH' for k in os.environ))\n")
    r = tb.run_python(code=code)
    assert r.ok
    for secret in ("key-value-123", "token-value-456", "me@example.com", "FAKE_API_KEY", "SEC_USER_AGENT"):
        assert secret not in r.text
    assert "HARMLESS_NAME=visible-ok" in r.text
    assert "has_path True" in r.text                                 # the child still has a working environment


def test_scrubbed_env_rule():
    env = scrubbed_env(["Custom_Name"])
    assert not any(k.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) for k in env)
    assert "CUSTOM_NAME" not in {k.upper() for k in env}
    assert ("PATH" in env) or ("Path" in env)


def test_run_python_argument_errors(tb):
    assert not tb.run_python().ok
    assert not tb.run_python(path="nothing.py").ok
    assert sys.executable                                            # (the child uses this interpreter)