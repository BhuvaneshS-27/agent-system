"""Tests for the agent loop. A fake llama-server plays the model from a script, so no real
model is needed. Action requests (they carry response_format) take the next entry of
`actions`; plan and summary requests take the next entry of `texts`."""
import dataclasses
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agents.coding.actions import DEFAULT_ENABLED, TOOLS
from agents.coding.approvals import Approvals
from agents.coding.llm import LLMClient
from agents.coding.loop import agent_loop
from agents.coding.tasks import TASKS
from core.bus import EventBus
from core.jobs import JobFailed, JobRecord
from core.orchestrator import Orchestrator
from core.settings import load_settings


def act(tool, thought="ok", **args):
    return json.dumps({"thought": thought, "tool": tool, "args": args})


class FakeLlama(BaseHTTPRequestHandler):
    actions: list = []
    texts: list = []
    seen: list = []
    after_action = None         # called with the number of actions served so far

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = FakeLlama
        cls.seen.append(body)
        is_action = "response_format" in body
        queue = cls.actions if is_action else cls.texts
        if not queue:
            if is_action:
                return self._send(500, {"error": {"message": "script exhausted"}})
            entry = "1. do the task"
        else:
            entry = queue.pop(0)
        if isinstance(entry, tuple):                                # ("http", code, message)
            return self._send(entry[1], {"error": {"message": entry[2]}})
        if is_action and cls.after_action:
            cls.after_action(sum(1 for b in cls.seen if "response_format" in b))
        if body.get("stream"):                                      # the loop streams: answer like llama-server
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            half = len(entry) // 2
            chunks = [{"choices": [{"delta": {"content": entry[:half]}}]},
                      {"choices": [{"delta": {"content": entry[half:]}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                       "timings": {"predicted_per_second": 9.0}}]
            self.wfile.write(b": ping\n\n")
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send(200, {"choices": [{"message": {"content": entry}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                         "timings": {"predicted_per_second": 9.0}})


@pytest.fixture()
def env(tmp_path):
    FakeLlama.actions, FakeLlama.texts, FakeLlama.seen, FakeLlama.after_action = [], [], [], None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    bus = EventBus(tmp_path / "events.jsonl")
    s = dataclasses.replace(load_settings(), llm_url=f"http://127.0.0.1:{srv.server_address[1]}",
                            max_tokens=256, max_steps=12, workspace_dir=tmp_path / "ws")

    class Env:
        pass
    e = Env()
    e.bus, e.settings, e.job = bus, s, JobRecord(agent="coding")
    e.client = LLMClient(s, bus, retry_wait=0)

    def run(task="do it", **kw):
        return agent_loop(e.job, bus, s, task, client=e.client, **kw)
    e.run = run
    e.events = lambda: bus.read_since(0)[0]
    e.kinds = lambda: [x["kind"] for x in bus.read_since(0)[0]]
    e.folder = lambda: s.workspace_dir / f"run_{e.job.id}"
    yield e
    srv.shutdown()


BAD_PRIME = "def is_prime(n):\n    return all(n % i for i in range(2, n))\n"
GOOD_PRIME = "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, n))\n"
TESTS = ("from solution import is_prime\n"
         "assert is_prime(0) is False and is_prime(1) is False and is_prime(97) is True\n"
         "print('ALL TESTS PASSED')\n")


def test_write_run_fail_fix_run_finish(env):
    FakeLlama.texts = ["1. write solution\n2. write tests\n3. run"]
    FakeLlama.actions = [
        act("write_file", path="solution.py", content=BAD_PRIME),
        act("write_file", path="tests.py", content=TESTS),
        act("run_python", path="tests.py"),                           # fails: is_prime(0) is True
        act("write_file", path="solution.py", content=GOOD_PRIME),    # the fix
        act("run_python", path="tests.py"),                           # passes
        act("finish", answer="is_prime written; tests.py printed ALL TESTS PASSED"),
    ]
    env.run()
    story = [(e["kind"], e.get("tool"), e.get("ok")) for e in env.events()
             if e["kind"] in ("action", "observation")]
    assert story[4:10] == [("action", "run_python", None), ("observation", "run_python", False),
                           ("action", "write_file", None), ("observation", "write_file", True),
                           ("action", "run_python", None), ("observation", "run_python", True)]
    kinds = env.kinds()
    assert kinds.count("plan") == 1 and "reflect" in kinds and kinds[-1] == "done"
    done = env.events()[-1]
    assert done["verified"] is True and done["steps"] == 6
    assert (env.folder() / "solution.py").read_text() == GOOD_PRIME
    # the traceback was sent back to the model with the 'fix the cause' line
    obs_msg = [m["content"] for m in FakeLlama.seen[4]["messages"] if m["role"] == "user"][-1]
    assert "AssertionError" in obs_msg and "Fix the cause" in obs_msg
    # the schema only offers the tools that work
    enum = FakeLlama.seen[1]["response_format"]["json_schema"]["schema"]["properties"]["tool"]["enum"]
    # no approval gate was given, so run_shell is dropped although it is in DEFAULT_ENABLED
    assert enum == [t for t in TOOLS if t in DEFAULT_ENABLED and t != "run_shell"]


def test_through_the_orchestrator_the_job_ends_done(env):
    FakeLlama.actions = [act("finish", answer="nothing to do")]
    orch = Orchestrator(env.bus)
    job = orch.start("coding", agent_loop, settings=env.settings, task="hi", client=env.client)
    for _ in range(100):
        if not job.active:
            break
        time.sleep(0.05)
    assert job.status == "done"


def test_bad_arguments_are_rejected_and_retried(env):
    FakeLlama.actions = [act("run_python"),                               # neither code nor path
                         act("write_file", path="a.txt", content=5),      # content is not text
                         act("finish", answer="done")]
    env.run()
    msgs = [m["content"] for m in FakeLlama.seen[-1]["messages"] if m["role"] == "user"]
    assert "give exactly one of 'code' or 'path'" in msgs[-2]
    assert "content" in msgs[-1] and "valid string" in msgs[-1]
    assert [e for e in env.events() if e["kind"] == "action"][0]["attempts"] == 3
    assert sum(1 for e in env.events() if e["kind"] == "retry") == 2


def test_a_tool_that_is_not_available_yet_is_rejected(env):
    FakeLlama.actions = [act("web_search", query="x"), act("finish", answer="done")]
    env.run()
    last_user = [m["content"] for m in FakeLlama.seen[-1]["messages"] if m["role"] == "user"][-1]
    assert "not available" in last_user and "web_search" in last_user


def test_finish_is_refused_while_the_last_run_failed_or_code_changed(env):
    FakeLlama.actions = [act("write_file", path="a.py", content="print(1)\n"),
                         act("finish", answer="done"),                    # refused: never run
                         act("run_python", path="a.py"),
                         act("finish", answer="done, a.py printed 1")]
    env.run()
    assert env.kinds().count("finish_refused") == 1
    done = env.events()[-1]
    assert done["kind"] == "done" and done["verified"] is True


def test_finish_is_accepted_unverified_after_two_refusals(env):
    FakeLlama.actions = [act("write_file", path="a.py", content="print(1)\n")] + \
        [act("finish", answer=f"x{i}") for i in range(3)]
    env.run()
    assert env.kinds().count("finish_refused") == 2
    done = env.events()[-1]
    assert done["kind"] == "done" and done["verified"] is False


def test_repeating_the_same_action_replans_once_then_fails(env):
    same = act("write_file", path="a.txt", content="x")
    FakeLlama.actions = [same] * 6
    with pytest.raises(JobFailed, match="stuck"):
        env.run()
    k = env.kinds()
    assert k.count("repeat_guard") == 2
    assert [e["reason"] for e in env.events() if e["kind"] == "plan"] == ["initial", "replan"]
    assert k.count("observation") == 4                  # steps 3 and 6 were never executed


def test_three_failures_in_a_row_trigger_a_new_plan(env):
    FakeLlama.actions = [act("read_file", path="nope1.txt"), act("read_file", path="nope2.txt"),
                         act("read_file", path="nope3.txt"), act("finish", answer="gave up")]
    env.run()
    assert [e["reason"] for e in env.events() if e["kind"] == "plan"] == ["initial", "replan"]


def test_step_budget_ends_in_a_failed_job(env):
    env.settings = dataclasses.replace(env.settings, max_steps=3)
    FakeLlama.actions = [act("write_file", path=f"n{i}.txt", content="x") for i in range(3)]
    with pytest.raises(JobFailed, match="step budget exhausted"):
        agent_loop(env.job, env.bus, env.settings, "t", client=env.client)
    assert "budget_exhausted" in env.kinds()


def test_stop_between_steps_and_after_a_model_call(env):
    FakeLlama.actions = [act("write_file", path="a.txt", content="x"),
                         act("write_file", path="b.txt", content="x")]
    FakeLlama.after_action = lambda n: env.job.stop_event.set() if n == 2 else None
    env.run()
    assert env.kinds()[-1] == "stopped"
    assert (env.folder() / "a.txt").exists() and not (env.folder() / "b.txt").exists()


def test_history_is_summarised_every_five_steps(env):
    FakeLlama.texts = ["1. plan", "SUMMARY: five notes written"]
    FakeLlama.actions = [act("write_file", path=f"n{i}.txt", content="x") for i in range(5)] + \
                        [act("finish", answer="done")]
    env.run()
    assert env.kinds().count("compact") == 1
    msgs = FakeLlama.seen[-1]["messages"]                  # the request of step 6
    assert len(msgs) == 4 and "SUMMARY: five notes written" in msgs[1]["content"]
    assert "TASK:" in msgs[1]["content"] and "YOUR PLAN" in msgs[1]["content"]


def test_summary_falls_back_to_the_step_log_if_the_model_fails(env):
    FakeLlama.texts = ["1. plan", ("http", 500, "boom")]
    FakeLlama.actions = [act("write_file", path=f"n{i}.txt", content="x") for i in range(5)] + \
                        [act("finish", answer="done")]
    env.run()
    msgs = FakeLlama.seen[-1]["messages"]
    assert "step 5: write_file n4.txt -> ok" in msgs[1]["content"]


def test_planning_failure_does_not_stop_the_loop(env):
    FakeLlama.texts = [("http", 500, "boom")]
    FakeLlama.actions = [act("finish", answer="ok")]
    env.run()
    plan = [e for e in env.events() if e["kind"] == "plan"][0]
    assert "planning failed" in plan["text"] and env.kinds()[-1] == "done"


def test_seeded_task_and_the_protected_file_check(env):
    preset = TASKS["4. Fix a bug (seeded files)"]
    FakeLlama.actions = [act("write_file", path="test_buggy.py", content="print('cheat')\n"),
                         act("run_python", path="test_buggy.py"),
                         act("finish", answer="made them pass")]
    env.run(preset.task, files=preset.files, protected=preset.protected)
    first_user = [m["content"] for m in FakeLlama.seen[0]["messages"]][-1]
    assert "buggy.py" in first_user and "test_buggy.py" in first_user      # the plan call sees the file list
    done = env.events()[-1]
    assert done["integrity_ok"] is False and done["changed_protected"] == ["test_buggy.py"]
    assert "seeded" in env.kinds()


def test_seeded_bug_really_fails_and_the_fix_really_passes(env):
    """The seeded files must fail at first, and fail again on the empty-list case after the
    obvious first fix, so the demo shows two fixes."""
    preset = TASKS["4. Fix a bug (seeded files)"]
    FakeLlama.actions = [
        act("run_python", path="test_buggy.py"),
        act("write_file", path="buggy.py", content=(
            "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n"
            "def is_even(n):\n    return n % 2 == 0\n")),
        act("run_python", path="test_buggy.py"),
        act("write_file", path="buggy.py", content=(
            "def average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0\n\n"
            "def is_even(n):\n    return n % 2 == 0\n")),
        act("run_python", path="test_buggy.py"),
        act("finish", answer="fixed average and is_even; test_buggy.py prints ALL TESTS PASSED"),
    ]
    env.run(preset.task, files=preset.files, protected=preset.protected)
    runs = [e["ok"] for e in env.events() if e["kind"] == "observation" and e["tool"] == "run_python"]
    assert runs == [False, False, True]
    done = env.events()[-1]
    assert done["verified"] is True and done["integrity_ok"] is True


# ---------------------------------------------------------------- Step 5: shell and approvals
PYQ = f'"{sys.executable}"'


def answer_requests(env, approvals, ok):
    """Plays the user: answers every approval request with Approve (ok=True) or Deny."""
    def go():
        answered = set()
        for _ in range(400):
            for e in env.events():
                if e["kind"] == "approval_request" and e["id"] not in answered:
                    answered.add(e["id"])
                    time.sleep(0.1)
                    approvals.answer(e["id"], ok)
            time.sleep(0.05)
    threading.Thread(target=go, daemon=True).start()


def test_shell_command_waits_for_approval_then_runs(env):
    approvals = Approvals(env.bus, timeout=10)
    answer_requests(env, approvals, True)
    FakeLlama.actions = [act("run_shell", command=f'{PYQ} -c "print(5 * 5)"'),
                         act("finish", answer="the shell printed 25")]
    env.run(approvals=approvals)
    kinds = env.kinds()
    order = [k for k in kinds if k in ("action", "approval_request", "approval_result", "terminal", "observation", "done")]
    assert order == ["action", "approval_request", "approval_result", "terminal", "observation", "action", "done"]
    obs = next(e for e in env.events() if e["kind"] == "observation")
    assert obs["ok"] and "25" in obs["text"]
    enum = FakeLlama.seen[1]["response_format"]["json_schema"]["schema"]["properties"]["tool"]["enum"]
    assert "run_shell" in enum
    assert "run_shell needs the user" in FakeLlama.seen[1]["messages"][0]["content"]
    assert env.job.status == "running"


def test_denied_command_is_reported_to_the_model(env):
    approvals = Approvals(env.bus, timeout=10)
    answer_requests(env, approvals, False)
    FakeLlama.actions = [act("run_shell", command="pip install rich"),
                         act("finish", answer="the user denied the install, so I could not use rich")]
    env.run(approvals=approvals)
    obs = next(e for e in env.events() if e["kind"] == "observation")
    assert not obs["ok"] and obs["text"].startswith("DENIED")
    assert "terminal" not in env.kinds()                              # nothing ran
    last_user = [m["content"] for m in FakeLlama.seen[-1]["messages"] if m["role"] == "user"][-1]
    assert "DENIED" in last_user and "Fix the cause" in last_user
    done = next(e for e in env.events() if e["kind"] == "done")
    assert done["verified"] is True                                   # a denied shell is not a failed run


def test_blocked_command_never_asks_the_user(env):
    approvals = Approvals(env.bus, timeout=10)
    FakeLlama.actions = [act("run_shell", command="rm -rf ."), act("finish", answer="the command was blocked")]
    env.run(approvals=approvals)
    kinds = env.kinds()
    assert "blocked" in kinds and "approval_request" not in kinds
    obs = next(e for e in env.events() if e["kind"] == "observation")
    assert obs["text"].startswith("BLOCKED")


def test_stop_during_a_model_call_ends_the_job_quietly(env):
    def stop_then_stall(n):
        if n == 1:
            env.job.stop_event.set()
            time.sleep(2)                                             # the 'model' is still busy
    FakeLlama.after_action = stop_then_stall
    FakeLlama.actions = [act("list_files", path=".")]
    t0 = time.time()
    env.run()
    assert time.time() - t0 < 2                                       # did not wait for the reply
    ev = env.events()[-1]
    assert ev["kind"] == "stopped" and ev["during"] == "model call"
    assert "action" not in env.kinds()


def test_stop_while_waiting_for_approval(env):
    approvals = Approvals(env.bus, timeout=30)
    threading.Timer(0.8, env.job.stop_event.set).start()
    FakeLlama.actions = [act("run_shell", command="echo hi"), act("finish", answer="x")]
    env.run(approvals=approvals)
    results = [e for e in env.events() if e["kind"] == "approval_result"]
    assert results and results[0]["result"] == "stopped"
    assert env.kinds()[-1] == "stopped" and "done" not in env.kinds()