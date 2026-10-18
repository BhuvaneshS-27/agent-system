"""Tests for the LLM client against a fake llama-server (no model needed)."""
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agents.coding.llm import LLMClient, LLMError, LLMUnavailable
from core.bus import EventBus
from core.settings import load_settings

GOOD = json.dumps({"thought": "List the files first.", "tool": "list_files", "args": {"path": "."}})


class FakeLlama(BaseHTTPRequestHandler):
    script: list = []       # one entry per POST: ("ok", content) | ("length", content) | ("http", code, message)
    requests_seen: list = []
    health_code = 200

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(FakeLlama.health_code, {"status": "ok"} if FakeLlama.health_code == 200
                       else {"error": {"code": 503, "message": "Loading model"}})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.requests_seen.append(body)
        step = FakeLlama.script.pop(0)
        if step[0] == "http":
            return self._send(step[1], {"error": {"message": step[2]}})
        finish = "length" if step[0] == "length" else "stop"
        self._send(200, {"choices": [{"message": {"content": step[1]}, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                         "timings": {"predicted_per_second": 7.5}})


@pytest.fixture()
def server():
    FakeLlama.script, FakeLlama.requests_seen, FakeLlama.health_code = [], [], 200
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_client(url, tmp_path):
    s = dataclasses.replace(load_settings(), llm_url=url, max_tokens=256)
    bus = EventBus(tmp_path / "events.jsonl")
    return LLMClient(s, bus, retry_wait=0), bus


def kinds(bus):
    return [e["kind"] for e in bus.read_since(0)[0]]


def test_health_states(server, tmp_path):
    client, _ = make_client(server, tmp_path)
    assert client.health()[0] == "online"
    FakeLlama.health_code = 503
    assert client.health()[0] == "loading"
    offline, _ = make_client("http://127.0.0.1:9", tmp_path)       # nothing listens on port 9
    assert offline.health()[0] == "offline"


def test_valid_action_and_request_format(server, tmp_path):
    FakeLlama.script = [("ok", GOOD)]
    client, bus = make_client(server, tmp_path)
    res = client.action([{"role": "user", "content": "hi"}], tags={"job": "j1", "step": 1})
    assert res.action.tool == "list_files" and res.attempts == 1
    sent = FakeLlama.requests_seen[0]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}        # thinking off
    assert sent["max_tokens"] == 256
    rf = sent["response_format"]
    assert rf["type"] == "json_schema"
    assert "finish" in rf["json_schema"]["schema"]["properties"]["tool"]["enum"]
    call = [e for e in bus.read_since(0)[0] if e["kind"] == "llm_call"][0]
    assert call["completion_tokens"] == 30 and call["tokens_per_second"] == 7.5 and call["job"] == "j1"


def test_invalid_json_is_retried_and_the_error_is_told_to_the_model(server, tmp_path):
    FakeLlama.script = [("ok", "this is not json"), ("ok", GOOD)]
    client, bus = make_client(server, tmp_path)
    res = client.action([{"role": "user", "content": "hi"}])
    assert res.attempts == 2
    assert kinds(bus).count("retry") == 1
    second_request = FakeLlama.requests_seen[1]["messages"]
    assert "Invalid action" in second_request[-1]["content"]


def test_wrong_tool_name_is_rejected(server, tmp_path):
    bad = json.dumps({"thought": "x", "tool": "delete_everything", "args": {}})
    FakeLlama.script = [("ok", bad), ("ok", GOOD)]
    client, _ = make_client(server, tmp_path)
    assert client.action([{"role": "user", "content": "hi"}]).attempts == 2


def test_cut_off_answer_counts_as_invalid(server, tmp_path):
    FakeLlama.script = [("length", GOOD[:20]), ("ok", GOOD)]
    client, bus = make_client(server, tmp_path)
    assert client.action([{"role": "user", "content": "hi"}]).attempts == 2
    assert "cut off" in [e for e in bus.read_since(0)[0] if e["kind"] == "retry"][0]["reason"]


def test_gives_up_after_three_invalid_answers(server, tmp_path):
    FakeLlama.script = [("ok", "nope")] * 3
    client, _ = make_client(server, tmp_path)
    with pytest.raises(LLMError, match="no valid action after 3 tries"):
        client.action([{"role": "user", "content": "hi"}])
    assert len(FakeLlama.requests_seen) == 3


def test_server_down_is_retried_three_times_then_reported(tmp_path):
    client, bus = make_client("http://127.0.0.1:9", tmp_path)
    with pytest.raises(LLMUnavailable, match="Is it running"):
        client.action([{"role": "user", "content": "hi"}])
    assert kinds(bus).count("retry") == 2           # 3 tries = 2 waits


def test_http_error_is_not_retried_and_shows_the_server_message(server, tmp_path):
    FakeLlama.script = [("http", 400, "request (9000 tokens) exceeds the available context size (8192 tokens)")]
    client, _ = make_client(server, tmp_path)
    with pytest.raises(LLMError, match="exceeds the available context size"):
        client.action([{"role": "user", "content": "hi"}])
    assert len(FakeLlama.requests_seen) == 1