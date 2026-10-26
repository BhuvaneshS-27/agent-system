"""Tests for the streaming model call and Stop, against a fake llama-server that speaks SSE
the way llama.cpp does (usage and timings in the last chunk, ': ping' comments, [DONE])."""
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agents.coding.llm import LLMClient, LLMError, LLMStopped, LLMUnavailable
from core.bus import EventBus
from core.settings import load_settings

GOOD = json.dumps({"thought": "List the files first.", "tool": "list_files", "args": {"path": "."}})


class Fake(BaseHTTPRequestHandler):
    mode = "normal"
    seen: list = []
    server_saw_disconnect = threading.Event()

    def log_message(self, *a):
        pass

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        mode = Fake.mode
        if mode == "503":
            data = json.dumps({"error": {"code": 503, "message": "Loading model"}}).encode()
            self.send_response(503)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if mode == "400":
            data = json.dumps({"error": {"message": "bad schema"}}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if mode == "empty":
            return
        try:
            if mode == "silent":                                     # still reading the prompt
                for _ in range(100):
                    time.sleep(0.1)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            self.wfile.write(b": ping\n\n")
            third = len(GOOD) // 3
            self._sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
            self._sse({"choices": [{"delta": {"content": GOOD[:third]}}]})
            if mode == "slow":                                       # then generates "forever"
                for _ in range(200):
                    time.sleep(0.1)
                    self._sse({"choices": [{"delta": {"content": " "}}]})
                return
            self._sse({"choices": [{"delta": {"reasoning_content": "hmm"}}]})
            self._sse({"choices": [{"delta": {"content": GOOD[third:]}}]})
            if mode == "separate_usage":
                self._sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                self._sse({"choices": [], "usage": {"prompt_tokens": 77, "completion_tokens": 11},
                           "timings": {"predicted_per_second": 6.5}})
            elif mode == "cut_off":
                self._sse({"choices": [{"delta": {}, "finish_reason": "length"}],
                           "usage": {"prompt_tokens": 50, "completion_tokens": 256}})
            elif mode == "dropped":
                return                                               # connection closes mid-answer, no finish
            else:
                self._sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                           "timings": {"predicted_per_second": 7.5, "predicted_n": 30}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            Fake.server_saw_disconnect.set()


@pytest.fixture()
def rig(tmp_path):
    Fake.mode, Fake.seen = "normal", []
    Fake.server_saw_disconnect.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = dataclasses.replace(load_settings(), llm_url=f"http://127.0.0.1:{srv.server_address[1]}", max_tokens=256)
    bus = EventBus(tmp_path / "events.jsonl")
    stop = threading.Event()

    class R:
        pass
    r = R()
    r.client = LLMClient(s, bus, retry_wait=0, stop_event=stop)
    r.stop, r.bus = stop, bus
    yield r
    srv.shutdown()


MSG = [{"role": "user", "content": "go"}]


def test_streamed_answer_is_assembled_with_usage_from_the_last_chunk(rig):
    res = rig.client.action(MSG)
    assert res.action.tool == "list_files" and res.attempts == 1
    assert res.raw == GOOD
    assert (res.reply.prompt_tokens, res.reply.completion_tokens) == (120, 30)
    assert res.reply.tokens_per_second == 7.5 and res.reply.reasoning_chars == 3
    body = Fake.seen[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" in body and body["response_format"]["type"] == "json_schema"


def test_usage_in_a_separate_chunk_with_empty_choices(rig):
    Fake.mode = "separate_usage"
    reply = rig.client.chat(MSG)
    assert reply.content == GOOD and (reply.prompt_tokens, reply.completion_tokens) == (77, 11)
    assert reply.tokens_per_second == 6.5


def test_cut_off_answer_is_reported_to_the_model_and_retried(rig):
    Fake.mode = "cut_off"
    with pytest.raises(LLMError, match="no valid action"):
        rig.client.action(MSG)
    retries = [e for e in rig.bus.read_since(0)[0] if e["kind"] == "retry"]
    assert len(retries) == 3 and "cut off" in retries[0]["reason"]


def test_loading_server_is_retried_then_unavailable(rig):
    Fake.mode = "503"
    with pytest.raises(LLMUnavailable):
        rig.client.chat(MSG)
    assert len(Fake.seen) == 3                                       # 3 tries
    assert [e["what"] for e in rig.bus.read_since(0)[0] if e["kind"] == "retry"] == ["llm_connect"] * 2


def test_http_error_message_is_kept(rig):
    Fake.mode = "400"
    with pytest.raises(LLMError, match="bad schema"):
        rig.client.chat(MSG)


def test_stream_that_drops_after_the_full_answer_still_works(rig):
    Fake.mode = "dropped"            # no finish_reason and no usage, but the whole JSON arrived
    reply = rig.client.chat(MSG)
    assert reply.content == GOOD and reply.finish_reason == "stop"


def test_stream_that_closes_with_nothing_is_unavailable(rig):
    Fake.mode = "empty"
    with pytest.raises(LLMUnavailable, match="without an answer"):
        rig.client.chat(MSG)


def test_connection_refused_is_unavailable(rig, tmp_path):
    s = dataclasses.replace(load_settings(), llm_url="http://127.0.0.1:1", max_tokens=256)
    c = LLMClient(s, EventBus(tmp_path / "e2.jsonl"), retry_wait=0, stop_event=threading.Event())
    with pytest.raises(LLMUnavailable):
        c.chat(MSG)


def test_stop_in_the_middle_of_a_reply(rig):
    Fake.mode = "slow"
    threading.Timer(0.6, rig.stop.set).start()
    t0 = time.time()
    with pytest.raises(LLMStopped):
        rig.client.chat(MSG)
    assert time.time() - t0 < 3
    # our side closed the socket: the fake server notices on its next write
    assert Fake.server_saw_disconnect.wait(5)


def test_stop_while_the_server_is_still_reading_the_prompt(rig):
    Fake.mode = "silent"                                              # nothing but pings for 10 s
    threading.Timer(0.5, rig.stop.set).start()
    t0 = time.time()
    with pytest.raises(LLMStopped):
        rig.client.chat(MSG)
    assert time.time() - t0 < 3


def test_stopped_is_not_an_llm_error():
    assert not issubclass(LLMStopped, LLMError)


def test_without_stop_event_the_plain_path_is_used(tmp_path):
    # the old tests (test_llm.py) cover this path; here only check the switch
    s = dataclasses.replace(load_settings(), llm_url="http://127.0.0.1:1")
    c = LLMClient(s, None, retry_wait=0)
    assert c.stop_event is None