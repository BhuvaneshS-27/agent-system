import json
import threading
import time

from core.bus import EventBus
from core.orchestrator import Orchestrator


def test_bus_writes_jsonl_and_cursors_are_independent(tmp_path):
    bus = EventBus(tmp_path / "events.jsonl")
    bus.emit("a", x=1)
    bus.emit("b", x=2)
    first, c1 = bus.read_since(0)           # tab 1 reads everything
    second, c2 = bus.read_since(0)          # tab 2 also reads everything
    assert [e["kind"] for e in first] == [e["kind"] for e in second] == ["a", "b"]
    assert bus.read_since(c1)[0] == []      # nothing new after the cursor
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["kind"] for l in lines] == ["a", "b"]


def test_bus_is_thread_safe(tmp_path):
    bus = EventBus(tmp_path / "events.jsonl")
    threads = [threading.Thread(target=lambda: [bus.emit("n") for _ in range(100)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert bus.last_seq == 800
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 800


def test_failing_job_is_isolated_and_always_emits_final_event(tmp_path):
    bus = EventBus(tmp_path / "events.jsonl")
    orch = Orchestrator(bus)

    def boom(job, bus):
        raise RuntimeError("boom")

    job = orch.start("coding", boom)
    for _ in range(50):                     # wait up to 2.5 s for the final event
        kinds = [e["kind"] for e in bus.read_since(0)[0]]
        if kinds and kinds[-1] == "job_failed":
            break
        time.sleep(0.05)
    assert job.status == "failed" and "boom" in job.error
    assert "error" in kinds and kinds[-1] == "job_failed"
    