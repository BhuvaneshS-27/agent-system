"""Fake jobs for Step 1 only. They prove the bus, the threads and the dashboard work.
We delete this file in Step 4, when the real coding agent replaces it."""


def fake_job(job, bus, interval: float = 1.0, max_ticks: int = 60, fail_after: int | None = None):
    for n in range(1, max_ticks + 1):
        bus.emit("tick", job=job.id, agent=job.agent, n=n, text=f"fake event number {n}")
        if fail_after and n >= fail_after:
            raise RuntimeError("deliberate failure to prove failure isolation")
        if job.stop_event.wait(interval):      # sleeps, but wakes at once when Stop is pressed
            break