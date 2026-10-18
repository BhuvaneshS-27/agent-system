"""Orchestrator: runs each agent as a background job (thread) with failure isolation.

Rules (these are what you explain to the professor):
- every job body is wrapped in try/except, so one failing job never touches the others;
- every job always emits a final event: job_done, job_failed or job_stopped;
- every job has a stop_event that the job body must check regularly.
"""
import threading
import time
import traceback

from core.jobs import JobRecord

AGENTS = ("coding", "image", "video")


class Orchestrator:
    def __init__(self, bus):
        self.bus = bus
        self.jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def start(self, agent: str, target, **kwargs) -> JobRecord:
        """Start target(job, bus, **kwargs) in a background thread."""
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent!r}")
        if self.is_running(agent):
            raise RuntimeError(f"{agent} agent is already running")
        job = JobRecord(agent=agent)
        with self._lock:
            self.jobs[job.id] = job
        self.bus.emit("job_queued", job=job.id, agent=agent)
        threading.Thread(target=self._run, args=(job, target, kwargs),
                         name=f"job-{agent}-{job.id}", daemon=True).start()
        return job

    def _run(self, job: JobRecord, target, kwargs: dict) -> None:
        job.status = "running"
        job.started = time.time()
        self.bus.emit("job_started", job=job.id, agent=job.agent)
        try:
            target(job, self.bus, **kwargs)
            job.status = "stopped" if job.stop_event.is_set() else "done"
        except Exception as e:                                    # failure isolation
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            self.bus.emit("error", job=job.id, agent=job.agent, error=job.error,
                          traceback=traceback.format_exc())
        finally:                                                  # always a final event
            job.ended = time.time()
            self.bus.emit(f"job_{job.status}", job=job.id, agent=job.agent,
                          seconds=round(job.ended - job.started, 1), error=job.error)

    def stop(self, agent: str) -> bool:
        job = self.latest(agent)
        if job and job.active:
            job.stop_event.set()
            self.bus.emit("stop_requested", job=job.id, agent=agent)
            return True
        return False

    def latest(self, agent: str):
        with self._lock:
            mine = [j for j in self.jobs.values() if j.agent == agent]
        return max(mine, key=lambda j: j.started or float("inf"), default=None)

    def is_running(self, agent: str) -> bool:
        job = self.latest(agent)
        return bool(job and job.active)