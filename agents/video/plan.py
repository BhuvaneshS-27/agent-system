"""The 8-slide plan. Narration comes from templates filled with real facts, never from the LLM."""
import json

from agents.video.diagrams import SAMPLE_TIMELINE, highlight, timeline_mermaid
from core.settings import ROOT

DIAG = ROOT / "assets" / "diagrams"


def read_diagram(name):
    try:
        return (DIAG / f"{name}.mmd").read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    except OSError:
        return None


def _frames(name, steps):
    code = read_diagram(name)
    return [{"text": t, "code": (highlight(code, n) if code and n else code)} for n, t in steps]


def find_run(runs_dir):
    """Key events of the newest coding run found in runs/*/events.jsonl. Returns (events, facts)."""
    files = sorted(runs_dir.glob("session-*/events.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    for f in files:
        evs = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") in ("action", "observation", "approval_request", "approval_result") and e.get("agent", "coding") == "coding":
                evs.append(e)
        actions = [e for e in evs if e["kind"] == "action"]
        if len(actions) < 3:
            continue
        job = actions[-1].get("job")
        mine = [e for e in evs if e.get("job") == job]
        first_err = next((i for i, e in enumerate(mine) if e["kind"] == "observation" and not e.get("ok", True)), None)
        window = mine[max(0, first_err - 2):] if first_err is not None else mine
        return window, {"steps": sum(e["kind"] == "action" for e in mine),
                        "approvals": sum(e["kind"] == "approval_request" for e in mine),
                        "errors": sum(e["kind"] == "observation" and not e.get("ok", True) for e in mine)}
    return [], {}


def build_slides(f: dict, run_events: list) -> list:
    team, model = f["team"], f["model"]
    if run_events:
        code = timeline_mermaid(run_events)
        n = (f"Here is a real run from our event log. The coding agent took {f['steps']} steps")
        n += f", you approved {f['approvals']} shell commands" if f["approvals"] else ""
        n += (f", and it hit {f['errors']} errors that it fixed itself." if f["errors"] else ".")
        n += " The chart shows the key events in order."
    else:
        code = SAMPLE_TIMELINE
        n = "No coding run was recorded yet, so this is an example of the log format: an action, an observation, an error, and a fix."
    return [
        {"title": "Intro", "kind": "intro", "frames": [{"text": (
            f"Hello, we are {team}. This is our multi-agent system, with a coding agent, an image agent, a video agent "
            "and a startup research agent, all controlled from one dashboard.")}]},
        {"title": "The problem", "kind": "text",
         "bullets": ["One chat assistant is not enough for a real project",
                     "Agents must run side by side and fail safely",
                     "Every step, retry and approval must be visible",
                     "Everything runs locally and costs nothing"],
         "frames": [{"text": ("A single chat assistant is not enough for a real project. We need several agents that work side by side, "
                              f"fail safely, and show every step. Our system runs locally, uses a {model} model, and costs zero rupees.")}]},
        {"title": "Three agents, one dashboard", "kind": "diagram", "frames": _frames("agents", [
            ("OR", "The orchestrator starts each agent as its own background job."),
            ("BUS", "Every agent writes each step to one event bus, and the dashboard shows it live.")])},
        {"title": "The coding agent loop", "kind": "diagram", "frames": _frames("loop", [
            ("P", "The coding agent first makes a short plan."),
            ("A", f"Then it asks the local {model} model for exactly one action."),
            ("G", "The action is checked against a strict JSON schema, and invalid answers are retried."),
            ("T", "A valid action runs one tool, such as writing a file or running Python."),
            ("O", "The result comes back as an observation."),
            ("R", "If steps keep failing, the agent reflects and changes its approach."),
            ("F", "It may only finish when the code has really run successfully."),
            ("C", "Long histories are compacted so the small model keeps its focus.")])},
        {"title": "Approval gate and sandbox", "kind": "diagram", "frames": _frames("approval", [
            ("B", "Shell commands go through a safety list first. Dangerous ones are blocked and logged."),
            ("Q", "Everything else waits for your click in the dashboard."),
            ("S", "Approved commands run in a sandbox folder. Denied or unanswered requests are refused.")])},
        {"title": "Orchestration and fallbacks", "kind": "diagram", "frames": _frames("orchestrator", [
            ("OR", "The orchestrator runs the coding, image and video jobs in parallel."),
            ("J2", f"The image job asks Cloudflare FLUX first. This time the logo came from {f['image']}."),
            ("P2", "If that fails after retries, it falls back to Pollinations."),
            ("LM", "Only the coding and research agents use the local model, one at a time."),
            ("BUS", "A failed job never stops the others. Each one reports done, failed or stopped.")])},
        {"title": "A real run, from the log", "kind": "timeline", "frames": [{"text": n, "code": code}]},
        {"title": "Dashboard tour and limits", "kind": "text",
         "bullets": ["Tabs: team and logo, video, coding agent, startup research",
                     "Every step, retry and approval is shown and logged",
                     "Limits: a small local model can make mistakes",
                     "Startup figures come from public sources; check them"],
         "frames": [{"text": ("The dashboard has four tabs: team and logo, this video, the coding agent, and startup research. "
                              "Every step, retry and approval appears there and in the log file. The main limits are that a small local "
                              "model can make mistakes, and that startup figures come from public sources and must be checked.")}]},
    ]