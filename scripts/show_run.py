"""Prints a short timeline of a run from its events.jsonl (nothing secret is in there):

    python scripts/show_run.py              the newest runs/session-*/events.jsonl
    python scripts/show_run.py path.jsonl   a specific file

Paste this short output instead of the whole log when you report a problem."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def first_line(text: str, n: int = 110) -> str:
    lines = [l for l in str(text).strip().splitlines() if l.strip()] or [""]
    return (lines[-1] if str(text).startswith(("EXIT", "TIMEOUT")) else lines[0])[:n]


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted((ROOT / "runs").glob("session-*/events.jsonl"))
        if not files:
            sys.exit("No runs/session-*/events.jsonl yet.")
        path = files[-1]
    print(f"# {path}")
    t0 = None
    for line in path.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        k = ev["kind"]
        t0 = t0 or ev["t"]
        at = f"{ev['t'] - t0:6.1f}s"
        s = f"s{ev['step']}" if "step" in ev else "  "
        if k == "task_started":
            print(f"{at} TASK   {str(ev.get('task'))[:100]}")
        elif k == "plan":
            print(f"{at} PLAN   ({ev['reason']}) {ev['text'].splitlines()[0][:90] if ev['text'] else ''}")
        elif k == "action":
            a = ev["args"]
            tgt = a.get("path") or a.get("command") or a.get("query") or (a.get("code") or "")[:40] or a.get("answer", "")[:60]
            print(f"{at} {s} ACT    {ev['tool']} {tgt} (tries {ev.get('attempts', 1)})")
        elif k == "observation":
            print(f"{at} {s} OBS    {'ok ' if ev['ok'] else 'ERR'} {first_line(ev['text'])}")
        elif k == "llm_call":
            print(f"{at} {s} llm    {ev['completion_tokens']} tokens, {ev['seconds']} s, prompt {ev['prompt_tokens']}")
        elif k in ("retry", "reflect", "compact", "finish_refused", "repeat_guard", "blocked", "seeded",
                   "approval_request", "approval_result",
                   "done", "budget_exhausted", "stopped", "error", "job_done", "job_failed", "job_stopped"):
            extra = {x: v for x, v in ev.items() if x not in ("seq", "t", "kind", "job", "agent", "step", "traceback", "summary")}
            print(f"{at} {s} {k.upper():<8}{json.dumps(extra, default=str)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())