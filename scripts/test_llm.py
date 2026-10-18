"""Smoke test for the model server:  python scripts/test_llm.py
Checks /health, then asks for ONE action through the same client the agent uses."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.coding.llm import LLMClient, LLMError  # noqa: E402
from agents.coding.prompts import SYSTEM_PROMPT  # noqa: E402
from core.settings import load_settings  # noqa: E402

s = load_settings()
client = LLMClient(s)
state, detail = client.health()
print(f"server: {state} ({detail}) at {s.llm_url}")
if state != "online":
    sys.exit("Start the server first:  python scripts/start_llm.py   (wait until /health answers)")

task = "List the files in the workspace folder."
print(f"task:   {task}")
try:
    res = client.action([{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": task}])
except LLMError as e:
    sys.exit(f"FAILED: {e}")

print("action:", json.dumps(res.action.model_dump(), indent=2))
r = res.reply
print(f"usage:  prompt {r.prompt_tokens} tokens | answer {r.completion_tokens} tokens | "
      f"{r.seconds} s | {r.tokens_per_second} tok/s | tries {res.attempts} | "
      f"hidden reasoning chars {r.reasoning_chars}")

