"""Coding agent. Step 2 only has run_one_step(); Step 4 adds the full agent_loop()."""
from agents.coding.llm import LLMClient
from agents.coding.prompts import SYSTEM_PROMPT


def run_one_step(job, bus, settings, task: str) -> None:
    """One LLM call: the model picks the first action for the task. Nothing is executed yet."""
    tags = {"job": job.id, "agent": job.agent, "step": 1}
    bus.emit("task_started", job=job.id, agent=job.agent, task=task, model=settings.model_name)
    client = LLMClient(settings, bus)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task}]
    res = client.action(messages, tags=tags)
    bus.emit("thought", **tags, text=res.action.thought)
    bus.emit("action", **tags, tool=res.action.tool, args=res.action.args, attempts=res.attempts)