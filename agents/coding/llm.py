"""Client for llama-server (OpenAI-style HTTP API).

One place for: the health check, JSON-constrained action calls, retries, and logging of
seconds and token usage. The agent never needs to know HTTP details.
"""
import time
from dataclasses import dataclass

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_fixed

from agents.coding.actions import ACTION_SCHEMA, Action


class LLMError(RuntimeError):
    """The server answered, but we cannot use the answer (or it refused the request)."""


class LLMUnavailable(LLMError):
    """The server is not reachable or is still loading the model. Worth retrying."""


@dataclass
class LLMReply:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    tokens_per_second: float
    reasoning_chars: int = 0


@dataclass
class ActionResult:
    action: Action
    raw: str            # the exact JSON text; put this in the history as the assistant message
    reply: LLMReply
    attempts: int       # 1 = valid on the first try


def _server_message(r: requests.Response) -> str:
    try:
        return str(r.json()["error"]["message"])[:300]
    except Exception:
        return r.text[:300]


class LLMClient:
    def __init__(self, settings, bus=None, retry_wait=None, json_retries: int = 2, read_timeout: int = 900):
        self.base = settings.llm_url
        self.max_tokens = settings.max_tokens
        self.bus = bus
        self.json_retries = json_retries
        self.timeout = (5, read_timeout)        # (connect, read) seconds; CPU generation is slow
        # Connection retries: 3 tries, waits 2 s then 4 s. Tests pass retry_wait=0.
        self._wait = wait_exponential(multiplier=2, min=2, max=8) if retry_wait is None else wait_fixed(retry_wait)

    def _emit(self, kind: str, tags, **data) -> None:
        if self.bus:
            self.bus.emit(kind, **{**(tags or {}), **data})

    # ---------------------------------------------------------------- health
    def health(self, timeout: float = 1.5):
        """Returns (state, detail); state is 'online', 'loading' or 'offline'."""
        try:
            r = requests.get(f"{self.base}/health", timeout=timeout)
        except requests.RequestException:
            return "offline", f"cannot reach {self.base}"
        if r.status_code == 200:
            return "online", "model loaded"
        if r.status_code == 503:
            return "loading", "model is still loading"
        return "offline", f"unexpected HTTP {r.status_code}"

    # ------------------------------------------------------------------ http
    def _post(self, body: dict) -> dict:
        try:
            r = requests.post(f"{self.base}/v1/chat/completions", json=body, timeout=self.timeout)
        except requests.ReadTimeout:
            raise LLMError(f"no answer within {self.timeout[1]} s") from None
        except requests.ConnectionError:
            raise LLMUnavailable(f"cannot reach llama-server at {self.base}. Is it running?") from None
        except requests.RequestException as e:
            raise LLMError(f"request failed: {e}") from None
        if r.status_code == 503:
            raise LLMUnavailable("llama-server is still loading the model")
        if r.status_code != 200:
            raise LLMError(f"llama-server HTTP {r.status_code}: {_server_message(r)}")
        return r.json()

    def _on_retry(self, state, tags) -> None:
        self._emit("retry", tags, what="llm_connect", attempt=state.attempt_number,
                   reason=str(state.outcome.exception()), wait_s=round(state.next_action.sleep, 1))

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list, schema=None, tags=None, temperature: float = 0.2,
             max_tokens=None) -> LLMReply:
        body = {
            "model": "local",                                   # llama-server ignores the name
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},  # no hidden reasoning: too slow on CPU
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "action", "schema": schema}}

        for attempt in Retrying(stop=stop_after_attempt(3), wait=self._wait,
                                retry=retry_if_exception_type(LLMUnavailable), reraise=True,
                                before_sleep=lambda st: self._on_retry(st, tags)):
            with attempt:
                t0 = time.time()
                data = self._post(body)
                seconds = time.time() - t0

        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            usage = data.get("usage") or {}
            timings = data.get("timings") or {}
            completion = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
            tps = timings.get("predicted_per_second") or (completion / seconds if seconds > 0 else 0)
            reply = LLMReply(
                content=msg.get("content") or "",
                finish_reason=choice.get("finish_reason") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=completion,
                seconds=round(seconds, 1),
                tokens_per_second=round(float(tps), 1),
                reasoning_chars=len(msg.get("reasoning_content") or ""),
            )
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"unexpected answer from llama-server: {e!r}") from None

        self._emit("llm_call", tags, seconds=reply.seconds, prompt_tokens=reply.prompt_tokens,
                   completion_tokens=reply.completion_tokens, tokens_per_second=reply.tokens_per_second,
                   finish_reason=reply.finish_reason, reasoning_chars=reply.reasoning_chars)
        return reply

    # ---------------------------------------------------------------- action
    def action(self, messages: list, tags=None) -> ActionResult:
        """Ask for one action. Invalid JSON is retried up to json_retries times, and the
        problem is told to the model so it can correct itself."""
        msgs = list(messages)
        problem = ""
        for attempt in range(1, self.json_retries + 2):
            reply = self.chat(msgs, schema=ACTION_SCHEMA, tags=tags)
            try:
                if reply.finish_reason == "length":
                    raise ValueError(f"your answer was cut off after {reply.completion_tokens} tokens; "
                                     "write shorter code or split the work into smaller steps")
                act = Action.model_validate_json(reply.content)
                return ActionResult(action=act, raw=reply.content, reply=reply, attempts=attempt)
            except ValueError as e:                 # pydantic's ValidationError is a ValueError
                problem = " ".join(str(e).split())[:300]
                self._emit("retry", tags, what="invalid_json", attempt=attempt, reason=problem)
                msgs += [{"role": "assistant", "content": reply.content[:1500]},
                         {"role": "user", "content": f"Invalid action: {problem}. "
                                                     "Reply again with one valid JSON action."}]
        raise LLMError(f"the model gave no valid action after {self.json_retries + 1} tries: {problem}")
    