"""The three things a provider can do besides succeeding, and two helpers for HTTP errors."""
import requests


class WebStopped(Exception):
    """The Stop button was pressed while a web call was waiting or retrying."""


class Skip(Exception):
    """The provider is not even tried (no key, switched off, free quota used up). No network call."""


class ProviderError(Exception):
    """The provider was tried and failed. retryable=True for timeouts, 429 and 5xx; a rejected
    key, an empty result or a plan limit is not retried: the chain moves on to the next provider."""

    def __init__(self, reason: str, retryable: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


def status_problem(r) -> None:
    """Raise a ProviderError for an HTTP error status (does nothing for 2xx and 3xx)."""
    c = r.status_code
    if c < 400:
        return
    if c in (401, 403):
        raise ProviderError(f"HTTP {c}: key rejected or access denied")
    if c == 402:
        raise ProviderError("HTTP 402: credits used up or payment required")
    if c == 429:
        raise ProviderError("HTTP 429: rate limited", True)
    if c in (432, 433):
        raise ProviderError(f"HTTP {c}: plan or credit limit reached")
    if c >= 500:
        raise ProviderError(f"HTTP {c}: server error", True)
    raise ProviderError(f"HTTP {c}")


def request_problem(e: Exception) -> ProviderError:
    """Turn a requests exception into a ProviderError (the message never contains a key)."""
    if isinstance(e, requests.ConnectTimeout):                  # the server never answered the connection
        return ProviderError("timeout: could not connect in time", True)
    if isinstance(e, requests.Timeout):                         # connected, but the answer was too slow
        return ProviderError("timeout", True)
    if isinstance(e, requests.ConnectionError):
        return ProviderError("connection error (offline, DNS or firewall)", True)
    return ProviderError(f"request failed ({type(e).__name__})")