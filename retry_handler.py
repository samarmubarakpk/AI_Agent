"""
retry_handler.py
Robust retry logic using tenacity.
Handles 429 rate limits, transient API failures, network errors.
All AI calls should be wrapped with these decorators.
"""

import time
import anthropic
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    before_sleep_log,
    RetryError
)
import logging

logger = logging.getLogger("agent")


def is_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    # Anthropic rate limit or server error
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code in (429, 500, 502, 503, 529):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    # Requests errors
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        if hasattr(exc, "response") and exc.response is not None:
            return exc.response.status_code in (429, 500, 502, 503)
    return False


# ── Decorators ──────────────────────────────────────────────────────────────────

def with_retry(max_attempts: int = 4, min_wait: int = 2, max_wait: int = 60):
    """
    Decorator factory for retrying AI API calls.
    Uses exponential backoff: 2s, 4s, 8s, 16s...
    """
    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


def call_with_retry(fn, *args, max_attempts=4, **kwargs):
    """
    Functional version — wrap any callable with retry logic.
    Returns result or raises after max attempts.

    Usage:
        result = call_with_retry(client.messages.create, model=..., ...)
    """
    decorated = with_retry(max_attempts=max_attempts)(fn)
    return decorated(*args, **kwargs)


def safe_api_call(fn, *args, fallback=None, label="API call", **kwargs):
    """
    Wraps an API call with retry + final fallback.
    If all retries fail, returns fallback value instead of crashing.

    Usage:
        result = safe_api_call(client.messages.create, model=..., fallback=None, label="Claude vision")
    """
    try:
        return call_with_retry(fn, *args, **kwargs)
    except RetryError as e:
        logger.error(f"{label} failed after all retries: {e}")
        return fallback
    except Exception as e:
        logger.error(f"{label} unexpected error: {e}")
        return fallback
