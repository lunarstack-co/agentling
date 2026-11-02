"""Retrying failed model turns.

The agent loop owns retries; adapters never retry. RetryPolicy says how often
and how long to wait, and retry_delay() is the pure delay calculation.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .errors import ModelError

# on_retry(attempt, max_retries, exc, delay): attempt counts from 1.
OnRetry = Callable[[int, int, ModelError, float], Awaitable[None] | None]


@dataclass(frozen=True)
class RetryPolicy:
    """How the loop retries a model turn.

    A turn is retried only when it raised a ModelError with retryable=True
    before streaming anything, at most `max_retries` times. Before each wait
    the loop calls `on_retry` (awaiting it if it returns an awaitable); if it
    raises, the run stops with that exception.
    """

    max_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 30.0
    # Each delay is scaled by a random factor in [1 - jitter, 1 + jitter] so
    # many clients don't retry in step.
    jitter: float = 0.2
    on_retry: OnRetry | None = None


# Frozen, so every Agent can share it.
DEFAULT_RETRY = RetryPolicy()


def retry_delay(
    policy: RetryPolicy,
    attempt: int,
    exc: ModelError,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Seconds to wait before retrying, where `attempt` counts from 0.

    Doubles from `base_delay` each attempt, unless the backend said how long
    to wait (`exc.retry_after`). Never more than `max_delay`, so a hostile
    Retry-After header can't stall a run, and never below zero.
    """

    if exc.retry_after is not None:
        delay = exc.retry_after
    else:
        delay = policy.base_delay * 2**attempt
    if policy.jitter:
        delay *= rng(1 - policy.jitter, 1 + policy.jitter)
    return max(0.0, min(policy.max_delay, delay))
