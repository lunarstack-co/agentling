import dataclasses
import random
from collections.abc import AsyncIterator, Sequence

import pytest

from agentling.agent import Agent
from agentling.errors import ModelError, ModelRateLimitError, ModelServerError
from agentling.models import ChatMessage, Delta, ToolDefinition
from agentling.retry import DEFAULT_RETRY, RetryPolicy, retry_delay


class _FixedRng:
    """An rng for retry_delay that returns one value and records its bounds."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[float, float]] = []

    def __call__(self, low: float, high: float) -> float:
        self.calls.append((low, high))
        return self.value


def _never_called(low: float, high: float) -> float:
    raise AssertionError("rng should not be called")


class _StreamOnly:
    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        yield Delta(content="unused")


def test_defaults_and_frozen() -> None:
    policy = RetryPolicy()

    assert policy.max_retries == 2
    assert policy.base_delay == 0.5
    assert policy.max_delay == 30.0
    assert policy.jitter == 0.2
    assert policy.on_retry is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_retries = 5  # type: ignore[misc]


@pytest.mark.parametrize(("attempt", "expected"), [(0, 0.5), (1, 1.0), (2, 2.0)])
def test_delay_grows_exponentially(attempt: int, expected: float) -> None:
    assert retry_delay(RetryPolicy(), attempt, ModelServerError(), _FixedRng(1.0)) == (
        expected
    )


def test_retry_after_wins() -> None:
    exc = ModelRateLimitError(retry_after=3)
    assert retry_delay(RetryPolicy(), 2, exc, _FixedRng(1.0)) == 3.0


@pytest.mark.parametrize(
    ("attempt", "exc"),
    [(0, ModelRateLimitError(retry_after=10_000)), (10, ModelServerError())],
)
def test_delay_is_capped(attempt: int, exc: ModelError) -> None:
    assert retry_delay(RetryPolicy(), attempt, exc, _FixedRng(1.0)) == 30.0


def test_cap_applies_after_jitter() -> None:
    exc = ModelRateLimitError(retry_after=30)
    assert retry_delay(RetryPolicy(), 0, exc, _FixedRng(1.2)) == 30.0


def test_jitter_bounds() -> None:
    rng = _FixedRng(1.0)
    retry_delay(RetryPolicy(), 0, ModelServerError(), rng)
    assert rng.calls == [pytest.approx((0.8, 1.2))]

    seeded = random.Random(0)
    samples = [
        retry_delay(RetryPolicy(), 0, ModelServerError(), seeded.uniform)
        for _ in range(1000)
    ]
    assert all(0.4 <= sample <= 0.6 for sample in samples)


def test_no_jitter_means_no_rng_call() -> None:
    policy = RetryPolicy(jitter=0)
    assert retry_delay(policy, 0, ModelServerError(), _never_called) == 0.5


def test_retry_after_zero_gives_zero_delay() -> None:
    exc = ModelRateLimitError(retry_after=0)
    assert retry_delay(RetryPolicy(), 3, exc) == 0.0


@pytest.mark.parametrize(
    ("policy", "exc", "factor"),
    [
        (RetryPolicy(), ModelRateLimitError(retry_after=-5), 1.0),
        (RetryPolicy(jitter=2.0), ModelServerError(), -1.0),
    ],
)
def test_delay_is_never_negative(
    policy: RetryPolicy, exc: ModelError, factor: float
) -> None:
    assert retry_delay(policy, 0, exc, _FixedRng(factor)) == 0.0


def test_agent_retry_wiring() -> None:
    assert Agent(model=_StreamOnly()).retry is DEFAULT_RETRY
    assert DEFAULT_RETRY == RetryPolicy()
    assert Agent(model=_StreamOnly(), retry=None).retry is None
