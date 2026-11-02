import pytest

from agentling.errors import (
    AgentlingError,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContextLengthError,
    ModelError,
    ModelOutputError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)

MODEL_ERRORS: list[tuple[type[ModelError], bool]] = [
    (ModelConnectionError, True),
    (ModelRateLimitError, True),
    (ModelServerError, True),
    (ModelTimeoutError, True),
    (ModelAuthError, False),
    (ModelBadRequestError, False),
    (ModelContextLengthError, False),
    (ModelOutputError, False),
]


@pytest.mark.parametrize(("cls", "expected"), MODEL_ERRORS)
def test_default_retryable_per_class(cls: type[ModelError], expected: bool) -> None:
    exc = cls()
    assert exc.retryable is expected
    assert exc.retry_after is None
    assert exc.status is None


def test_base_model_error_is_not_retryable() -> None:
    assert ModelError("boom").retryable is False


def test_constructor_overrides_and_attributes() -> None:
    exc = ModelServerError("x", retryable=False, retry_after=2.5, status=503)

    assert exc.retryable is False
    assert exc.retry_after == 2.5
    assert exc.status == 503
    assert str(exc) == "x"
    assert ModelAuthError(retryable=True).retryable is True


@pytest.mark.parametrize("cls", [cls for cls, _ in MODEL_ERRORS])
def test_hierarchy(cls: type[ModelError]) -> None:
    assert issubclass(cls, ModelError)
    assert issubclass(cls, AgentlingError)
