"""Exception hierarchy for agentling.

agentling's own errors descend from AgentlingError. Model adapters report
failures as ModelError subclasses, which tell the loop whether a retry can
help. Other exceptions can still reach the caller: ValueError and TypeError for
bad configuration, and anything else an adapter raises, unchanged, since that
is a bug in the adapter rather than a model failure.
"""

from __future__ import annotations

from typing import ClassVar


class AgentlingError(Exception):
    """Base class for agentling's own errors."""


class AgentRunError(AgentlingError):
    """A run could not start or continue.

    Reserved: agentling does not raise this yet. It is exported so code can
    catch it once it does.
    """


class ModelError(AgentlingError):
    """The run stopped because of the model.

    Raised by adapters when a request fails, and by the loop when a model turn
    times out (`model_timeout`) or the model keeps producing tool calls that
    can't be parsed.

    Attributes:
        retryable: whether trying the same request again might succeed.
            Defaults to the subclass's `default_retryable`.
        retry_after: seconds the backend asked us to wait, if it said.
        status: the HTTP status code, when there was one.
    """

    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str = "",
        *,
        retryable: bool | None = None,
        retry_after: float | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable
        self.retry_after = retry_after
        self.status = status


class ModelConnectionError(ModelError):
    """The backend could not be reached (DNS, refused or dropped connection)."""

    default_retryable = True


class ModelRateLimitError(ModelError):
    """The backend asked us to slow down (HTTP 429)."""

    default_retryable = True


class ModelServerError(ModelError):
    """The backend failed on its side (HTTP 5xx)."""

    default_retryable = True


class ModelTimeoutError(ModelError):
    """The model sent nothing within `model_timeout`."""

    default_retryable = True


class ModelAuthError(ModelError):
    """The credentials were missing, wrong or not allowed (HTTP 401/403)."""


class ModelBadRequestError(ModelError):
    """The backend rejected the request itself (HTTP 400/404/422)."""


class ModelContextLengthError(ModelError):
    """The conversation no longer fits the model's context window."""


class ModelOutputError(ModelError):
    """The model's output couldn't be used.

    Covers invalid or non-object JSON arguments and a missing tool name; the
    agent loop asks the model to try again, up to twice, before raising
    ModelError. Also raised, without a retry, when an adapter's provider_data
    isn't JSON-serialisable.
    """


class ToolExecutionError(AgentlingError):
    """A tool failed while running, as opposed to receiving bad arguments.

    Reserved: agentling does not raise this yet. Tool failures reach the model
    as error observations instead.
    """


class ToolTimeoutError(ToolExecutionError):
    """A tool ran past its timeout.

    The loop does not raise it. It uses it to word the error observation the
    model sees.
    """


class MemoryLoadError(AgentlingError):
    """Saved memory could not be loaded (unknown version or bad shape)."""


class SkillLoadError(AgentlingError):
    """A skill could not be loaded.

    Reserved: agentling does not raise this yet. Today a bad SKILL.md raises
    ValueError or yaml.YAMLError, and a bad tool entry point raises ValueError
    or TypeError.
    """
