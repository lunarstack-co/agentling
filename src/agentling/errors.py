"""Exception hierarchy for agentling.

agentling's own errors descend from AgentlingError. Other exceptions can still
reach the caller: ValueError and TypeError for bad configuration, and the
provider SDK's errors (such as openai.APIError) once retries are used up.
"""

from __future__ import annotations


class AgentlingError(Exception):
    """Base class for agentling's own errors."""


class AgentRunError(AgentlingError):
    """A run could not start or continue.

    Reserved: agentling does not raise this yet. It is exported so code can
    catch it once it does.
    """


class ModelError(AgentlingError):
    """The run stopped because of the model.

    Raised when a model turn times out (`model_timeout`) or the model keeps
    producing tool calls that can't be parsed.
    """


class ModelOutputError(AgentlingError):
    """The model produced a tool call that couldn't be parsed.

    Covers invalid or non-object JSON arguments and a missing tool name or id.
    The agent loop catches it and asks the model to try again, up to twice,
    before raising ModelError.
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

    Reserved: agentling does not raise this yet. Today a malformed SKILL.md
    raises ValueError, and a bad tool entry point raises ValueError or TypeError.
    """
