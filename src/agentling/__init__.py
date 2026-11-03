"""Agentling: a small agent engine for Python.

Import everything from `agentling`. The submodules are internal and may move.
"""

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent, AgentSession, StepCallback
from .errors import (
    AgentlingError,
    AgentRunError,
    MemoryLoadError,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContextLengthError,
    ModelError,
    ModelOutputError,
    ModelOutputTruncatedError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    SkillLoadError,
    ToolExecutionError,
    ToolTimeoutError,
)
from .events import (
    Event,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    print_events,
)
from .memory import ActionStep, FinalStep, Memory, Step, TaskStep, ToolResult
from .models import (
    ChatMessage,
    Delta,
    FinishReason,
    Model,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
    deltas_from_message,
)
from .retry import RetryPolicy
from .skills import Skill
from .tools import Tool, ToolCallError, tool

try:
    __version__ = version("agentling")
except PackageNotFoundError:  # Running from a source tree without an install.
    __version__ = "0.0.0"

__all__ = [
    # Core
    "Agent",
    "AgentSession",
    "StepCallback",
    "Model",
    "RetryPolicy",
    "Skill",
    "Tool",
    "tool",
    "print_events",
    # Errors
    "AgentlingError",
    "AgentRunError",
    "ModelError",
    "ModelConnectionError",
    "ModelRateLimitError",
    "ModelServerError",
    "ModelTimeoutError",
    "ModelAuthError",
    "ModelBadRequestError",
    "ModelContextLengthError",
    "ModelOutputError",
    "ModelOutputTruncatedError",
    "ToolCallError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "MemoryLoadError",
    "SkillLoadError",
    # Messages and model types
    "ChatMessage",
    "ToolCall",
    "ToolCallDelta",
    "Delta",
    "Usage",
    "ToolDefinition",
    "FinishReason",
    "agglomerate_deltas",
    "deltas_from_message",
    # Memory
    "Memory",
    "Step",
    "TaskStep",
    "ActionStep",
    "FinalStep",
    "ToolResult",
    # Events
    "Event",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "StepEvent",
    "FinalEvent",
    # Version
    "__version__",
]
