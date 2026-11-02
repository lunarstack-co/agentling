import pytest

import agentling


def test_every_name_in_all_resolves() -> None:
    for name in agentling.__all__:
        assert hasattr(agentling, name), f"{name!r} is in __all__ but not importable"


def test_key_symbols_are_the_expected_objects() -> None:
    from agentling import Agent, RetryPolicy, Skill, ToolDefinition, print_events, tool
    from agentling.agent import Agent as AgentImpl
    from agentling.models import ToolDefinition as ToolDefinitionImpl
    from agentling.retry import RetryPolicy as RetryPolicyImpl
    from agentling.skills import Skill as SkillImpl

    assert Agent is AgentImpl
    assert ToolDefinition is ToolDefinitionImpl
    assert RetryPolicy is RetryPolicyImpl
    assert Skill is SkillImpl
    assert callable(tool)
    assert callable(print_events)


def test_tool_call_error_is_under_the_agentling_base() -> None:
    from agentling import AgentlingError, ToolCallError

    assert issubclass(ToolCallError, AgentlingError)


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(agentling.__version__, str)
    assert agentling.__version__


def test_openai_model_is_gone() -> None:
    import agentling.models

    with pytest.raises(ImportError):
        from agentling import OpenAIModel  # type: ignore[attr-defined]  # noqa: F401
    assert not hasattr(agentling.models, "ToolSpec")
    assert "OpenAIModel" not in agentling.__all__


_MODEL_ERRORS = [
    "ModelError",
    "ModelConnectionError",
    "ModelRateLimitError",
    "ModelServerError",
    "ModelTimeoutError",
    "ModelAuthError",
    "ModelBadRequestError",
    "ModelContextLengthError",
    "ModelOutputError",
]


@pytest.mark.parametrize("name", _MODEL_ERRORS)
def test_model_errors_are_exported_from_both_places(name: str) -> None:
    import agentling.errors

    assert name in agentling.__all__
    assert getattr(agentling, name) is getattr(agentling.errors, name)


@pytest.mark.parametrize(
    "name", ["ToolDefinition", "FinishReason", "deltas_from_message", "RetryPolicy"]
)
def test_provider_contract_names_are_exported(name: str) -> None:
    assert name in agentling.__all__
    assert hasattr(agentling, name)
