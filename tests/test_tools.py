from typing import Any, Literal, Optional

import pytest

from agentling.tools import FinalAnswerTool, Tool, ToolCallError, _check_tool, tool


# --------------------------------------------------------------------------- #
# Schema generation
# --------------------------------------------------------------------------- #
def test_basic_types_required_and_descriptions() -> None:
    @tool
    def add(a: int, b: int = 0) -> int:
        """Add two integers.

        Args:
            a: The first number.
            b: The second number.
        """
        return a + b

    fn = add.to_schema()["function"]
    props = fn["parameters"]["properties"]

    assert fn["name"] == "add"
    assert fn["description"] == "Add two integers."
    assert props["a"]["type"] == "integer"
    assert props["a"]["description"] == "The first number."
    assert props["b"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["a"]  # b has a default


def test_literal_becomes_enum() -> None:
    @tool
    def pick(unit: Literal["c", "f"] = "c") -> str:
        return unit

    prop = pick.to_schema()["function"]["parameters"]["properties"]["unit"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["c", "f"]
    assert prop["default"] == "c"  # defaults are now exposed in the schema


def test_optional_is_unwrapped() -> None:
    @tool
    def f(x: Optional[int] = None, y: str | None = None) -> str:
        return "ok"

    schema = f.to_schema()["function"]["parameters"]
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "string"
    assert schema["required"] == []


def test_list_and_dict() -> None:
    @tool
    def f(tags: list[str], meta: dict) -> str:
        return "ok"

    props = f.to_schema()["function"]["parameters"]["properties"]
    assert props["tags"]["type"] == "array"
    assert props["meta"]["type"] == "object"


def test_array_includes_items() -> None:
    @tool
    def f(tags: list[str]) -> str:
        return ",".join(tags)

    prop = f.to_schema()["function"]["parameters"]["properties"]["tags"]
    assert prop == {"type": "array", "items": {"type": "string"}}


def test_default_is_exposed() -> None:
    @tool
    def f(limit: int = 5) -> int:
        return limit

    prop = f.to_schema()["function"]["parameters"]["properties"]["limit"]
    assert prop["default"] == 5


def test_schema_forbids_additional_properties() -> None:
    @tool
    def f(a: int) -> int:
        return a

    assert f.to_schema()["function"]["parameters"]["additionalProperties"] is False


def test_varargs_are_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported parameter"):

        @tool
        def f(*args: str) -> str:
            return "x"


# --------------------------------------------------------------------------- #
# @tool execution (sync + async)
# --------------------------------------------------------------------------- #
async def test_sync_tool_runs() -> None:
    @tool
    def greet(name: str) -> str:
        return f"hi {name}"

    assert await greet.call({"name": "Ada"}) == "hi Ada"


async def test_async_tool_runs() -> None:
    @tool
    async def fetch(url: str) -> str:
        return f"data:{url}"

    assert await fetch.call({"url": "x"}) == "data:x"


# --------------------------------------------------------------------------- #
# Argument validation -> ToolCallError
# --------------------------------------------------------------------------- #
@pytest.fixture
def weather() -> Tool:
    @tool
    def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
        """Get weather.

        Args:
            city: The city.
        """
        return f"{city}:{units}"

    return get_weather


async def test_valid_call(weather: Tool) -> None:
    assert await weather.call({"city": "Paris"}) == "Paris:c"


async def test_missing_required(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="missing required"):
        await weather.call({})


async def test_unknown_argument(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="unexpected argument"):
        await weather.call({"city": "Paris", "foo": 1})


async def test_wrong_type(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="expects string"):
        await weather.call({"city": 123})


async def test_bad_enum(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="must be one of"):
        await weather.call({"city": "Paris", "units": "kelvin"})


async def test_non_dict_arguments_rejected(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="arguments object"):
        await weather.call(["Paris"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FinalAnswerTool
# --------------------------------------------------------------------------- #
async def test_final_answer_tool() -> None:
    final = FinalAnswerTool()
    props = final.to_schema()["function"]["parameters"]["properties"]

    assert final.name == "final_answer"
    assert "answer" in props
    assert await final.call({"answer": "42"}) == "42"


# --------------------------------------------------------------------------- #
# @tool metadata and name validation
# --------------------------------------------------------------------------- #
def test_bare_tool_decorator_has_default_metadata() -> None:
    @tool
    def t(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert isinstance(t, Tool)
    assert t.timeout is None
    assert t.parallel_safe is True
    assert t.max_output_chars is None


def test_tool_decorator_accepts_metadata() -> None:
    @tool(timeout=1.5, parallel_safe=False, max_output_chars=100)
    def t(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert isinstance(t, Tool)
    assert t.timeout == 1.5
    assert t.parallel_safe is False
    assert t.max_output_chars == 100


def test_invalid_tool_name_is_rejected() -> None:
    def f() -> str:
        """A tool with a bad name."""
        return ""

    f.__name__ = "bad name!"
    with pytest.raises(ValueError, match="valid tool name"):
        tool(f)


# --------------------------------------------------------------------------- #
# Schemas from outside (JSON Schema tolerance)
# --------------------------------------------------------------------------- #
class _EchoTool(Tool):
    """A tool built from a raw schema, as an MCP or OpenAPI bridge would."""

    async def forward(self, **kwargs: Any) -> Any:
        return kwargs


def _schema_tool(
    parameters: Any, *, name: str = "t", validate_arguments: bool = True
) -> Tool:
    built = _EchoTool()
    built.name = name
    built.description = "A tool whose schema came from somewhere else."
    built.parameters = parameters
    built.validate_arguments = validate_arguments
    return built


def _one_property(spec: Any) -> dict[str, Any]:
    return {"type": "object", "properties": {"x": spec}}


async def test_union_type_accepts_each_member() -> None:
    t = _schema_tool(_one_property({"type": ["string", "null"]}))

    assert await t.call({"x": "a"}) == {"x": "a"}
    assert await t.call({"x": None}) == {"x": None}


async def test_union_type_rejects_non_member() -> None:
    t = _schema_tool(_one_property({"type": ["string", "null"]}))

    with pytest.raises(
        ToolCallError, match=r"argument 'x' expects \['string', 'null'\], got int"
    ):
        await t.call({"x": 5})


async def test_union_type_never_raises_type_error() -> None:
    t = _schema_tool(_one_property({"type": ["integer", "weird"]}))

    assert await t.call({"x": "s"}) == {"x": "s"}


async def test_bool_rejected_for_integer_in_union() -> None:
    t = _schema_tool(_one_property({"type": ["integer", "null"]}))

    with pytest.raises(ToolCallError, match="expects"):
        await t.call({"x": True})


async def test_null_type_alone() -> None:
    t = _schema_tool(_one_property({"type": "null"}))

    assert await t.call({"x": None}) == {"x": None}
    with pytest.raises(ToolCallError, match="expects null"):
        await t.call({"x": 0})


async def test_unknown_type_name_is_not_enforced() -> None:
    t = _schema_tool(_one_property({"type": "uuid"}))

    assert await t.call({"x": 123}) == {"x": 123}


async def test_unimplemented_keywords_are_ignored() -> None:
    t = _schema_tool(
        {
            "type": "object",
            "properties": {
                "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "b": {"$ref": "#/defs/x"},
                "c": {"type": "string", "format": "email"},
            },
        }
    )

    arguments = {"a": 1, "b": [], "c": "not-an-email"}
    assert await t.call(arguments) == arguments


async def test_missing_properties_accepts_any_argument() -> None:
    t = _schema_tool({"type": "object"})

    assert await t.call({"q": "hi", "n": 2}) == {"q": "hi", "n": 2}


async def test_additional_properties_true_allows_extras() -> None:
    t = _schema_tool(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
    )

    assert await t.call({"a": "x", "extra": 1}) == {"a": "x", "extra": 1}


async def test_additional_properties_absent_allows_extras() -> None:
    t = _schema_tool({"type": "object", "properties": {"a": {"type": "string"}}})

    assert await t.call({"a": "x", "extra": 1}) == {"a": "x", "extra": 1}


async def test_additional_properties_false_rejects_extras() -> None:
    t = _schema_tool(
        {"type": "object", "properties": {}, "additionalProperties": False}
    )

    with pytest.raises(ToolCallError, match=r"got unexpected argument\(s\): q"):
        await t.call({"q": 1})


async def test_required_still_enforced_for_schema_tools() -> None:
    t = _schema_tool(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
    )

    with pytest.raises(ToolCallError, match=r"missing required argument\(s\): a"):
        await t.call({})


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "properties": "oops"},
        {"type": "object", "properties": {"a": "string"}},
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"},
        {"type": "object", "properties": {"a": {"enum": "abc"}}},
        {"type": "object", "properties": {"a": {"type": 5}}},
    ],
)
async def test_malformed_schema_parts_never_raise_other_errors(
    parameters: dict[str, Any],
) -> None:
    t = _schema_tool(parameters)

    try:
        await t.call({"a": "x"})
    except ToolCallError:
        pass


async def test_enum_still_enforced() -> None:
    t = _schema_tool(_one_property({"type": "string", "enum": ["c", "f"]}))

    with pytest.raises(ToolCallError, match=r"must be one of \['c', 'f'\]"):
        await t.call({"x": "k"})


# --------------------------------------------------------------------------- #
# validate_arguments and the public validate()
# --------------------------------------------------------------------------- #
def test_validate_arguments_defaults_true() -> None:
    @tool
    def t(x: int) -> int:
        """Doc."""
        return x

    assert _EchoTool.validate_arguments is True
    assert t.validate_arguments is True


async def test_validation_disabled_skips_schema_checks() -> None:
    t = _schema_tool(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        validate_arguments=False,
    )

    assert await t.call({"zzz": 1}) == {"zzz": 1}


async def test_validation_disabled_still_requires_dict() -> None:
    t = _schema_tool({"type": "object"}, validate_arguments=False)

    with pytest.raises(ToolCallError, match="expected an arguments object, got list"):
        await t.call(["a"])  # type: ignore[arg-type]


async def test_tool_decorator_sets_validate_arguments() -> None:
    @tool(validate_arguments=False)
    def t(x: int) -> int:
        """Doc."""
        return x

    assert t.validate_arguments is False
    # Reaching the function's own TypeError proves validation was skipped.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        await t.call({"x": 1, "unknown": 2})


def test_validate_is_public_and_honours_flag() -> None:
    t = _schema_tool(
        {"type": "object", "properties": {}, "additionalProperties": False}
    )

    with pytest.raises(ToolCallError, match=r"got unexpected argument\(s\): bad"):
        t.validate({"bad": 1})

    t.validate_arguments = False
    t.validate({"bad": 1})


async def test_call_goes_through_validate() -> None:
    order: list[str] = []

    class _Recording(_EchoTool):
        def validate(self, arguments: dict[str, Any]) -> None:
            order.append(f"validate:{arguments}")
            super().validate(arguments)

        async def forward(self, **kwargs: Any) -> Any:
            order.append("forward")
            return kwargs

    t = _Recording()
    t.name = "rec"
    t.description = "Records the call order."
    t.parameters = {"type": "object"}

    await t.call({"x": 1})

    assert order == ["validate:{'x': 1}", "forward"]


# --------------------------------------------------------------------------- #
# Registration checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_name", ["issues.create", "a b", "", "x" * 65, "naïve", "trailing\n"]
)
def test_check_tool_rejects_invalid_names(bad_name: str) -> None:
    t = _schema_tool({"type": "object"}, name=bad_name)

    with pytest.raises(ValueError) as excinfo:
        _check_tool(t)

    assert str(excinfo.value) == (
        f"Tool name {bad_name!r} is not a valid tool name: use letters, "
        "digits, underscore, or hyphen, up to 64 characters."
    )


@pytest.mark.parametrize("good_name", ["a", "x" * 64, "mcp__linear__create-issue"])
def test_check_tool_accepts_valid_names(good_name: str) -> None:
    _check_tool(_schema_tool({"type": "object"}, name=good_name))


@pytest.mark.parametrize("parameters", [{"type": "string"}, {}, [], None])
def test_check_tool_rejects_non_object_parameters(parameters: Any) -> None:
    t = _schema_tool(parameters)

    with pytest.raises(ValueError) as excinfo:
        _check_tool(t)

    assert str(excinfo.value) == (
        'Tool \'t\' parameters must be a JSON Schema object with "type": "object".'
    )
