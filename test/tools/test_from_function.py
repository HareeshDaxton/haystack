# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal

import jsonschema
import pytest
from docstring_parser import ParseError
from pydantic import BaseModel, Field

from haystack.components.agents.state import State
from haystack.tools.errors import SchemaGenerationError
from haystack.tools.from_function import _remove_title_from_schema, create_tool_from_function, tool
from haystack.tools.tool import Tool


def function_with_docstring(city: str) -> str:
    """Get weather report for a city."""
    return f"Weather report for {city}: 20°C, sunny"


class title(BaseModel):  # noqa: N801 # deliberately lowercase: Pydantic keys '$defs' by class name
    text: str


class Report(BaseModel):
    heading: title
    body: str


def make_report(report: Report) -> str:
    """Create a report."""
    return "ok"


def test_from_function_description_from_docstring():
    tool = create_tool_from_function(function=function_with_docstring)

    assert tool.name == "function_with_docstring"
    assert tool.description == "Get weather report for a city."
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_empty_description():
    tool = create_tool_from_function(function=function_with_docstring, description="")

    assert tool.name == "function_with_docstring"
    assert tool.description == ""
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_custom_description():
    tool = create_tool_from_function(function=function_with_docstring, description="custom description")

    assert tool.name == "function_with_docstring"
    assert tool.description == "custom description"
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_custom_name():
    tool = create_tool_from_function(function=function_with_docstring, name="custom_name")

    assert tool.name == "custom_name"
    assert tool.description == "Get weather report for a city."
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def lookup_order(_tenant: str, order_id: str) -> str:
    """Look up an order for a tenant."""
    return f"{_tenant}/{order_id}"


def test_from_function_required_underscore_parameter_raises():
    with pytest.raises(
        ValueError,
        match=(
            "Function 'lookup_order': required parameter '_tenant' starts with an underscore and "
            "can't be part of the tool schema. Provide it through `inputs_from_state` or give it a default."
        ),
    ):
        create_tool_from_function(function=lookup_order)


def test_from_function_required_underscore_parameter_covered_by_inputs_from_state():
    tool = create_tool_from_function(function=lookup_order, inputs_from_state={"tenant": "_tenant"})

    assert tool.parameters == {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }
    assert tool.invoke(_tenant="acme", order_id="A-1") == "acme/A-1"


def test_from_function_defaulted_underscore_parameter_omitted_from_schema():
    def lookup_with_default(_tenant: str = "acme", order_id: str = "A-1") -> str:
        """Look up an order for a tenant."""
        return f"{_tenant}/{order_id}"

    tool = create_tool_from_function(function=lookup_with_default)

    assert "_tenant" not in tool.parameters["properties"]
    assert tool.parameters == {"type": "object", "properties": {"order_id": {"type": "string", "default": "A-1"}}}
    assert tool.invoke() == "acme/A-1"


def test_from_function_annotated():
    def function_with_annotations(
        city: Annotated[str, "the city for which to get the weather"] = "Munich",
        unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
        nullable_param: Annotated[str | None, "a nullable parameter"] = None,
    ) -> str:
        """A simple function to get the current weather for a location."""
        return f"Weather report for {city}: 20 {unit}, sunny"

    tool = create_tool_from_function(function=function_with_annotations)

    assert tool.name == "function_with_annotations"
    assert tool.description == "A simple function to get the current weather for a location."
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "the city for which to get the weather", "default": "Munich"},
            "unit": {
                "type": "string",
                "enum": ["Celsius", "Fahrenheit"],
                "description": "the unit for the temperature",
                "default": "Celsius",
            },
            "nullable_param": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "a nullable parameter",
                "default": None,
            },
        },
    }


def test_from_function_with_postponed_annotations():
    # String annotations are what `from __future__ import annotations` turns every annotation into.
    def function_with_postponed_annotations(
        city: "Annotated[str, 'the city for which to get the weather']" = "Munich",
        unit: "Annotated[Literal['Celsius', 'Fahrenheit'], 'the unit for the temperature']" = "Celsius",
        state: "State | None" = None,
    ) -> str:
        """A simple function to get the current weather for a location."""
        return f"Weather report for {city}: 20 {unit}, sunny"

    tool = create_tool_from_function(function=function_with_postponed_annotations)

    assert tool.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "the city for which to get the weather", "default": "Munich"},
            "unit": {
                "type": "string",
                "enum": ["Celsius", "Fahrenheit"],
                "description": "the unit for the temperature",
                "default": "Celsius",
            },
        },
    }


def test_from_function_missing_type_hint():
    def function_missing_type_hint(city) -> str:  # type: ignore[no-untyped-def]
        return f"Weather report for {city}: 20°C, sunny"

    with pytest.raises(ValueError):
        create_tool_from_function(function=function_missing_type_hint)


def test_from_function_schema_generation_error():
    def function_with_invalid_type_hint(city: "invalid") -> str:  # type: ignore[name-defined] # noqa: F821
        return f"Weather report for {city}: 20°C, sunny"

    with pytest.raises(SchemaGenerationError):
        create_tool_from_function(function=function_with_invalid_type_hint)


def test_from_function_with_callable_params_skipped():
    def function_with_callback(query: str, callback: Callable[[str], None] | None = None) -> str:
        """A function with a callable parameter."""
        return query

    tool = create_tool_from_function(function=function_with_callback)

    assert tool.name == "function_with_callback"
    param_names = list(tool.parameters.get("properties", {}).keys())
    assert "callback" not in param_names
    assert "query" in param_names


def test_from_function_variadic_params_excluded_from_schema():
    def search(query: str, *args: str, top_k: int = 3, **kwargs: str) -> str:
        """Search the docs."""
        return f"{query} {args} {top_k} {kwargs}"

    tool = create_tool_from_function(function=search)

    assert tool.parameters == {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}},
        "required": ["query"],
    }
    assert tool.invoke(query="haystack", lang="en") == "haystack () 3 {'lang': 'en'}"


def test_from_function_untyped_variadic_params_excluded_from_schema():
    def search(query: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Search the docs."""
        return query

    tool = create_tool_from_function(function=search)

    assert tool.parameters == {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


@pytest.mark.parametrize(
    "callback_type",
    [
        Annotated[Callable[[str], str], "Format the output"],
        Annotated[Callable[[str], str] | None, "Format the output"],
        Annotated[Callable[[str], str], "Format the output"] | None,
    ],
)
def test_from_function_with_annotated_callable_params_skipped(callback_type):
    def format_query(query: str, callback: Callable[[str], str] | None = str.upper) -> str:
        return callback(query) if callback else query

    format_query.__annotations__["callback"] = callback_type
    tool = create_tool_from_function(function=format_query)

    assert tool.parameters == {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    assert tool.invoke(query="hello") == "HELLO"


@pytest.mark.parametrize("state_annotation", [State, Annotated[State, "Live agent state"]])
def test_from_function_state_param_excluded_from_schema(state_annotation: Any) -> None:
    def function_with_state(city: str, state: State) -> str:
        """Get weather for a city, with access to agent state."""
        return f"Weather in {city}: sunny"

    function_with_state.__annotations__["state"] = state_annotation
    tool = create_tool_from_function(function=function_with_state)

    assert tool.name == "function_with_state"
    param_names = list(tool.parameters.get("properties", {}).keys())
    assert "state" not in param_names
    assert "city" in param_names
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def test_tool_decorator_state_param_excluded_from_schema():
    @tool
    def function_with_state(city: str, state: State) -> str:
        """Get weather for a city, with access to agent state."""
        return f"Weather in {city}: sunny"

    param_names = list(function_with_state.parameters.get("properties", {}).keys())
    assert "state" not in param_names
    assert "city" in param_names


def test_from_function_optional_state_param_excluded_from_schema():
    def function_with_optional_state(city: str, state: State | None = None) -> str:
        """Get weather for a city, optionally using agent state."""
        return f"Weather in {city}: sunny"

    tool = create_tool_from_function(function=function_with_optional_state)

    param_names = list(tool.parameters.get("properties", {}).keys())
    assert "state" not in param_names
    assert "city" in param_names


def test_tool_decorator():
    @tool
    def get_weather(city: str) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city}: 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get weather report for a city."
    assert get_weather.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert callable(get_weather.function)
    assert get_weather.function("Berlin") == "Weather report for Berlin: 20°C, sunny"


# Test function for decorator deserialization
@tool
def weather_tool_with_decorator(city: str) -> str:
    """Get weather report for a city."""
    return f"Weather report for {city}: 20°C, sunny"


def test_tool_decorator_deserialization():
    serialized = weather_tool_with_decorator.to_dict()
    deserialized = Tool.from_dict(serialized)
    assert deserialized.name == "weather_tool_with_decorator"
    assert deserialized.description == "Get weather report for a city."
    assert deserialized.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


def test_tool_decorator_with_annotated_params():
    @tool
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        output_format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({output_format} format): 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get weather report for a city."
    assert get_weather.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The target city", "default": "Berlin"},
            "output_format": {
                "type": "string",
                "enum": ["short", "long"],
                "description": "Output format",
                "default": "short",
            },
        },
    }
    assert callable(get_weather.function)
    assert get_weather.function("Berlin", "short") == "Weather report for Berlin (short format): 20°C, sunny"


def test_tool_decorator_with_parameters():
    @tool(name="fetch_weather", description="A tool to check the weather.")
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        output_format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({output_format} format): 20°C, sunny"

    assert get_weather.name == "fetch_weather"
    assert get_weather.description == "A tool to check the weather."


def test_tool_decorator_with_inputs_and_outputs():
    @tool(inputs_from_state={"output_format": "output_format"}, outputs_to_state={"output": {"source": "output"}})
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        output_format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({output_format} format): 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.inputs_from_state == {"output_format": "output_format"}
    assert get_weather.outputs_to_state == {"output": {"source": "output"}}
    # Inputs should be excluded from auto-generated parameters
    assert get_weather.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The target city", "default": "Berlin"}},
    }


def test_remove_title_from_schema():
    complex_schema = {
        "properties": {
            "parameter1": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "default": "default_value",
                "title": "Parameter1",
            },
            "parameter2": {
                "default": [1, 2, 3],
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "title": "Parameter2",
                "type": "array",
            },
            "parameter3": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "type": "array"},
                ],
                "default": 42,
                "title": "Parameter3",
            },
            "parameter4": {
                "anyOf": [{"type": "string"}, {"items": {"type": "integer"}, "type": "array"}, {"type": "object"}],
                "default": {"key": "value"},
                "title": "Parameter4",
            },
        },
        "title": "complex_function",
        "type": "object",
    }

    _remove_title_from_schema(complex_schema)

    assert complex_schema == {
        "properties": {
            "parameter1": {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "default_value"},
            "parameter2": {
                "default": [1, 2, 3],
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "type": "array",
            },
            "parameter3": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "type": "array"},
                ],
                "default": 42,
            },
            "parameter4": {
                "anyOf": [{"type": "string"}, {"items": {"type": "integer"}, "type": "array"}, {"type": "object"}],
                "default": {"key": "value"},
            },
        },
        "type": "object",
    }


def test_remove_title_from_schema_do_not_remove_title_property():
    """Test that the utility function only removes the 'title' keywords and not the 'title' property (if present)."""
    schema = {
        "properties": {
            "parameter1": {"type": "string", "title": "Parameter1"},
            "title": {"type": "string", "title": "Title"},
        },
        "title": "complex_function",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {"properties": {"parameter1": {"type": "string"}, "title": {"type": "string"}}, "type": "object"}


def test_remove_title_from_schema_property_named_properties():
    """Test that a property named 'properties' is not misinterpreted as the 'properties' schema keyword."""
    schema = {
        "properties": {
            "entity_id": {"type": "string", "title": "Entity Id"},
            "properties": {"type": "object", "additionalProperties": True, "title": "Properties"},
        },
        "title": "set_properties",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {
        "properties": {"entity_id": {"type": "string"}, "properties": {"type": "object", "additionalProperties": True}},
        "type": "object",
    }


def test_from_function_with_parameter_named_properties():
    """Creating a tool from a function with a parameter named 'properties' must not crash."""

    def set_properties(
        entity_id: Annotated[str, "the entity to update"], properties: Annotated[dict, "the properties to set"]
    ) -> str:
        """Set properties on an entity."""
        return f"Set {properties} on {entity_id}"

    tool = create_tool_from_function(function=set_properties)

    assert tool.parameters == {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "the entity to update"},
            "properties": {"type": "object", "additionalProperties": True, "description": "the properties to set"},
        },
        "required": ["entity_id", "properties"],
    }


def test_remove_title_from_schema_definition_named_title():
    """Test that a definition named 'title' is kept, so that '$ref's pointing at it keep resolving."""
    schema = {
        "$defs": {
            "title": {"properties": {"text": {"type": "string", "title": "Text"}}, "title": "title", "type": "object"},
            "Addr": {"properties": {"city": {"type": "string", "title": "City"}}, "title": "Addr", "type": "object"},
        },
        "properties": {"heading": {"$ref": "#/$defs/title"}, "addr": {"$ref": "#/$defs/Addr"}},
        "title": "make_report",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    # The definition names survive; only the 'title' *keywords* inside them are stripped.
    assert schema == {
        "$defs": {
            "title": {"properties": {"text": {"type": "string"}}, "type": "object"},
            "Addr": {"properties": {"city": {"type": "string"}}, "type": "object"},
        },
        "properties": {"heading": {"$ref": "#/$defs/title"}, "addr": {"$ref": "#/$defs/Addr"}},
        "type": "object",
    }
    # The schema is still usable: a dangling '$ref' would make validation raise.
    jsonschema.Draft202012Validator(schema).validate({"heading": {"text": "hi"}, "addr": {"city": "Berlin"}})


def test_remove_title_from_schema_definition_named_title_draft_07_spelling():
    """Test that the draft-07 'definitions' spelling is handled like '$defs'."""
    schema = {
        "definitions": {"title": {"type": "string", "title": "title"}},
        "properties": {"heading": {"$ref": "#/definitions/title"}},
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {
        "definitions": {"title": {"type": "string"}},
        "properties": {"heading": {"$ref": "#/definitions/title"}},
        "type": "object",
    }


def test_remove_title_from_schema_keeps_instance_data():
    """Test that 'title' keys inside instance data are left untouched.

    Covers JSON Schema ``default``/``const``/``enum``/``examples`` and the OpenAPI 3.0
    singular ``example`` spelling (Pydantic ``json_schema_extra={"example": ...}``).
    """
    schema = {
        "properties": {
            "cfg": {"type": "object", "default": {"title": "Untitled", "width": 80}, "title": "Cfg"},
            "mode": {"const": {"title": "fast", "workers": 2}, "title": "Mode"},
            "choice": {"enum": [{"title": "A", "id": 1}, {"title": "B", "id": 2}], "title": "Choice"},
            "sample": {
                "type": "object",
                "examples": [{"title": "Example", "id": 1}],
                "example": {"title": "Untitled", "width": "80"},
                "title": "Sample",
            },
        },
        "title": "configure",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    # A 'title' key in a default/const/enum/examples/example is part of the *value*, not a
    # schema keyword: removing it would silently change the tool's contract.
    assert schema == {
        "properties": {
            "cfg": {"type": "object", "default": {"title": "Untitled", "width": 80}},
            "mode": {"const": {"title": "fast", "workers": 2}},
            "choice": {"enum": [{"title": "A", "id": 1}, {"title": "B", "id": 2}]},
            "sample": {
                "type": "object",
                "examples": [{"title": "Example", "id": 1}],
                "example": {"title": "Untitled", "width": "80"},
            },
        },
        "type": "object",
    }


def test_remove_title_from_schema_keeps_pattern_and_dependent_keys():
    """Test that keys of 'patternProperties'/'dependentSchemas'/'dependentRequired' are kept."""
    schema = {
        "patternProperties": {"title": {"type": "string", "title": "Title"}},
        "dependentSchemas": {"title": {"required": ["subtitle"], "title": "Dep"}},
        "dependentRequired": {"title": ["subtitle"]},
        "title": "annotate",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    # These are keyed by regexes and property names, so dropping 'title' would drop a rule.
    assert schema == {
        "patternProperties": {"title": {"type": "string"}},
        "dependentSchemas": {"title": {"required": ["subtitle"]}},
        "dependentRequired": {"title": ["subtitle"]},
        "type": "object",
    }


def test_remove_title_from_schema_still_removes_title_in_subschema_keywords():
    """Test that keywords whose value is a genuine subschema keep losing their 'title'."""
    schema = {
        "properties": {
            "tags": {"items": {"type": "string", "title": "Tag"}, "title": "Tags", "type": "array"},
            "meta": {
                "additionalProperties": {"type": "string", "title": "Value"},
                "propertyNames": {"pattern": "^x-", "title": "Key"},
                "title": "Meta",
                "type": "object",
            },
        },
        "title": "index",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {
        "properties": {
            "tags": {"items": {"type": "string"}, "type": "array"},
            "meta": {"additionalProperties": {"type": "string"}, "propertyNames": {"pattern": "^x-"}, "type": "object"},
        },
        "type": "object",
    }


def test_from_function_with_nested_model_named_title():
    """Creating a tool from a function whose nested model is named 'title' must not dangle a '$ref'."""

    tool = create_tool_from_function(function=make_report)

    assert "title" in tool.parameters["$defs"]
    # A dangling '#/$defs/title' would make any validation of a tool call raise.
    jsonschema.Draft202012Validator(tool.parameters).validate({"report": {"heading": {"text": "hi"}, "body": "b"}})


def test_from_function_with_default_containing_title_key():
    """A default value carrying a 'title' key must reach the model unchanged."""

    def render(options: Annotated[dict, "rendering options"] = {"title": "Untitled", "width": 80}) -> str:  # noqa: B006
        """Render a report."""
        return ""

    tool = create_tool_from_function(function=render)

    assert tool.parameters["properties"]["options"]["default"] == {"title": "Untitled", "width": 80}


def test_from_function_with_openapi_example_containing_title_key():
    """OpenAPI 3.0 singular ``example`` values must keep nested ``title`` keys."""

    def render(
        options: Annotated[dict, "rendering options"] = Field(  # noqa: B008
            default={}, json_schema_extra={"example": {"title": "Untitled", "width": "80"}}
        ),
    ) -> str:
        """Render a document."""
        return ""

    tool = create_tool_from_function(function=render)

    assert tool.parameters["properties"]["options"]["example"] == {"title": "Untitled", "width": "80"}


def test_remove_title_from_schema_handle_no_title_in_top_level():
    schema = {
        "properties": {
            "parameter1": {"type": "string", "title": "Parameter1"},
            "parameter2": {"type": "integer", "title": "Parameter2"},
        },
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {
        "properties": {"parameter1": {"type": "string"}, "parameter2": {"type": "integer"}},
        "type": "object",
    }


async def async_function_with_docstring(city: str) -> str:
    """Get weather report for a city."""
    return f"Weather report for {city}: 20°C, sunny"


class TestFromFunctionAsync:
    def test_create_tool_from_async_function(self):
        tool_obj = create_tool_from_function(async_function_with_docstring)

        assert tool_obj.function is None
        assert tool_obj.async_function is async_function_with_docstring
        assert tool_obj.name == "async_function_with_docstring"
        assert tool_obj.parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    def test_tool_decorator_on_async_function(self):
        decorated = tool(async_function_with_docstring)

        assert decorated.function is None
        assert decorated.async_function is async_function_with_docstring
        assert decorated.name == "async_function_with_docstring"

    @pytest.mark.asyncio
    async def test_invoke_async(self):
        decorated = tool(async_function_with_docstring)

        assert await decorated.invoke_async(city="Berlin") == "Weather report for Berlin: 20°C, sunny"


class TestToolDecoratorExport:
    """`haystack.tools.tool` must be the decorator, although the package also has a `tool` submodule."""

    def test_tool_is_bound_in_package_namespace(self):
        # A fresh interpreter, since other tests may already have cached the attribute on the package.
        code = (
            "import haystack.tools\n"
            "from haystack.tools.from_function import tool\n"
            "assert vars(haystack.tools)['tool'] is tool, vars(haystack.tools).get('tool')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
    @pytest.mark.parametrize("lazy_imports_mode", ["normal", "all"])
    def test_lazy_import_resolves_to_decorator(self, lazy_imports_mode):
        code = (
            "lazy from haystack.tools import tool\n"
            "from haystack.tools.from_function import tool as decorator\n"
            "assert tool is decorator, tool\n"
        )
        env = {**os.environ, "PYTHON_LAZY_IMPORTS": lazy_imports_mode}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env)
        assert result.returncode == 0, result.stderr


@tool(parse_docstring=True)
def weather_tool_with_parsed_docstring(city: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a city.

    :param city: The city to get the weather for.
    :param unit: The temperature unit.
    """
    return f"Weather report for {city}: 20 {unit}, sunny"


class TestParseDocstring:
    def test_default_keeps_whole_docstring_and_adds_no_descriptions(self):
        def get_weather(city: str) -> str:
            """
            Get the current weather.

            :param city: The city to get the weather for.
            """
            return city

        tool_obj = create_tool_from_function(get_weather)

        assert ":param city:" in tool_obj.description
        assert tool_obj.parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    def test_rest_style(self):
        def get_weather(city: str, unit: str = "celsius") -> str:
            """
            Get the current weather for a city.

            Use it only for current conditions, not forecasts.

            :param city: The city to get the weather for.
            :param unit: The temperature unit.
            :returns: A short weather report.
            :raises ValueError: If the city is unknown.
            """
            return city

        tool_obj = create_tool_from_function(get_weather, parse_docstring=True)

        assert tool_obj.description == (
            "Get the current weather for a city.\n\nUse it only for current conditions, not forecasts."
        )
        assert tool_obj.parameters == {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city to get the weather for."},
                "unit": {"type": "string", "description": "The temperature unit.", "default": "celsius"},
            },
            "required": ["city"],
        }

    def test_google_style(self):
        def search_flights(origin: str, destination: str, max_stops: int = 1) -> list[str]:
            """Search flights between two airports.

            Args:
                origin (str): IATA code of the departure airport.
                destination: IATA code of the arrival airport.
                max_stops: Maximum number of stops.

            Returns:
                A list of flight numbers.
            """
            return []

        tool_obj = create_tool_from_function(search_flights, parse_docstring=True)

        assert tool_obj.description == "Search flights between two airports."
        properties = tool_obj.parameters["properties"]
        assert properties["origin"]["description"] == "IATA code of the departure airport."
        assert properties["destination"]["description"] == "IATA code of the arrival airport."
        assert properties["max_stops"]["description"] == "Maximum number of stops."

    def test_numpy_style(self):
        def convert(amount: float, currency: str) -> float:
            """
            Convert an amount to EUR.

            Parameters
            ----------
            amount : float
                The amount of money.
            currency : str
                ISO 4217 currency code.
            """
            return amount

        tool_obj = create_tool_from_function(convert, parse_docstring=True)

        assert tool_obj.description == "Convert an amount to EUR."
        properties = tool_obj.parameters["properties"]
        assert properties["amount"]["description"] == "The amount of money."
        assert properties["currency"]["description"] == "ISO 4217 currency code."

    def test_annotated_description_takes_precedence(self):
        def book(city: Annotated[str, "City from Annotated"], nights: int) -> str:
            """
            Book a hotel.

            :param city: City from the docstring.
            :param nights: Number of nights.
            """
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.parameters["properties"]["city"]["description"] == "City from Annotated"
        assert tool_obj.parameters["properties"]["nights"]["description"] == "Number of nights."

    def test_explicit_description_takes_precedence(self):
        def get_weather(city: str) -> str:
            """
            Get the current weather.

            :param city: The city to get the weather for.
            """
            return city

        tool_obj = create_tool_from_function(get_weather, description="Weather now.", parse_docstring=True)

        assert tool_obj.description == "Weather now."
        assert tool_obj.parameters["properties"]["city"]["description"] == "The city to get the weather for."

    def test_explicit_empty_description_is_kept(self):
        def get_weather(city: str) -> str:
            """
            Get the current weather.

            :param city: The city to get the weather for.
            """
            return city

        tool_obj = create_tool_from_function(get_weather, description="", parse_docstring=True)

        assert tool_obj.description == ""
        assert tool_obj.parameters["properties"]["city"]["description"] == "The city to get the weather for."

    def test_hidden_parameters_are_skipped_without_warning(self, caplog):
        def lookup_order(
            order_id: str,
            user_id: str,
            state: State,
            callback: Callable[[str], None] | None = None,
            _trace_id: str = "",
            *args: Any,
            **kwargs: Any,
        ) -> str:
            """
            Look up an order.

            :param order_id: The order number.
            :param user_id: Comes from the agent state.
            :param state: The agent state.
            :param callback: A callback.
            :param _trace_id: Internal trace id.
            :param *args: Extra positional arguments.
            :param **kwargs: Extra keyword arguments.
            """
            return order_id

        tool_obj = create_tool_from_function(lookup_order, inputs_from_state={"user": "user_id"}, parse_docstring=True)

        assert tool_obj.parameters == {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order number."}},
            "required": ["order_id"],
        }
        assert "docstring documents" not in caplog.text

    def test_google_style_variadic_parameters_are_skipped_without_warning(self, caplog):
        def run(query: str, *args: Any, **kwargs: Any) -> str:
            """
            Run a query.

            Args:
                query: The query.
                *args: Extra positional arguments.
                **kwargs: Extra keyword arguments.
            """
            return query

        tool_obj = create_tool_from_function(run, parse_docstring=True)

        assert list(tool_obj.parameters["properties"]) == ["query"]
        assert "docstring documents" not in caplog.text

    def test_unknown_documented_parameter_logs_warning(self, caplog):
        def book(nights: int) -> str:
            """
            Book a hotel.

            :param night: Typo in the parameter name.
            """
            return str(nights)

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert "docstring documents a parameter 'night'" in caplog.text
        assert "description" not in tool_obj.parameters["properties"]["nights"]

    def test_parameter_without_description_gets_no_description(self):
        def book(city: str, nights: int) -> str:
            """
            Book a hotel.

            :param city:
            :param nights: Number of nights.
            """
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert "description" not in tool_obj.parameters["properties"]["city"]
        assert tool_obj.parameters["properties"]["nights"]["description"] == "Number of nights."

    def test_multiline_parameter_description(self):
        def book(city: str) -> str:
            """
            Book a hotel.

            :param city: The city where the hotel is,
                for example "Berlin".
            """
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.parameters["properties"]["city"]["description"] == (
            'The city where the hotel is,\nfor example "Berlin".'
        )

    def test_summary_without_blank_line_is_joined_with_newline(self):
        def book(city: str) -> str:
            """Book a hotel
            in the given city.
            """
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.description == "Book a hotel\nin the given city."

    def test_no_docstring(self):
        def book(city: str) -> str:
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.description == ""
        assert tool_obj.parameters["properties"] == {"city": {"type": "string"}}

    def test_docstring_with_only_parameters(self):
        def book(city: str) -> str:
            """
            :param city: The city.
            """
            return city

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.description == ""
        assert tool_obj.parameters["properties"]["city"]["description"] == "The city."

    def test_unparsable_docstring_falls_back_to_whole_docstring(self, monkeypatch, caplog):
        def book(city: str) -> str:
            """
            Book a hotel.

            :param city: The city.
            """
            return city

        def failing_parse(_docstring: str) -> None:
            raise ParseError("broken docstring")

        monkeypatch.setattr(sys.modules["haystack.tools.from_function"], "parse", failing_parse)

        tool_obj = create_tool_from_function(book, parse_docstring=True)

        assert tool_obj.description == "Book a hotel.\n\n:param city: The city."
        assert tool_obj.parameters["properties"] == {"city": {"type": "string"}}
        assert "could not parse the docstring" in caplog.text

    def test_tool_decorator_with_parse_docstring(self):
        assert weather_tool_with_parsed_docstring.description == "Get the current weather for a city."
        assert weather_tool_with_parsed_docstring.parameters["properties"] == {
            "city": {"type": "string", "description": "The city to get the weather for."},
            "unit": {"type": "string", "description": "The temperature unit.", "default": "celsius"},
        }
        assert (
            weather_tool_with_parsed_docstring.invoke(city="Berlin") == "Weather report for Berlin: 20 celsius, sunny"
        )

    def test_tool_decorator_with_parse_docstring_on_async_function(self):
        @tool(parse_docstring=True)
        async def get_weather(city: str) -> str:
            """
            Get the current weather.

            :param city: The city to get the weather for.
            """
            return city

        assert get_weather.async_function is not None
        assert get_weather.description == "Get the current weather."
        assert get_weather.parameters["properties"]["city"]["description"] == "The city to get the weather for."

    def test_serialization_round_trip_keeps_parsed_descriptions(self):
        deserialized = Tool.from_dict(weather_tool_with_parsed_docstring.to_dict())

        assert deserialized.description == weather_tool_with_parsed_docstring.description
        assert deserialized.parameters == weather_tool_with_parsed_docstring.parameters

    def test_parameters_schema_is_valid_json_schema(self):
        jsonschema.Draft202012Validator.check_schema(weather_tool_with_parsed_docstring.parameters)
        jsonschema.validate({"city": "Berlin"}, weather_tool_with_parsed_docstring.parameters)
