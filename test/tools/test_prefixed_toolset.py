# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated, Any

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import ComponentTool, PrefixedToolset, SearchableToolset, Tool, Toolset, tool

SEARCH_PARAMETERS = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


@dataclass
class ServerTool(Tool):
    """A Tool that, like MCPTool, sends its own name to a remote server when invoked."""

    server: str = ""

    def invoke(self, **kwargs: Any) -> Any:
        return f"{self.server}:{self.name}:{kwargs['query']}"

    async def invoke_async(self, **kwargs: Any) -> Any:
        return f"async-{self.server}:{self.name}:{kwargs['query']}"


def _noop(**kwargs: Any) -> None:
    return None


def _server_toolset(server: str) -> Toolset:
    return Toolset(
        [
            ServerTool(
                name="search",
                description=f"Search {server}",
                parameters=SEARCH_PARAMETERS,
                function=_noop,
                server=server,
            ),
            ServerTool(
                name="get_item",
                description=f"Get an item from {server}",
                parameters=SEARCH_PARAMETERS,
                function=_noop,
                server=server,
            ),
        ]
    )


@tool
def add(a: Annotated[int, "The first number"], b: Annotated[int, "The second number"]) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: Annotated[int, "The first number"], b: Annotated[int, "The second number"]) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def remember(note: Annotated[str, "The note to save"], state: State) -> str:
    """Save a note in the Agent state."""
    state.set("notes", [note])
    return "saved"


@tool
async def async_add(a: Annotated[int, "The first number"], b: Annotated[int, "The second number"]) -> int:
    """Add two numbers asynchronously."""
    return a + b


@component
class Echo:
    @component.output_types(reply=str)
    def run(self, text: str) -> dict[str, str]:
        return {"reply": text}


class LazyToolset(Toolset):
    """A Toolset that loads its tools during warm_up, like an MCPToolset that connects lazily."""

    def __init__(self) -> None:
        super().__init__(tools=[])
        self.warm_up_calls = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1
        if not self.tools:
            self.tools.append(add)


class LifecycleToolset(Toolset):
    """A Toolset that records the lifecycle calls it receives."""

    def __init__(self) -> None:
        super().__init__(tools=[add])
        self.calls: list[str] = []

    def warm_up(self) -> None:
        self.calls.append("warm_up")

    async def warm_up_async(self) -> None:
        self.calls.append("warm_up_async")

    def close(self) -> None:
        self.calls.append("close")

    async def close_async(self) -> None:
        self.calls.append("close_async")


@dataclass
class WarmUpTool(Tool):
    """A Tool that counts its warm-up calls."""

    warm_up_calls: int = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1


def _tool_results(messages: list[ChatMessage]) -> dict[str, Any]:
    return {result.origin.tool_name: result.result for message in messages for result in message.tool_call_results}


class TestPrefixedToolsetInit:
    def test_init(self):
        toolset = Toolset([add])
        prefixed = PrefixedToolset(toolset, prefix="math_")
        assert prefixed.toolset is toolset
        assert prefixed.prefix == "math_"
        assert prefixed.tools == []

    @pytest.mark.parametrize("prefix", ["math_", "math-", "Math2", "a"])
    def test_init_valid_prefixes(self, prefix):
        assert PrefixedToolset(Toolset([add]), prefix=prefix).prefix == prefix

    @pytest.mark.parametrize("prefix", ["", "math tools", "math.", "math/", "mäth_", "math:"])
    def test_init_invalid_prefix(self, prefix):
        with pytest.raises(ValueError, match="Invalid prefix"):
            PrefixedToolset(Toolset([add]), prefix=prefix)

    def test_init_non_string_prefix(self):
        with pytest.raises(ValueError, match="Invalid prefix"):
            PrefixedToolset(Toolset([add]), prefix=None)  # type: ignore[arg-type]

    def test_init_with_list_of_tools_raises(self):
        with pytest.raises(TypeError, match="Expected a Toolset, got list"):
            PrefixedToolset([add], prefix="math_")  # type: ignore[arg-type]

    def test_init_with_run_scoped_toolset_raises(self):
        with pytest.raises(TypeError, match="SearchableToolset has run-scoped state"):
            PrefixedToolset(SearchableToolset(catalog=[add]), prefix="math_")

    def test_init_with_prefixed_toolset_is_allowed(self):
        nested = PrefixedToolset(PrefixedToolset(Toolset([add]), prefix="inner_"), prefix="outer_")
        assert [t.name for t in nested] == ["outer_inner_add"]


class TestPrefixedToolsetTools:
    def test_iter_prefixes_names(self):
        prefixed = PrefixedToolset(Toolset([add, multiply]), prefix="math_")
        assert [t.name for t in prefixed] == ["math_add", "math_multiply"]

    def test_prefixed_tool_keeps_everything_else(self):
        prefixed_add = next(iter(PrefixedToolset(Toolset([add]), prefix="math_")))
        assert type(prefixed_add) is type(add)
        assert prefixed_add.description == add.description
        assert prefixed_add.parameters == add.parameters
        assert prefixed_add.function is add.function

    def test_wrapped_toolset_is_not_modified(self):
        toolset = Toolset([add, multiply])
        list(PrefixedToolset(toolset, prefix="math_"))
        assert [t.name for t in toolset] == ["add", "multiply"]
        assert toolset.tools == [add, multiply]

    def test_iter_returns_the_same_objects(self):
        prefixed = PrefixedToolset(Toolset([add]), prefix="math_")
        assert next(iter(prefixed)) is next(iter(prefixed))

    def test_cached_tool_is_not_reused_for_another_tool_with_the_same_id(self):
        prefixed = PrefixedToolset(Toolset([add]), prefix="math_")
        stale = Tool(name="stale", description="stale", parameters=SEARCH_PARAMETERS, function=_noop)
        # Simulate id reuse: a cache entry under add's id that belongs to another (garbage-collected) tool
        prefixed._prefixed_tools[id(add)] = (multiply, stale)
        assert next(iter(prefixed)).name == "math_add"

    def test_len_getitem_and_contains(self):
        prefixed = PrefixedToolset(Toolset([add, multiply]), prefix="math_")
        assert len(prefixed) == 2
        assert prefixed[1].name == "math_multiply"
        assert "math_add" in prefixed
        assert "add" not in prefixed
        assert prefixed[0] in prefixed

    def test_get_selectable_tools(self):
        prefixed = PrefixedToolset(Toolset([add, multiply]), prefix="math_")
        assert [t.name for t in prefixed.get_selectable_tools()] == ["math_add", "math_multiply"]

    def test_lazy_toolset_is_prefixed_after_warm_up(self):
        lazy = LazyToolset()
        prefixed = PrefixedToolset(lazy, prefix="math_")
        assert list(prefixed) == []
        prefixed.warm_up()
        assert [t.name for t in prefixed] == ["math_add"]
        assert lazy.warm_up_calls == 1

    def test_add(self):
        toolset = Toolset([add])
        prefixed = PrefixedToolset(toolset, prefix="math_")
        prefixed.add(multiply)
        assert [t.name for t in toolset] == ["add", "multiply"]
        assert [t.name for t in prefixed] == ["math_add", "math_multiply"]

    def test_add_duplicate_raises(self):
        prefixed = PrefixedToolset(Toolset([add]), prefix="math_")
        with pytest.raises(ValueError, match="Duplicate tool names"):
            prefixed.add(add)

    def test_add_non_tool_raises(self):
        prefixed = PrefixedToolset(Toolset([add]), prefix="math_")
        with pytest.raises(TypeError, match="Expected Tool"):
            prefixed.add("not a tool")  # type: ignore[arg-type]


class TestPrefixedToolsetInvocation:
    def test_invoke_uses_the_original_tool(self):
        prefixed_search = next(iter(PrefixedToolset(_server_toolset("github"), prefix="github_")))
        assert prefixed_search.name == "github_search"
        assert prefixed_search.invoke(query="bug") == "github:search:bug"

    @pytest.mark.asyncio
    async def test_invoke_async_uses_the_original_tool(self):
        prefixed_search = next(iter(PrefixedToolset(_server_toolset("github"), prefix="github_")))
        assert await prefixed_search.invoke_async(query="bug") == "async-github:search:bug"

    def test_invoke_function_tool(self):
        prefixed_add = next(iter(PrefixedToolset(Toolset([add]), prefix="math_")))
        assert prefixed_add.invoke(a=2, b=3) == 5

    @pytest.mark.asyncio
    async def test_invoke_async_function_tool(self):
        prefixed = PrefixedToolset(Toolset([add, async_add]), prefix="math_")
        prefixed_add, prefixed_async_add = list(prefixed)
        assert await prefixed_add.invoke_async(a=2, b=3) == 5
        assert await prefixed_async_add.invoke_async(a=2, b=3) == 5

    def test_nested_prefixed_toolsets_invoke_the_original_tool(self):
        nested = PrefixedToolset(PrefixedToolset(_server_toolset("github"), prefix="gh_"), prefix="team_")
        nested_search = next(iter(nested))
        assert nested_search.name == "team_gh_search"
        assert nested_search.invoke(query="bug") == "github:search:bug"


class TestPrefixedToolsetLifecycle:
    def test_warm_up_and_close_are_delegated(self):
        inner = LifecycleToolset()
        prefixed = PrefixedToolset(inner, prefix="math_")
        prefixed.warm_up()
        prefixed.close()
        assert inner.calls == ["warm_up", "close"]

    @pytest.mark.asyncio
    async def test_async_warm_up_and_close_are_delegated(self):
        inner = LifecycleToolset()
        prefixed = PrefixedToolset(inner, prefix="math_")
        await prefixed.warm_up_async()
        await prefixed.close_async()
        assert inner.calls == ["warm_up_async", "close_async"]

    def test_warm_up_plain_toolset_warms_up_its_tools(self):
        warm_tool = WarmUpTool(name="warm", description="warm", parameters=SEARCH_PARAMETERS, function=_noop)
        PrefixedToolset(Toolset([warm_tool]), prefix="x_").warm_up()
        assert warm_tool.warm_up_calls == 1

    @pytest.mark.asyncio
    async def test_warm_up_async_plain_toolset_warms_up_its_tools(self):
        warm_tool = WarmUpTool(name="warm", description="warm", parameters=SEARCH_PARAMETERS, function=_noop)
        await PrefixedToolset(Toolset([warm_tool]), prefix="x_").warm_up_async()
        assert warm_tool.warm_up_calls == 1


class TestPrefixedToolsetEqualityAndRepr:
    def test_eq(self):
        toolset = Toolset([add])
        assert PrefixedToolset(toolset, prefix="math_") == PrefixedToolset(toolset, prefix="math_")
        assert PrefixedToolset(toolset, prefix="math_") != PrefixedToolset(toolset, prefix="calc_")
        assert PrefixedToolset(toolset, prefix="math_") != PrefixedToolset(Toolset([multiply]), prefix="math_")
        assert PrefixedToolset(toolset, prefix="math_") != toolset

    def test_repr(self):
        prefixed = PrefixedToolset(Toolset([add]), prefix="math_")
        assert repr(prefixed).startswith("PrefixedToolset(prefix='math_', toolset=Toolset(")


class TestPrefixedToolsetSerialization:
    def test_to_dict(self):
        toolset = Toolset([add])
        assert PrefixedToolset(toolset, prefix="math_").to_dict() == {
            "type": "haystack.tools.prefixed_toolset.PrefixedToolset",
            "data": {"toolset": toolset.to_dict(), "prefix": "math_"},
        }

    def test_from_dict(self):
        data = PrefixedToolset(Toolset([add, multiply]), prefix="math_").to_dict()
        restored = PrefixedToolset.from_dict(data)
        assert isinstance(restored, PrefixedToolset)
        assert restored.prefix == "math_"
        assert [t.name for t in restored] == ["math_add", "math_multiply"]
        assert restored[0].invoke(a=2, b=3) == 5

    def test_from_dict_does_not_modify_input(self):
        data = PrefixedToolset(Toolset([add]), prefix="math_").to_dict()
        PrefixedToolset.from_dict(data)
        assert isinstance(data["data"]["toolset"], dict)

    def test_from_dict_nested(self):
        nested = PrefixedToolset(PrefixedToolset(Toolset([add]), prefix="inner_"), prefix="outer_")
        restored = PrefixedToolset.from_dict(nested.to_dict())
        assert restored == nested
        assert [t.name for t in restored] == ["outer_inner_add"]

    def test_from_dict_with_non_toolset_raises(self):
        data = {
            "type": "haystack.tools.prefixed_toolset.PrefixedToolset",
            "data": {"toolset": {"type": generate_qualified_class_name(Echo), "data": {}}, "prefix": "math_"},
        }
        with pytest.raises(TypeError, match="is not a subclass of Toolset"):
            PrefixedToolset.from_dict(data)


class TestPrefixedToolsetWithAgent:
    def test_agent_calls_tools_with_the_same_name_from_two_toolsets(self):
        generator = MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(
                    tool_calls=[
                        ToolCall(tool_name="github_search", arguments={"query": "bug"}),
                        ToolCall(tool_name="jira_search", arguments={"query": "bug"}),
                    ]
                ),
                "Done.",
            ]
        )
        agent = Agent(
            chat_generator=generator,
            tools=[
                PrefixedToolset(_server_toolset("github"), prefix="github_"),
                PrefixedToolset(_server_toolset("jira"), prefix="jira_"),
            ],
        )
        result = agent.run(messages=[ChatMessage.from_user("Find the bug")])
        assert _tool_results(result["messages"]) == {
            "github_search": "github:search:bug",
            "jira_search": "jira:search:bug",
        }
        assert result["last_message"].text == "Done."

    @pytest.mark.asyncio
    async def test_agent_run_async(self):
        generator = MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="github_search", arguments={"query": "x"})]),
                "Done.",
            ]
        )
        agent = Agent(chat_generator=generator, tools=PrefixedToolset(_server_toolset("github"), prefix="github_"))
        result = await agent.run_async(messages=[ChatMessage.from_user("Find it")])
        assert _tool_results(result["messages"]) == {"github_search": "async-github:search:x"}

    def test_agent_tool_selection_by_prefixed_name(self):
        def list_tools(messages: list[ChatMessage], tools: Any) -> str:
            return ",".join(sorted(t.name for t in tools))

        agent = Agent(
            chat_generator=MockChatGenerator(response_fn=list_tools),
            tools=[
                PrefixedToolset(_server_toolset("github"), prefix="github_"),
                PrefixedToolset(_server_toolset("jira"), prefix="jira_"),
            ],
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")], tools=["jira_get_item"])
        assert result["last_message"].text == "jira_get_item"

    def test_agent_with_state_tool_and_component_tool(self):
        generator = MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(
                    tool_calls=[
                        ToolCall(tool_name="mem_remember", arguments={"note": "buy milk"}),
                        ToolCall(tool_name="mem_echo", arguments={"text": "hello"}),
                    ]
                ),
                "Done.",
            ]
        )
        toolset = PrefixedToolset(Toolset([remember, ComponentTool(component=Echo(), name="echo")]), prefix="mem_")
        agent = Agent(chat_generator=generator, tools=toolset, state_schema={"notes": {"type": list[str]}})
        result = agent.run(messages=[ChatMessage.from_user("Remember to buy milk")])
        assert result["notes"] == ["buy milk"]
        assert _tool_results(result["messages"]) == {"mem_remember": "saved", "mem_echo": '{"reply": "hello"}'}

    def test_state_parameter_is_hidden_from_the_llm(self):
        prefixed_remember = next(iter(PrefixedToolset(Toolset([remember]), prefix="mem_")))
        assert list(prefixed_remember.parameters["properties"]) == ["note"]

    def test_agent_serialization_round_trip(self):
        agent = Agent(
            chat_generator=MockChatGenerator(responses="Done."), tools=[PrefixedToolset(Toolset([add]), prefix="math_")]
        )
        restored = Agent.from_dict(agent.to_dict())
        assert isinstance(restored.tools, list)
        assert restored.tools[0] == PrefixedToolset(Toolset([add]), prefix="math_")
        