# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import re
from collections.abc import Iterator
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset
from haystack.tools.utils import close_tools, close_tools_async, warm_up_tools, warm_up_tools_async

# Characters that all major LLM providers accept in tool names (OpenAI, Anthropic, Google, ...).
_VALID_PREFIX_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class PrefixedToolset(Toolset):
    """
    A Toolset that exposes every tool of another Toolset with a prefix added to its name.

    Use it to combine toolsets whose tools share names, such as two MCP servers that both have a `search` tool,
    or to make tool names clearer for the Language Model (`github_search` instead of `search`).

    Only the name the Language Model sees changes. Calls are delegated to the original tool, so tools that use their
    own name internally, like `MCPTool`, keep working. The wrapped Toolset is never modified.

    The prefix is added when the tools are listed, so toolsets that load their tools lazily, like an `MCPToolset`
    that connects during `warm_up()`, are supported.

    ### Usage example

    ```python
    from typing import Annotated

    from haystack.tools import PrefixedToolset, Toolset, tool

    @tool
    def search(query: Annotated[str, "The search query"]) -> str:
        '''Search the issue tracker.'''
        return f"Results for {query}"

    toolset = PrefixedToolset(Toolset([search]), prefix="github_")

    print([t.name for t in toolset])
    # ['github_search']
    print(toolset[0].invoke(query="login bug"))
    # Results for login bug
    ```
    """

    def __init__(self, toolset: Toolset, prefix: str) -> None:
        """
        Initialize the PrefixedToolset.

        :param toolset: The Toolset whose tools are exposed with the prefix.
        :param prefix: The prefix added to every tool name, for example `"github_"`. It is used as is, so include a
            separator like `_` if you want one. It may only contain letters, digits, `_` and `-`, the characters that
            LLM providers accept in tool names.
        :raises TypeError: If `toolset` is not a Toolset, or if it is a Toolset with run-scoped state that overrides
            `spawn()`, such as `SearchableToolset`.
        :raises ValueError: If `prefix` is empty or contains characters other than letters, digits, `_` and `-`.
        """
        if not isinstance(toolset, Toolset):
            raise TypeError(f"Expected a Toolset, got {type(toolset).__name__}. Wrap Tools in a Toolset first.")
        if type(toolset).spawn is not Toolset.spawn:
            raise TypeError(
                f"{type(toolset).__name__} has run-scoped state (it overrides `spawn()`), so it can't be wrapped in a "
                "PrefixedToolset. Wrap the toolsets it contains instead."
            )
        if not isinstance(prefix, str) or not _VALID_PREFIX_PATTERN.match(prefix):
            raise ValueError(
                f"Invalid prefix {prefix!r}. The prefix must be a non-empty string containing only letters, digits, "
                "'_' and '-'."
            )

        self._toolset = toolset
        self._prefix = prefix
        # Maps id(original tool) -> (original tool, prefixed tool), so iterating twice returns the same objects.
        # The original is kept to guard against id reuse after the original tool is garbage-collected.
        self._prefixed_tools: dict[int, tuple[Tool, Tool]] = {}

        # Initialize parent with an empty tools list: the tools are read from the wrapped toolset on demand
        super().__init__(tools=[])

    @property
    def toolset(self) -> Toolset:
        """The wrapped Toolset."""
        return self._toolset

    @property
    def prefix(self) -> str:
        """The prefix added to every tool name."""
        return self._prefix

    def _prefixed(self, original: Tool) -> Tool:
        """
        Return the prefixed copy of `original`, creating it on first use.

        The copy has the same class as the original, so the Agent handles it the same way (for example, `State`
        injection and `ComponentTool` inputs). Its `invoke` and `invoke_async` are bound to the original tool, so the
        original keeps its own name when it runs.

        :param original: A tool of the wrapped toolset.
        :returns: The copy of the tool with the prefixed name.
        """
        cached = self._prefixed_tools.get(id(original))
        if cached is not None and cached[0] is original:
            return cached[1]

        prefixed = copy.copy(original)
        prefixed.name = f"{self._prefix}{original.name}"
        prefixed.invoke = original.invoke  # type: ignore[method-assign]
        prefixed.invoke_async = original.invoke_async  # type: ignore[method-assign]
        self._prefixed_tools[id(original)] = (original, prefixed)
        return prefixed

    def __iter__(self) -> Iterator[Tool]:
        """
        Return an iterator over the tools of the wrapped toolset, with prefixed names.

        :returns: An iterator yielding the prefixed tools.
        """
        return iter([self._prefixed(tool) for tool in self._toolset])

    def get_selectable_tools(self) -> list[Tool]:
        """
        Return the selectable tools of the wrapped toolset, with prefixed names.

        Select them by their prefixed names, for example `Agent.run(tools=["github_search"])`.

        :returns: The prefixed tools available for name-based selection.
        """
        return [self._prefixed(tool) for tool in self._toolset.get_selectable_tools()]

    def add(self, tool: Tool) -> None:
        """
        Add a Tool to the wrapped toolset. It is exposed with the prefix, like the other tools.

        :param tool: The Tool to add, with its original (unprefixed) name.
        :raises ValueError: If the wrapped toolset already has a tool with the same name.
        :raises TypeError: If the provided object is not a Tool.
        """
        self._toolset.add(tool)

    def warm_up(self) -> None:
        """Warm up the wrapped toolset."""
        warm_up_tools(self._toolset)

    async def warm_up_async(self) -> None:
        """Warm up the wrapped toolset asynchronously."""
        await warm_up_tools_async(self._toolset)

    def close(self) -> None:
        """Close the wrapped toolset."""
        close_tools(self._toolset)

    async def close_async(self) -> None:
        """Close the wrapped toolset asynchronously."""
        await close_tools_async(self._toolset)

    # Like Toolset, a PrefixedToolset is mutable (see `add()`), so it is compared by value and is not hashable
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """
        Check if another object is a PrefixedToolset with the same prefix and wrapped toolset.

        :param other: The object to compare with.
        :returns: True if both are equal, False otherwise.
        """
        if not isinstance(other, PrefixedToolset):
            return NotImplemented
        return self._prefix == other._prefix and self._toolset == other._toolset

    def __repr__(self) -> str:
        """
        Return a string representation of the PrefixedToolset.

        :returns: The string representation.
        """
        return f"{type(self).__name__}(prefix={self._prefix!r}, toolset={self._toolset!r})"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the PrefixedToolset to a dictionary.

        The wrapped toolset is serialized with its own `to_dict()`, so dynamic toolsets like `MCPToolset` keep
        serializing their connection settings instead of a fixed list of tools.

        :returns: A dictionary representation of the PrefixedToolset.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"toolset": self._toolset.to_dict(), "prefix": self._prefix},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefixedToolset":
        """
        Deserialize a PrefixedToolset from a dictionary.

        :param data: Dictionary representation of the PrefixedToolset.
        :returns: A new PrefixedToolset instance.
        :raises TypeError: If the serialized wrapped toolset is not a subclass of Toolset.
        """
        inner_data = dict(data["data"])
        deserialize_tools_or_toolset_inplace(inner_data, key="toolset")
        return cls(toolset=inner_data["toolset"], prefix=inner_data["prefix"])
    