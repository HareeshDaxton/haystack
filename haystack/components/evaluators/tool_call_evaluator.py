# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from enum import Enum
from typing import Any

from haystack import component, default_to_dict
from haystack.dataclasses import ChatMessage, ToolCall


class ToolCallMatchMode(Enum):
    """
    How the predicted tool calls of a sample must relate to the expected ones for the sample to pass.
    """

    # Same calls, in the same order.
    STRICT = "strict"
    # Same calls, in any order.
    UNORDERED = "unordered"
    # Every expected call was made. Extra calls are allowed.
    SUPERSET = "superset"
    # No call outside the expected ones was made. Missing calls are allowed.
    SUBSET = "subset"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "ToolCallMatchMode":
        """
        Convert a string to a ToolCallMatchMode enum.
        """
        enum_map = {e.value: e for e in ToolCallMatchMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown match mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


class ToolCallArgumentsMatch(Enum):
    """
    How the arguments of a predicted tool call are compared with the arguments of an expected tool call.
    """

    # Same arguments, nothing more and nothing less.
    EXACT = "exact"
    # Every expected argument is present with the same value. Extra arguments are allowed.
    SUBSET = "subset"
    # Arguments are not compared, only the tool name.
    IGNORE = "ignore"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "ToolCallArgumentsMatch":
        """
        Convert a string to a ToolCallArgumentsMatch enum.
        """
        enum_map = {e.value: e for e in ToolCallArgumentsMatch}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown arguments match '{string}'. Supported values are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


def _values_equal(expected: Any, predicted: Any) -> bool:
    """
    Compare two tool call argument values.

    Works like `==`, except that booleans only equal booleans. In Python `True == 1`, but for a tool call
    `{"verbose": 1}` is not the same as `{"verbose": True}`. Dictionaries and lists are compared recursively.
    Integers and floats with the same value are equal, as in JSON.
    """
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return isinstance(expected, bool) and isinstance(predicted, bool) and expected == predicted
    if isinstance(expected, Mapping) and isinstance(predicted, Mapping):
        return expected.keys() == predicted.keys() and all(_values_equal(v, predicted[k]) for k, v in expected.items())
    if isinstance(expected, (list, tuple)) and isinstance(predicted, (list, tuple)):
        return len(expected) == len(predicted) and all(map(_values_equal, expected, predicted))
    return expected == predicted


@component
class ToolCallEvaluator:
    """
    Evaluator that checks whether an Agent called the expected tools with the expected arguments.

    For each sample, it compares the tool calls found in the predicted messages, for example the `messages` output
    of an `Agent`, with a list of expected tool calls. A sample scores 1.0 if the calls match according to
    `match_mode` and `arguments_match`, and 0.0 otherwise. The evaluator is deterministic and does not use an LLM.

    An empty list of expected tool calls means that no tool should be called for that sample.

    Usage example:
    ```python
    from haystack.components.evaluators import ToolCallEvaluator
    from haystack.dataclasses import ChatMessage, ToolCall

    evaluator = ToolCallEvaluator(match_mode="unordered", arguments_match="exact")
    result = evaluator.run(
        ground_truth_tool_calls=[
            [ToolCall(tool_name="get_weather", arguments={"city": "Paris"})],
            [],
        ],
        predicted_messages=[
            [
                ChatMessage.from_user("What's the weather in Paris?"),
                ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Paris"})]),
            ],
            [ChatMessage.from_user("Hi!"), ChatMessage.from_assistant("Hello! How can I help?")],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 1.0]
    print(result["score"])
    # 1.0
    ```
    """

    def __init__(
        self,
        match_mode: str | ToolCallMatchMode = ToolCallMatchMode.UNORDERED,
        arguments_match: str | ToolCallArgumentsMatch = ToolCallArgumentsMatch.EXACT,
    ) -> None:
        """
        Create a ToolCallEvaluator component.

        :param match_mode:
            How the predicted tool calls of a sample must relate to the expected ones for the sample to pass:
            - `"strict"`: the same calls in the same order.
            - `"unordered"`: the same calls in any order. Use it when the model can run tools in parallel.
            - `"superset"`: every expected call was made. Extra calls are allowed.
            - `"subset"`: no call outside the expected ones was made. Missing calls are allowed.
            To check that an Agent did not make unwanted calls, use `"unordered"` or `"subset"`.
        :param arguments_match:
            How the arguments of two tool calls with the same tool name are compared:
            - `"exact"`: the arguments must be equal.
            - `"subset"`: every expected argument must be present with an equal value. Extra arguments are allowed,
              for example optional arguments the model filled in.
            - `"ignore"`: only the tool name is compared.
            Booleans only equal booleans, so `1` does not match `True`. Integers and floats with the same value match.
        :raises ValueError: If `match_mode` or `arguments_match` is not a supported value.
        """
        if isinstance(match_mode, str):
            match_mode = ToolCallMatchMode.from_str(match_mode)
        if isinstance(arguments_match, str):
            arguments_match = ToolCallArgumentsMatch.from_str(arguments_match)

        self.match_mode = match_mode
        self.arguments_match = arguments_match

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self, ground_truth_tool_calls: list[list[ToolCall]], predicted_messages: list[list[ChatMessage]]
    ) -> dict[str, Any]:
        """
        Run the ToolCallEvaluator on the given inputs.

        `ground_truth_tool_calls` and `predicted_messages` must have the same length.

        :param ground_truth_tool_calls:
            A list of expected tool calls for each sample. Use an empty list for a sample where no tool should be
            called.
        :param predicted_messages:
            A list of messages for each sample, for example the `messages` output of an `Agent`. The tool calls are
            read from the messages in order. Messages without tool calls are ignored.
        :returns:
            A dictionary with the following outputs:
            - `score` - The proportion of samples whose tool calls matched.
            - `individual_scores` - A list of 1.0 and 0.0, one per sample, where 1.0 means the tool calls matched.
        :raises ValueError: If the inputs are empty or don't have the same length.
        """
        if len(ground_truth_tool_calls) == 0 or len(predicted_messages) == 0:
            msg = "ground_truth_tool_calls and predicted_messages must be provided."
            raise ValueError(msg)

        if len(ground_truth_tool_calls) != len(predicted_messages):
            msg = "The length of ground_truth_tool_calls and predicted_messages must be the same."
            raise ValueError(msg)

        scores = []
        for expected_calls, messages in zip(ground_truth_tool_calls, predicted_messages, strict=True):
            predicted_calls = [tool_call for message in messages for tool_call in message.tool_calls]
            scores.append(1.0 if self._sample_matches(expected_calls, predicted_calls) else 0.0)

        return {"score": sum(scores) / len(scores), "individual_scores": scores}

    def _call_matches(self, expected: ToolCall, predicted: ToolCall) -> bool:
        """
        Check whether one predicted tool call matches one expected tool call.
        """
        if expected.tool_name != predicted.tool_name:
            return False
        if self.arguments_match == ToolCallArgumentsMatch.IGNORE:
            return True
        if self.arguments_match == ToolCallArgumentsMatch.EXACT:
            return _values_equal(expected.arguments, predicted.arguments)
        # SUBSET: every expected argument is present in the predicted call with an equal value
        return all(
            name in predicted.arguments and _values_equal(value, predicted.arguments[name])
            for name, value in expected.arguments.items()
        )

    def _sample_matches(self, expected_calls: list[ToolCall], predicted_calls: list[ToolCall]) -> bool:
        """
        Check whether the predicted tool calls of one sample match the expected ones according to `match_mode`.
        """
        if self.match_mode == ToolCallMatchMode.STRICT:
            return len(expected_calls) == len(predicted_calls) and all(
                self._call_matches(expected, predicted)
                for expected, predicted in zip(expected_calls, predicted_calls, strict=True)
            )

        matched = self._max_matching(expected_calls, predicted_calls)
        if self.match_mode == ToolCallMatchMode.UNORDERED:
            return matched == len(expected_calls) == len(predicted_calls)
        if self.match_mode == ToolCallMatchMode.SUPERSET:
            return matched == len(expected_calls)
        # SUBSET
        return matched == len(predicted_calls)

    def _max_matching(self, expected_calls: list[ToolCall], predicted_calls: list[ToolCall]) -> int:
        """
        Return the size of the largest one-to-one pairing of expected and predicted tool calls.

        Each expected call can be paired with at most one predicted call that matches it, and the other way around.
        A greedy pairing is not enough: with `arguments_match="subset"`, an expected call with few arguments could take
        the only predicted call that a more specific expected call needs. This uses the augmenting path algorithm
        for bipartite matching (Kuhn's algorithm), which is fast for the number of tool calls an Agent makes.
        """
        candidates = [
            [p for p, predicted in enumerate(predicted_calls) if self._call_matches(expected, predicted)]
            for expected in expected_calls
        ]
        # owner[p] is the index of the expected call paired with the predicted call p, or None if p is free
        owner: list[int | None] = [None] * len(predicted_calls)

        def _assign(e: int, visited: set[int]) -> bool:
            for p in candidates[e]:
                if p in visited:
                    continue
                visited.add(p)
                current = owner[p]
                if current is None or _assign(current, visited):
                    owner[p] = e
                    return True
            return False

        return sum(_assign(e, set()) for e in range(len(expected_calls)))

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, match_mode=str(self.match_mode), arguments_match=str(self.arguments_match))
