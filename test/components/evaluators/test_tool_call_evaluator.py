# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import Pipeline, default_from_dict
from haystack.components.agents import Agent
from haystack.components.evaluators import ToolCallEvaluator
from haystack.components.evaluators.tool_call_evaluator import ToolCallArgumentsMatch, ToolCallMatchMode, _values_equal
from haystack.components.generators.chat import MockChatGenerator
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool


def call(tool_name: str, **arguments: Any) -> ToolCall:
    return ToolCall(tool_name=tool_name, arguments=arguments)


def assistant(*tool_calls: ToolCall) -> ChatMessage:
    return ChatMessage.from_assistant(tool_calls=list(tool_calls))


WEATHER = call("get_weather", city="Paris")
FLIGHTS = call("search_flights", origin="Berlin", destination="Paris")
DELETE = call("delete_booking", booking_id="42")


def score_one(evaluator: ToolCallEvaluator, expected: list[ToolCall], messages: list[ChatMessage]) -> float:
    return evaluator.run(ground_truth_tool_calls=[expected], predicted_messages=[messages])["individual_scores"][0]


class TestInit:
    def test_defaults(self):
        evaluator = ToolCallEvaluator()
        assert evaluator.match_mode == ToolCallMatchMode.UNORDERED
        assert evaluator.arguments_match == ToolCallArgumentsMatch.EXACT

    def test_strings_are_converted_to_enums(self):
        evaluator = ToolCallEvaluator(match_mode="strict", arguments_match="subset")
        assert evaluator.match_mode == ToolCallMatchMode.STRICT
        assert evaluator.arguments_match == ToolCallArgumentsMatch.SUBSET

    def test_enums_are_accepted(self):
        evaluator = ToolCallEvaluator(
            match_mode=ToolCallMatchMode.SUPERSET, arguments_match=ToolCallArgumentsMatch.IGNORE
        )
        assert evaluator.match_mode == ToolCallMatchMode.SUPERSET
        assert evaluator.arguments_match == ToolCallArgumentsMatch.IGNORE

    def test_invalid_match_mode(self):
        with pytest.raises(ValueError, match="Unknown match mode 'any'"):
            ToolCallEvaluator(match_mode="any")

    def test_invalid_arguments_match(self):
        with pytest.raises(ValueError, match="Unknown arguments match 'fuzzy'"):
            ToolCallEvaluator(arguments_match="fuzzy")

    def test_enum_str(self):
        assert str(ToolCallMatchMode.SUBSET) == "subset"
        assert str(ToolCallArgumentsMatch.IGNORE) == "ignore"


class TestRunValidation:
    def test_empty_inputs(self):
        with pytest.raises(ValueError, match="must be provided"):
            ToolCallEvaluator().run(ground_truth_tool_calls=[], predicted_messages=[])

    def test_empty_predicted_messages(self):
        with pytest.raises(ValueError, match="must be provided"):
            ToolCallEvaluator().run(ground_truth_tool_calls=[[WEATHER]], predicted_messages=[])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="must be the same"):
            ToolCallEvaluator().run(ground_truth_tool_calls=[[WEATHER], [FLIGHTS]], predicted_messages=[[]])


class TestMatchModes:
    """Each row: expected calls, predicted messages, and the result for strict / unordered / superset / subset."""

    CASES = [
        pytest.param([WEATHER, FLIGHTS], [assistant(WEATHER), assistant(FLIGHTS)], (1, 1, 1, 1), id="same-order"),
        pytest.param([WEATHER, FLIGHTS], [assistant(FLIGHTS, WEATHER)], (0, 1, 1, 1), id="other-order"),
        pytest.param([WEATHER, FLIGHTS], [assistant(WEATHER)], (0, 0, 0, 1), id="missing-call"),
        pytest.param([WEATHER, FLIGHTS], [assistant(WEATHER, FLIGHTS, DELETE)], (0, 0, 1, 0), id="extra-call"),
        pytest.param([WEATHER], [assistant(FLIGHTS)], (0, 0, 0, 0), id="wrong-tool"),
        pytest.param([WEATHER], [assistant(WEATHER, WEATHER)], (0, 0, 1, 0), id="called-twice"),
        pytest.param([WEATHER, WEATHER], [assistant(WEATHER)], (0, 0, 0, 1), id="expected-twice-called-once"),
        pytest.param([], [ChatMessage.from_assistant("Hello!")], (1, 1, 1, 1), id="no-call-expected-none-made"),
        pytest.param([], [assistant(WEATHER)], (0, 0, 1, 0), id="no-call-expected-one-made"),
        pytest.param([WEATHER], [ChatMessage.from_assistant("I don't know.")], (0, 0, 0, 1), id="call-expected-none"),
    ]

    @pytest.mark.parametrize("expected, messages, results", CASES)
    def test_modes(self, expected, messages, results):
        for mode, result in zip(("strict", "unordered", "superset", "subset"), results, strict=True):
            assert score_one(ToolCallEvaluator(match_mode=mode), expected, messages) == float(result), mode


class TestArgumentsMatch:
    def test_exact_needs_equal_arguments(self):
        evaluator = ToolCallEvaluator(arguments_match="exact")
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Paris"))]) == 1.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Paris", unit="celsius"))]) == 0.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Rome"))]) == 0.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather"))]) == 0.0

    def test_subset_allows_extra_arguments(self):
        evaluator = ToolCallEvaluator(arguments_match="subset")
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Paris", unit="celsius"))]) == 1.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Rome", unit="celsius"))]) == 0.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", unit="celsius"))]) == 0.0

    def test_subset_with_no_expected_arguments_matches_any_arguments(self):
        evaluator = ToolCallEvaluator(arguments_match="subset")
        assert score_one(evaluator, [call("get_weather")], [assistant(WEATHER)]) == 1.0

    def test_ignore_compares_tool_name_only(self):
        evaluator = ToolCallEvaluator(arguments_match="ignore")
        assert score_one(evaluator, [WEATHER], [assistant(call("get_weather", city="Rome"))]) == 1.0
        assert score_one(evaluator, [WEATHER], [assistant(call("get_time", city="Paris"))]) == 0.0

    def test_strict_mode_uses_arguments_match(self):
        evaluator = ToolCallEvaluator(match_mode="strict", arguments_match="subset")
        predicted = [assistant(call("get_weather", city="Paris", unit="celsius"), FLIGHTS)]
        assert score_one(evaluator, [WEATHER, FLIGHTS], predicted) == 1.0

    def test_nested_arguments(self):
        expected = call("book", trip={"from": "Berlin", "to": "Paris", "stops": ["Cologne"]})
        evaluator = ToolCallEvaluator()
        same = call("book", trip={"to": "Paris", "from": "Berlin", "stops": ["Cologne"]})
        other = call("book", trip={"from": "Berlin", "to": "Paris", "stops": ["Brussels"]})
        assert score_one(evaluator, [expected], [assistant(same)]) == 1.0
        assert score_one(evaluator, [expected], [assistant(other)]) == 0.0

    def test_bool_does_not_match_int(self):
        evaluator = ToolCallEvaluator()
        assert score_one(evaluator, [call("search", verbose=True)], [assistant(call("search", verbose=1))]) == 0.0
        assert score_one(evaluator, [call("search", limit=1)], [assistant(call("search", limit=True))]) == 0.0
        assert score_one(evaluator, [call("search", verbose=True)], [assistant(call("search", verbose=True))]) == 1.0

    def test_int_matches_float_with_same_value(self):
        evaluator = ToolCallEvaluator()
        assert score_one(evaluator, [call("search", limit=5)], [assistant(call("search", limit=5.0))]) == 1.0


class TestValuesEqual:
    @pytest.mark.parametrize(
        "expected, predicted, result",
        [
            ("Paris", "Paris", True),
            ("Paris", "paris", False),
            (1, 1.0, True),
            (True, True, True),
            (True, 1, False),
            (0, False, False),
            (None, None, True),
            (None, 0, False),
            ([1, 2], [1, 2], True),
            ([1, 2], [2, 1], False),
            ([1, 2], [1, 2, 3], False),
            ([1, 2], (1, 2), True),
            ([True], [1], False),
            ({"a": 1}, {"a": 1}, True),
            ({"a": 1}, {"a": 1, "b": 2}, False),
            ({"a": {"b": True}}, {"a": {"b": 1}}, False),
            ({"a": 1}, [("a", 1)], False),
        ],
    )
    def test_values_equal(self, expected, predicted, result):
        assert _values_equal(expected, predicted) is result


class TestMatching:
    def test_greedy_pairing_would_fail_but_matching_succeeds(self):
        # A greedy pairing gives the first predicted call to the first expected call, which leaves nothing for the
        # second expected call. The one-to-one matching finds the valid pairing.
        expected = [call("get_weather", city="Paris"), call("get_weather", city="Paris", unit="fahrenheit")]
        predicted = [
            assistant(
                call("get_weather", city="Paris", unit="fahrenheit"), call("get_weather", city="Paris", unit="celsius")
            )
        ]
        for mode in ("unordered", "superset", "subset"):
            evaluator = ToolCallEvaluator(match_mode=mode, arguments_match="subset")
            assert score_one(evaluator, expected, predicted) == 1.0, mode

    def test_one_predicted_call_cannot_serve_two_expected_calls(self):
        evaluator = ToolCallEvaluator(match_mode="superset", arguments_match="ignore")
        assert score_one(evaluator, [WEATHER, call("get_weather", city="Rome")], [assistant(WEATHER)]) == 0.0

    def test_max_matching_size(self):
        evaluator = ToolCallEvaluator(arguments_match="subset")
        expected = [call("a"), call("a", x=1), call("b")]
        predicted = [call("a", x=1), call("a", x=2), call("c")]
        assert evaluator._max_matching(expected, predicted) == 2

    def test_many_calls(self):
        expected = [call("step", index=i) for i in range(200)]
        predicted = [assistant(*reversed(expected))]
        assert score_one(ToolCallEvaluator(), expected, predicted) == 1.0
        assert score_one(ToolCallEvaluator(match_mode="strict"), expected, predicted) == 0.0


class TestRun:
    def test_tool_calls_are_collected_from_all_messages(self):
        messages = [
            ChatMessage.from_system("You are helpful."),
            ChatMessage.from_user("Weather and flights?"),
            assistant(WEATHER),
            ChatMessage.from_tool(tool_result="18 degrees", origin=WEATHER),
            assistant(FLIGHTS),
            ChatMessage.from_tool(tool_result="3 flights", origin=FLIGHTS),
            ChatMessage.from_assistant("Here you go."),
        ]
        assert score_one(ToolCallEvaluator(match_mode="strict"), [WEATHER, FLIGHTS], messages) == 1.0

    def test_empty_messages_list_means_no_calls(self):
        assert score_one(ToolCallEvaluator(), [], []) == 1.0
        assert score_one(ToolCallEvaluator(), [WEATHER], []) == 0.0

    def test_batch_score_is_the_mean(self):
        result = ToolCallEvaluator().run(
            ground_truth_tool_calls=[[WEATHER], [FLIGHTS], [WEATHER], []],
            predicted_messages=[[assistant(WEATHER)], [assistant(WEATHER)], [assistant(WEATHER)], []],
        )
        assert result == {"score": 0.75, "individual_scores": [1.0, 0.0, 1.0, 1.0]}

    def test_inputs_are_not_modified(self):
        expected = [[WEATHER, FLIGHTS]]
        messages = [[assistant(FLIGHTS, WEATHER)]]
        ToolCallEvaluator().run(ground_truth_tool_calls=expected, predicted_messages=messages)
        assert expected == [[WEATHER, FLIGHTS]]
        assert messages[0][0].tool_calls == [FLIGHTS, WEATHER]

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("evaluator", ToolCallEvaluator(match_mode="superset", arguments_match="subset"))
        result = pipeline.run(
            {
                "evaluator": {
                    "ground_truth_tool_calls": [[WEATHER], [FLIGHTS]],
                    "predicted_messages": [[assistant(WEATHER, DELETE)], [assistant(WEATHER)]],
                }
            }
        )
        assert result == {"evaluator": {"score": 0.5, "individual_scores": [1.0, 0.0]}}


class TestWithAgent:
    @staticmethod
    def _agent(*steps: list[ToolCall]) -> Agent:
        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            """Get the weather for a city."""
            return f"18 degrees {unit} in {city}"

        @tool
        def search_flights(origin: str, destination: str) -> str:
            """Search flights between two cities."""
            return f"3 flights from {origin} to {destination}"

        replies: list[str | ChatMessage] = [assistant(*step) for step in steps]
        replies.append("Done.")
        return Agent(chat_generator=MockChatGenerator(responses=replies), tools=[get_weather, search_flights])

    def test_scores_the_messages_of_a_real_agent_run(self):
        question = [ChatMessage.from_user("Weather in Paris and flights from Berlin to Paris?")]
        good = self._agent([WEATHER], [FLIGHTS]).run(messages=question)["messages"]
        wrong_city = self._agent([call("get_weather", city="Rome")], [FLIGHTS]).run(messages=question)["messages"]

        result = ToolCallEvaluator().run(
            ground_truth_tool_calls=[[WEATHER, FLIGHTS], [WEATHER, FLIGHTS]], predicted_messages=[good, wrong_city]
        )
        assert result == {"score": 0.5, "individual_scores": [1.0, 0.0]}


class TestSerialization:
    def test_to_dict(self):
        evaluator = ToolCallEvaluator(match_mode="strict", arguments_match="subset")
        assert evaluator.to_dict() == {
            "type": "haystack.components.evaluators.tool_call_evaluator.ToolCallEvaluator",
            "init_parameters": {"match_mode": "strict", "arguments_match": "subset"},
        }

    def test_to_dict_defaults(self):
        assert ToolCallEvaluator().to_dict()["init_parameters"] == {
            "match_mode": "unordered",
            "arguments_match": "exact",
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.evaluators.tool_call_evaluator.ToolCallEvaluator",
            "init_parameters": {"match_mode": "superset", "arguments_match": "ignore"},
        }
        evaluator = default_from_dict(ToolCallEvaluator, data)
        assert evaluator.match_mode == ToolCallMatchMode.SUPERSET
        assert evaluator.arguments_match == ToolCallArgumentsMatch.IGNORE

    def test_round_trip(self):
        evaluator = ToolCallEvaluator(match_mode="subset", arguments_match="subset")
        restored = component_from_dict(ToolCallEvaluator, component_to_dict(evaluator, "evaluator"), "evaluator")
        assert restored.match_mode == evaluator.match_mode
        assert restored.arguments_match == evaluator.arguments_match

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("evaluator", ToolCallEvaluator(match_mode="strict"))
        restored = Pipeline.loads(pipeline.dumps())
        evaluator = restored.get_component("evaluator")
        assert isinstance(evaluator, ToolCallEvaluator)
        assert evaluator.match_mode == ToolCallMatchMode.STRICT
