# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

import pytest

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.agents.tool_calling import _find_tool_retry_error, _run_tool, _run_tool_async
from haystack.components.generators.chat import MockChatGenerator
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import ComponentTool, PipelineTool, Tool, create_tool_from_function
from haystack.tools.errors import ToolInvocationError, ToolRetryError

RETRY_MESSAGE = "'next friday' is not a valid date. Use the format YYYY-MM-DD, for example 2026-01-31."

DATE_PARAMETERS = {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}


def book_flight(date: Annotated[str, "Flight date, format YYYY-MM-DD"], api_key: str) -> str:
    """Book a flight."""
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ToolRetryError(f"'{date}' is not a valid date. Use the format YYYY-MM-DD, for example 2026-01-31.")
    return f"Booked a flight on {date}."


async def async_book_flight(date: str) -> str:
    if date == "next friday":
        raise ToolRetryError(RETRY_MESSAGE)
    return f"Booked a flight on {date}."


def always_retry(date: str) -> str:
    raise ToolRetryError(f"Cannot book '{date}'. Ask the user for an exact date.")


def failing_function(date: str) -> str:
    raise ValueError(f"database is down, cannot book {date}")


@pytest.fixture
def book_flight_tool() -> Tool:
    return create_tool_from_function(book_flight, inputs_from_state={"api_key": "api_key"})


@pytest.fixture
def async_book_flight_tool() -> Tool:
    return Tool(
        name="book_flight", description="Book a flight.", parameters=DATE_PARAMETERS, async_function=async_book_flight
    )


@pytest.fixture
def failing_tool() -> Tool:
    return Tool(name="failing_tool", description="Always fails.", parameters=DATE_PARAMETERS, function=failing_function)


def _state_with_api_key() -> State:
    return State(schema={"api_key": {"type": str}}, data={"api_key": "sk-SECRET-123"})


def _tool_call_message(*tool_calls: ToolCall) -> ChatMessage:
    return ChatMessage.from_assistant(tool_calls=list(tool_calls))


class TestToolRetryError:
    def test_message_is_kept(self):
        error = ToolRetryError(RETRY_MESSAGE)
        assert error.message == RETRY_MESSAGE
        assert str(error) == RETRY_MESSAGE

    def test_is_not_a_tool_invocation_error(self):
        # Code that catches ToolInvocationError around tool code must not silently swallow it
        assert not issubclass(ToolRetryError, ToolInvocationError)
        assert issubclass(ToolRetryError, Exception)

    def test_is_exported_from_haystack_tools(self):
        from haystack.tools import ToolRetryError as ExportedToolRetryError

        assert ExportedToolRetryError is ToolRetryError


class TestFindToolRetryError:
    def test_returns_the_error_itself(self):
        error = ToolRetryError(RETRY_MESSAGE)
        assert _find_tool_retry_error(error) is error

    def test_finds_error_wrapped_by_tool_invocation_error(self, book_flight_tool):
        with pytest.raises(ToolInvocationError) as exc_info:
            book_flight_tool.invoke(date="next friday", api_key="x")

        found = _find_tool_retry_error(exc_info.value)
        assert isinstance(found, ToolRetryError)
        assert found.message == RETRY_MESSAGE

    def test_finds_error_deep_in_the_cause_chain(self):
        retry = ToolRetryError(RETRY_MESSAGE)
        middle = PipelineRuntimeError(component_name="booker", component_type=None, message="component failed")
        middle.__cause__ = retry
        outer = ToolInvocationError("Failed to invoke Tool", tool_name="booker")
        outer.__cause__ = middle

        assert _find_tool_retry_error(outer) is retry

    def test_returns_none_without_retry_error(self):
        outer = ToolInvocationError("Failed to invoke Tool", tool_name="tool")
        outer.__cause__ = ValueError("boom")

        assert _find_tool_retry_error(outer) is None

    def test_ignores_implicit_context(self):
        # Only explicit causes (`raise ... from ...`) count: a new error raised while handling a ToolRetryError is a
        # real failure, not a retry request
        outer = ToolInvocationError("Failed to invoke Tool", tool_name="tool")
        outer.__context__ = ToolRetryError(RETRY_MESSAGE)

        assert _find_tool_retry_error(outer) is None

    def test_stops_on_cause_cycle(self):
        first = ValueError("first")
        second = ValueError("second")
        first.__cause__ = second
        second.__cause__ = first

        assert _find_tool_retry_error(first) is None


class TestToolInvokeContract:
    def test_invoke_still_raises_tool_invocation_error(self, book_flight_tool):
        with pytest.raises(ToolInvocationError) as exc_info:
            book_flight_tool.invoke(date="next friday", api_key="x")

        assert isinstance(exc_info.value.__cause__, ToolRetryError)
        assert exc_info.value.tool_name == "book_flight"

    @pytest.mark.asyncio
    async def test_invoke_async_still_raises_tool_invocation_error(self, async_book_flight_tool):
        with pytest.raises(ToolInvocationError) as exc_info:
            await async_book_flight_tool.invoke_async(date="next friday")

        assert isinstance(exc_info.value.__cause__, ToolRetryError)


class TestRunTool:
    @pytest.mark.parametrize("raise_on_failure", [True, False])
    def test_retry_message_is_sent_as_is(self, book_flight_tool, raise_on_failure):
        message = _tool_call_message(ToolCall(id="1", tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = _run_tool(
            messages=[message], state=_state_with_api_key(), tools=[book_flight_tool], raise_on_failure=raise_on_failure
        )

        assert len(tool_messages) == 1
        result = tool_messages[0].tool_call_results[0]
        assert result.result == RETRY_MESSAGE
        assert result.error is True
        assert result.origin.id == "1"

    def test_state_values_are_not_sent_to_the_llm(self, book_flight_tool):
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = _run_tool(messages=[message], state=_state_with_api_key(), tools=[book_flight_tool])

        result = tool_messages[0].tool_call_results[0].result
        assert "sk-SECRET-123" not in result
        assert "Failed to invoke" not in result

    def test_other_errors_still_raise_with_raise_on_failure(self, failing_tool):
        message = _tool_call_message(ToolCall(tool_name="failing_tool", arguments={"date": "2026-01-31"}))

        with pytest.raises(ToolInvocationError, match="database is down"):
            _run_tool(messages=[message], state=State(schema={}), tools=[failing_tool], raise_on_failure=True)

    def test_other_errors_keep_the_default_message(self, failing_tool):
        message = _tool_call_message(ToolCall(tool_name="failing_tool", arguments={"date": "2026-01-31"}))

        tool_messages, _ = _run_tool(
            messages=[message], state=State(schema={}), tools=[failing_tool], raise_on_failure=False
        )

        result = tool_messages[0].tool_call_results[0]
        assert result.error is True
        assert isinstance(result.result, str)
        assert result.result.startswith("Failed to invoke Tool `failing_tool`")

    def test_mixed_tool_calls_keep_call_order(self, book_flight_tool):
        message = _tool_call_message(
            ToolCall(id="1", tool_name="book_flight", arguments={"date": "next friday"}),
            ToolCall(id="2", tool_name="book_flight", arguments={"date": "2026-11-20"}),
        )

        tool_messages, _ = _run_tool(
            messages=[message], state=_state_with_api_key(), tools=[book_flight_tool], raise_on_failure=True
        )

        results = [m.tool_call_results[0] for m in tool_messages]
        assert [r.origin.id for r in results] == ["1", "2"]
        assert [r.error for r in results] == [True, False]
        assert results[0].result == RETRY_MESSAGE
        assert results[1].result == "Booked a flight on 2026-11-20."

    def test_outputs_to_state_is_not_merged(self):
        tool = Tool(
            name="book_flight",
            description="Book a flight.",
            parameters=DATE_PARAMETERS,
            function=always_retry,
            outputs_to_state={"booking": {}},
        )
        state = State(schema={"booking": {"type": str}})
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        _run_tool(messages=[message], state=state, tools=[tool])

        assert not state.has("booking")

    def test_streaming_callback_receives_retry_message(self, book_flight_tool):
        chunks: list[StreamingChunk] = []
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        _run_tool(
            messages=[message], state=_state_with_api_key(), tools=[book_flight_tool], streaming_callback=chunks.append
        )

        tool_results = [c.tool_call_result for c in chunks if c.tool_call_result is not None]
        assert len(tool_results) == 1
        assert tool_results[0].result == RETRY_MESSAGE
        assert tool_results[0].error is True

    def test_logs_a_warning(self, book_flight_tool, caplog):
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        with caplog.at_level(logging.WARNING):
            _run_tool(messages=[message], state=_state_with_api_key(), tools=[book_flight_tool])

        assert "book_flight" in caplog.text
        assert "ToolRetryError" in caplog.text
        assert "sk-SECRET-123" not in caplog.text


class TestRunToolAsync:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raise_on_failure", [True, False])
    async def test_async_function_retry_message_is_sent_as_is(self, async_book_flight_tool, raise_on_failure):
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = await _run_tool_async(
            messages=[message],
            state=State(schema={}),
            tools=[async_book_flight_tool],
            raise_on_failure=raise_on_failure,
        )

        result = tool_messages[0].tool_call_results[0]
        assert result.result == RETRY_MESSAGE
        assert result.error is True

    @pytest.mark.asyncio
    async def test_sync_function_retry_message_is_sent_as_is(self, book_flight_tool):
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = await _run_tool_async(
            messages=[message], state=_state_with_api_key(), tools=[book_flight_tool], raise_on_failure=True
        )

        result = tool_messages[0].tool_call_results[0]
        assert result.result == RETRY_MESSAGE
        assert "sk-SECRET-123" not in result.result

    @pytest.mark.asyncio
    async def test_other_errors_still_raise_with_raise_on_failure(self, failing_tool):
        message = _tool_call_message(ToolCall(tool_name="failing_tool", arguments={"date": "2026-01-31"}))

        with pytest.raises(ToolInvocationError, match="database is down"):
            await _run_tool_async(
                messages=[message], state=State(schema={}), tools=[failing_tool], raise_on_failure=True
            )


@component
class FlightBooker:
    @component.output_types(confirmation=str)
    def run(self, date: str) -> dict[str, str]:
        """
        Book a flight.

        :param date: Flight date, format YYYY-MM-DD.
        """
        if date == "next friday":
            raise ToolRetryError(RETRY_MESSAGE)
        return {"confirmation": f"Booked a flight on {date}."}


class TestWrappedTools:
    def test_component_tool(self):
        tool = ComponentTool(component=FlightBooker(), name="book_flight")
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = _run_tool(messages=[message], state=State(schema={}), tools=[tool], raise_on_failure=True)

        assert tool_messages[0].tool_call_results[0].result == RETRY_MESSAGE
        assert tool_messages[0].tool_call_results[0].error is True

    def test_pipeline_tool(self):
        pipeline = Pipeline()
        pipeline.add_component("booker", FlightBooker())
        tool = PipelineTool(
            pipeline=pipeline,
            name="book_flight",
            description="Book a flight.",
            input_mapping={"date": ["booker.date"]},
            output_mapping={"booker.confirmation": "confirmation"},
        )
        message = _tool_call_message(ToolCall(tool_name="book_flight", arguments={"date": "next friday"}))

        tool_messages, _ = _run_tool(messages=[message], state=State(schema={}), tools=[tool], raise_on_failure=True)

        assert tool_messages[0].tool_call_results[0].result == RETRY_MESSAGE
        assert tool_messages[0].tool_call_results[0].error is True


def _fixing_llm_response(messages: list[ChatMessage]) -> ChatMessage:
    """Scripted LLM: tries a bad date, fixes it after a ToolRetryError message, then answers."""
    last = messages[-1]
    if last.tool_call_result is None:
        return ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="book_flight", arguments={"date": "next friday"})]
        )
    if last.tool_call_result.error:
        assert last.tool_call_result.result == RETRY_MESSAGE
        return ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="book_flight", arguments={"date": "2026-11-20"})]
        )
    return ChatMessage.from_assistant("Your flight on 2026-11-20 is booked.")


class TestAgent:
    def test_agent_recovers_with_raise_on_tool_invocation_failure(self, book_flight_tool):
        agent = Agent(
            chat_generator=MockChatGenerator(response_fn=_fixing_llm_response),
            tools=[book_flight_tool],
            state_schema={"api_key": {"type": str}},
            raise_on_tool_invocation_failure=True,
        )

        result = agent.run(messages=[ChatMessage.from_user("Book a flight next friday")], api_key="sk-SECRET-123")

        assert result["last_message"].text == "Your flight on 2026-11-20 is booked."
        tool_results = [m.tool_call_results[0] for m in result["messages"] if m.tool_call_results]
        assert [r.result for r in tool_results] == [RETRY_MESSAGE, "Booked a flight on 2026-11-20."]
        assert all("sk-SECRET-123" not in m.text for m in result["messages"] if m.text)

    @pytest.mark.asyncio
    async def test_agent_run_async_recovers_with_raise_on_tool_invocation_failure(self, async_book_flight_tool):
        agent = Agent(
            chat_generator=MockChatGenerator(response_fn=_fixing_llm_response),
            tools=[async_book_flight_tool],
            raise_on_tool_invocation_failure=True,
        )

        result = await agent.run_async(messages=[ChatMessage.from_user("Book a flight next friday")])

        assert result["last_message"].text == "Your flight on 2026-11-20 is booked."
        tool_results = [m.tool_call_results[0] for m in result["messages"] if m.tool_call_results]
        assert tool_results[0].result == RETRY_MESSAGE
        assert tool_results[0].error is True

    def test_agent_still_stops_on_other_errors(self, failing_tool):
        llm = MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="failing_tool", arguments={"date": "2026-11-20"})]
                ),
                "done",
            ]
        )
        agent = Agent(chat_generator=llm, tools=[failing_tool], raise_on_tool_invocation_failure=True)

        with pytest.raises(ToolInvocationError, match="database is down"):
            agent.run(messages=[ChatMessage.from_user("Book a flight")])
