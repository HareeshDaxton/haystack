# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class SchemaGenerationError(Exception):
    """
    Exception raised when automatic schema generation fails.
    """

    pass


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
    """

    def __init__(self, message: str, tool_name: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ToolRetryError(Exception):
    """
    Exception a tool raises to send an error message back to the Language Model, so it can fix its tool call.

    Raise it when the failure is caused by the tool call itself, for example a wrong date format, an unknown ID, or a
    query that is too broad, and you know how the Language Model should fix it.

    When an `Agent` runs a tool that raises `ToolRetryError`:
    - The message is sent to the Language Model as it is, as an error tool result. Unlike other exceptions, it is not
      wrapped in a "Failed to invoke Tool" message and the tool arguments are not included, so values injected from
      the Agent's State never reach the Language Model.
    - The message is sent to the Language Model even when `raise_on_tool_invocation_failure=True`, so the Agent run
      continues. All other exceptions keep their usual behavior.

    `Tool.invoke` still raises a `ToolInvocationError` with the `ToolRetryError` as its `__cause__`.

    Usage example:
    ```python
    from datetime import date

    from haystack.tools import ToolRetryError, tool

    @tool
    def book_flight(day: str) -> str:
        '''Book a flight for the given day, in the format YYYY-MM-DD.'''
        try:
            flight_date = date.fromisoformat(day)
        except ValueError as e:
            raise ToolRetryError(f"'{day}' is not a valid date. Use the format YYYY-MM-DD, e.g. 2026-01-31.") from e
        return f"Booked a flight on {flight_date}."
    ```
    """

    def __init__(self, message: str) -> None:
        """
        Create a ToolRetryError.

        :param message: The message sent to the Language Model. Say what went wrong and how to fix the tool call.
        """
        super().__init__(message)
        self.message = message
