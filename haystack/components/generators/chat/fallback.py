# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, get_args

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.generators.utils import _normalize_messages, _trace_chat_generator_run
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.dataclasses.streaming_chunk import FinishReason
from haystack.tools import ToolsType
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.type_serialization import deserialize_type, serialize_type

logger = logging.getLogger(__name__)


@component
class FallbackChatGenerator:
    """
    A chat generator wrapper that tries multiple chat generators sequentially.

    It forwards all parameters transparently to the underlying chat generators and returns the first successful result.
    Calls chat generators sequentially until one succeeds. Falls back on any exception raised by a generator.
    If all chat generators fail, it raises a RuntimeError with details.

    Timeout enforcement is fully delegated to the underlying chat generators. The fallback mechanism will only
    work correctly if the underlying chat generators implement proper timeout handling and raise exceptions
    when timeouts occur. For predictable latency guarantees, ensure your chat generators:
    - Support a `timeout` parameter in their initialization
    - Implement timeout as total wall-clock time (shared deadline for both streaming and non-streaming)
    - Raise timeout exceptions (e.g., TimeoutError, asyncio.TimeoutError, httpx.TimeoutException) when exceeded

    Note: Most well-implemented chat generators (OpenAI, Anthropic, Cohere, etc.) support timeout parameters
    with consistent semantics. For HTTP-based LLM providers, a single timeout value (e.g., `timeout=30`)
    typically applies to all connection phases: connection setup, read, write, and pool. For streaming
    responses, read timeout is the maximum gap between chunks. For non-streaming, it's the time limit for
    receiving the complete response.

    Fail over is automatically triggered when a generator raises any exception, including:
    - Timeout errors (if the generator implements and raises them)
    - Rate limit errors (429)
    - Authentication errors (401)
    - Context length errors (400)
    - Server errors (500+)
    - Any other exception

    Two optional parameters refine this behavior:
    - `fallback_on_finish_reasons` also treats a returned reply as a failed attempt when its `finish_reason` is in
      the list, for example `["content_filter"]` when the provider's safety filter blocked the reply.
    - `raise_on_exceptions` re-raises the listed exception types immediately instead of falling back, so programming
      errors such as `TypeError` surface right away instead of silently running on the next chat generator.

    ### Usage example

    ```python
    from haystack.components.generators.chat import FallbackChatGenerator, OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = FallbackChatGenerator(
        chat_generators=[OpenAIChatGenerator(model="gpt-4o"), OpenAIChatGenerator(model="gpt-4o-mini")],
        fallback_on_finish_reasons=["content_filter"],
        raise_on_exceptions=[TypeError, ValueError],
    )
    result = generator.run(messages=[ChatMessage.from_user("Tell me about the history of mining explosives.")])
    print(result["replies"][0].text)
    print(result["meta"]["failed_chat_generators"], result["meta"]["failure_reasons"])
    ```
    """

    def __init__(
        self,
        chat_generators: list[ChatGenerator],
        *,
        fallback_on_finish_reasons: list[FinishReason] | None = None,
        raise_on_exceptions: list[type[Exception]] | None = None,
    ) -> None:
        """
        Creates an instance of FallbackChatGenerator.

        :param chat_generators: A non-empty list of chat generator components to try in order.
        :param fallback_on_finish_reasons: Finish reasons that make a returned reply count as a failed attempt,
            for example `["content_filter"]`. The next chat generator is tried when every reply of a chat generator
            has one of these finish reasons. If no chat generator returns an accepted reply, the first rejected
            result is returned, so this option never turns a returned reply into an error.
        :param raise_on_exceptions: Exception types that are re-raised immediately instead of triggering a fallback,
            for example `[TypeError, ValueError]` to surface programming errors.
        :raises ValueError: If `chat_generators` is empty or a finish reason is not valid.
        :raises TypeError: If `raise_on_exceptions` contains something that is not an exception class.
        """
        if not chat_generators:
            msg = "'chat_generators' must be a non-empty list"
            raise ValueError(msg)
        valid_reasons = get_args(FinishReason)
        for reason in fallback_on_finish_reasons or []:
            if reason not in valid_reasons:
                raise ValueError(f"Invalid finish reason '{reason}'. Valid values are: {', '.join(valid_reasons)}.")
        for exc in raise_on_exceptions or []:
            if not (isinstance(exc, type) and issubclass(exc, Exception)):
                raise TypeError(f"'raise_on_exceptions' must contain exception classes, got {exc!r}.")

        self.chat_generators = list(chat_generators)
        self.fallback_on_finish_reasons = fallback_on_finish_reasons
        self.raise_on_exceptions = raise_on_exceptions

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component, including nested chat generators."""
        return default_to_dict(
            self,
            chat_generators=[
                component_to_dict(gen, name=f"chat_generator_{idx}") for idx, gen in enumerate(self.chat_generators)
            ],
            fallback_on_finish_reasons=self.fallback_on_finish_reasons,
            raise_on_exceptions=[serialize_type(exc) for exc in self.raise_on_exceptions]
            if self.raise_on_exceptions is not None
            else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FallbackChatGenerator:
        """Rebuild the component from a serialized representation, restoring nested chat generators."""
        # Reconstruct nested chat generators from their serialized dicts
        init_params = data.get("init_parameters", {})
        serialized = init_params.get("chat_generators") or []
        deserialized: list[Any] = []
        for g in serialized:
            # Use the generic component deserializer available in Haystack
            holder = {"component": g}
            deserialize_component_inplace(holder, key="component")
            deserialized.append(holder["component"])
        init_params["chat_generators"] = deserialized
        if init_params.get("raise_on_exceptions") is not None:
            init_params["raise_on_exceptions"] = [deserialize_type(exc) for exc in init_params["raise_on_exceptions"]]
        data["init_parameters"] = init_params
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """Warm up all underlying chat generators."""
        for gen in self.chat_generators:
            if hasattr(gen, "warm_up"):
                gen.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up all underlying chat generators on the serving event loop."""
        for gen in self.chat_generators:
            if hasattr(gen, "warm_up_async"):
                await gen.warm_up_async()
            elif hasattr(gen, "warm_up"):
                gen.warm_up()

    def close(self) -> None:
        """Release the underlying chat generators' resources."""
        for gen in self.chat_generators:
            if hasattr(gen, "close"):
                gen.close()

    async def close_async(self) -> None:
        """Release the underlying chat generators' async resources."""
        for gen in self.chat_generators:
            if hasattr(gen, "close_async"):
                await gen.close_async()
            elif hasattr(gen, "close"):
                gen.close()

    def _rejected_finish_reason(self, result: dict[str, Any]) -> str | None:
        """Return the finish reason that rejects this result, or None if the result is accepted."""
        replies = result.get("replies") or []
        if not self.fallback_on_finish_reasons or not replies:
            return None
        reasons = [reply.meta.get("finish_reason") for reply in replies]
        # Only reject when no reply is usable, so a good reply is never thrown away.
        if all(reason in self.fallback_on_finish_reasons for reason in reasons):
            return reasons[0]
        return None

    def _should_raise(self, error: Exception) -> bool:
        if not self.raise_on_exceptions:
            return False
        return isinstance(error, tuple(self.raise_on_exceptions))

    @staticmethod
    def _build_output(
        result: dict[str, Any], idx: int, gen_name: str, attempts: int, failed: list[str], failure_reasons: list[str]
    ) -> dict[str, Any]:
        meta = dict(result.get("meta", {}))
        meta.update(
            {
                "successful_chat_generator_index": idx,
                "successful_chat_generator_class": gen_name,
                "total_attempts": attempts,
                "failed_chat_generators": failed,
                "failure_reasons": failure_reasons,
            }
        )
        return {"replies": result.get("replies", []), "meta": meta}

    def _finish(
        self,
        rejected: tuple[int, str, dict[str, Any]] | None,
        failed: list[str],
        failure_reasons: list[str],
        last_error: BaseException | None,
    ) -> dict[str, Any]:
        """Called when no chat generator returned an accepted reply."""
        if rejected is not None:
            # Same as before this option existed: hand back the reply, its finish_reason tells what happened.
            idx, gen_name, result = rejected
            logger.warning(
                "No chat generator returned an accepted reply. Returning the reply of {chat_generator}.",
                chat_generator=gen_name,
            )
            return self._build_output(result, idx, gen_name, len(self.chat_generators), failed, failure_reasons)
        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.chat_generators)} chat generators failed. "
            f"Last error: {last_error}. Failed chat generators: [{failed_names}]"
        )
        raise RuntimeError(msg) from last_error

    def _run_single_sync(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        generator_inputs = {
            "messages": messages,
            "generation_kwargs": generation_kwargs,
            "tools": tools,
            "streaming_callback": streaming_callback,
        }
        with _trace_chat_generator_run(chat_generator=gen, generator_inputs=generator_inputs):
            return gen.run(**generator_inputs)

    async def _run_single_async(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        generator_inputs = {
            "messages": messages,
            "generation_kwargs": generation_kwargs,
            "tools": tools,
            "streaming_callback": streaming_callback,
        }
        with _trace_chat_generator_run(chat_generator=gen, generator_inputs=generator_inputs):
            return await _execute_component_async(component_instance=gen, **generator_inputs)

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage] | dict[str, Any]]:
        """
        Execute chat generators sequentially until one succeeds.

        :param messages: The conversation history as a list of ChatMessage instances.
        :param generation_kwargs: Optional parameters for the chat generator (e.g., temperature, max_tokens).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for function calling capabilities.
        :param streaming_callback: Optional callable for handling streaming responses.
        :returns: A dictionary with:
            - "replies": Generated ChatMessage instances from the first successful generator.
            - "meta": Execution metadata including successful_chat_generator_index, successful_chat_generator_class,
              total_attempts, failed_chat_generators, failure_reasons (why each failed chat generator was skipped,
              aligned with failed_chat_generators), plus any metadata from the successful generator.
              If every chat generator returned a reply rejected by `fallback_on_finish_reasons`, the result of the
              first rejected chat generator is returned.
        :raises RuntimeError: If all chat generators raise an exception.
        """
        self.warm_up()

        messages = _normalize_messages(messages=messages)

        failed: list[str] = []
        failure_reasons: list[str] = []
        rejected: tuple[int, str, dict[str, Any]] | None = None
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.chat_generators):
            gen_name = gen.__class__.__name__
            try:
                result = self._run_single_sync(gen, messages, generation_kwargs, tools, streaming_callback)
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                if self._should_raise(e):
                    raise
                logger.warning(
                    "ChatGenerator {chat_generator} failed with error: {error}", chat_generator=gen_name, error=e
                )
                failed.append(gen_name)
                failure_reasons.append(f"{type(e).__name__}: {e}")
                last_error = e
                continue

            finish_reason = self._rejected_finish_reason(result)
            if finish_reason is None:
                return self._build_output(result, idx, gen_name, idx + 1, failed, failure_reasons)
            logger.warning(
                "ChatGenerator {chat_generator} returned finish reason '{finish_reason}'. Trying the next one.",
                chat_generator=gen_name,
                finish_reason=finish_reason,
            )
            failed.append(gen_name)
            failure_reasons.append(f"finish_reason: {finish_reason}")
            if rejected is None:
                rejected = (idx, gen_name, result)

        return self._finish(rejected, failed, failure_reasons, last_error)

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage] | dict[str, Any]]:
        """
        Asynchronously execute chat generators sequentially until one succeeds.

        :param messages: The conversation history as a list of ChatMessage instances.
        :param generation_kwargs: Optional parameters for the chat generator (e.g., temperature, max_tokens).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for function calling capabilities.
        :param streaming_callback: Optional callable for handling streaming responses.
        :returns: A dictionary with:
            - "replies": Generated ChatMessage instances from the first successful generator.
            - "meta": Execution metadata including successful_chat_generator_index, successful_chat_generator_class,
              total_attempts, failed_chat_generators, failure_reasons (why each failed chat generator was skipped,
              aligned with failed_chat_generators), plus any metadata from the successful generator.
              If every chat generator returned a reply rejected by `fallback_on_finish_reasons`, the result of the
              first rejected chat generator is returned.
        :raises RuntimeError: If all chat generators raise an exception.
        """
        await self.warm_up_async()

        messages = _normalize_messages(messages=messages)

        failed: list[str] = []
        failure_reasons: list[str] = []
        rejected: tuple[int, str, dict[str, Any]] | None = None
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.chat_generators):
            gen_name = gen.__class__.__name__
            try:
                result = await self._run_single_async(gen, messages, generation_kwargs, tools, streaming_callback)
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                if self._should_raise(e):
                    raise
                logger.warning(
                    "ChatGenerator {chat_generator} failed with error: {error}", chat_generator=gen_name, error=e
                )
                failed.append(gen_name)
                failure_reasons.append(f"{type(e).__name__}: {e}")
                last_error = e
                continue

            finish_reason = self._rejected_finish_reason(result)
            if finish_reason is None:
                return self._build_output(result, idx, gen_name, idx + 1, failed, failure_reasons)
            logger.warning(
                "ChatGenerator {chat_generator} returned finish reason '{finish_reason}'. Trying the next one.",
                chat_generator=gen_name,
                finish_reason=finish_reason,
            )
            failed.append(gen_name)
            failure_reasons.append(f"finish_reason: {finish_reason}")
            if rejected is None:
                rejected = (idx, gen_name, result)

        return self._finish(rejected, failed, failure_reasons, last_error)
