"""Anthropic Messages API variant of the MCP tool-use loop.

Mirrors :mod:`gateway.services.mcp_loop` but speaks the Anthropic wire shape:
``content`` blocks instead of ``tool_calls``, ``tool_use`` / ``tool_result``
blocks, ``stop_reason == "tool_use"`` as the round-continuation signal.

The duck-typed pool interface (``owns_tool`` / ``call_tool`` /
``openai_tools`` / ``purpose_hints``) is reused unchanged; the
``openai_tools`` shape is converted at the boundary in :mod:`tool_format`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from any_llm import amessages

from gateway.log_config import logger
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
)
from gateway.services.tool_format import openai_to_anthropic_tools

if TYPE_CHECKING:
    from any_llm.types.messages import (
        MessageResponse,
        MessageStreamEvent,
    )

    from gateway.services.mcp_client import MCPClientPool

# Re-export so callers in routes/messages.py have a single import surface.
__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "MaxToolIterationsExceeded",
    "anthropic_tool_loop",
    "anthropic_tool_loop_stream",
]


def _split_tool_uses(
    content: list[Any],
    pool: MCPClientPool,
) -> tuple[list[Any], bool]:
    """Return (owned_tool_use_blocks, has_foreign).

    Walks ``content`` for blocks with ``type == "tool_use"`` and partitions
    them by ``pool.owns_tool(block.name)``. Foreign = caller-supplied tool the
    gateway can't execute itself; the caller dispatches it.
    """
    owned: list[Any] = []
    has_foreign = False
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if pool.owns_tool(block.name):
            owned.append(block)
        else:
            has_foreign = True
    return owned, has_foreign


async def _execute_tool_uses(
    pool: MCPClientPool,
    blocks: list[Any],
) -> list[dict[str, Any]]:
    """Run each owned tool_use block and return the Anthropic tool_result blocks.

    Tool failures convert to a ``[tool error] …`` string in the result so the
    model can recover. Only cancellation-class exceptions
    (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape — they inherit
    from ``BaseException`` and skip the ``Exception`` clause. Same idiom as
    :func:`gateway.services.mcp_loop._execute_mcp_calls`.
    """
    out: list[dict[str, Any]] = []
    for block in blocks:
        try:
            text = await pool.call_tool(block.name, dict(block.input or {}))
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", block.name, exc)
            text = f"[tool error] {exc}"
        out.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
    return out


def _content_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Serialize a list of Anthropic content blocks back to wire shape.

    The model returned them as pydantic objects (TextBlock, ToolUseBlock,
    ThinkingBlock, …); when we feed them back as an assistant message on the
    next turn, Anthropic expects plain dicts.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True))
        elif isinstance(block, dict):
            out.append(block)
        else:
            # Defensive: any_llm should always hand us pydantic models, but
            # if a provider adapter returns a raw dict-like, accept it.
            out.append(dict(block))
    return out


async def anthropic_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: MCPClientPool,
    max_iterations: int,
) -> MessageResponse:
    """Non-streaming Anthropic Messages tool-use loop.

    Each iteration calls ``amessages``, walks the response's content blocks for
    ``tool_use`` entries, and if any are gateway-owned, executes them and
    appends the assistant + tool_result messages for the next round.

    Loop terminates when:
      * the response has no ``tool_use`` blocks (final answer);
      * ``stop_reason != "tool_use"`` (model decided to stop);
      * the response contains foreign ``tool_use`` blocks — those are returned
        to the caller for client-side dispatch. If the batch is mixed
        (owned + foreign), the owned subset is executed for its side effects
        but the response is returned with the owned blocks filtered out so
        the caller only sees what it can dispatch.

    Accumulates usage across iterations into the returned ``MessageResponse``.
    """
    messages = list(completion_kwargs.get("messages") or [])
    user_tools = list(completion_kwargs.get("tools") or [])
    merged_tools = user_tools + openai_to_anthropic_tools(pool.openai_tools)

    base = {k: v for k, v in completion_kwargs.items() if k not in {"messages", "tools", "stream"}}

    acc_input = 0
    acc_output = 0

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, "messages": messages, "stream": False}
        if merged_tools:
            kwargs["tools"] = merged_tools

        result: MessageResponse = await amessages(**kwargs)  # type: ignore[assignment]
        if result.usage:
            acc_input += result.usage.input_tokens or 0
            acc_output += result.usage.output_tokens or 0

        content = list(result.content or [])
        owned, has_foreign = _split_tool_uses(content, pool)

        if has_foreign:
            # Mixed batch: execute owned subset for its side effects, then
            # filter from the returned content so the caller only sees blocks
            # it can dispatch. Mirrors the chat-completions mixed-batch
            # handling in mcp_loop._mcp_tool_loop.
            if owned:
                await _execute_tool_uses(pool, owned)
                owned_ids = {b.id for b in owned}
                try:
                    result.content = [
                        b
                        for b in content
                        if not (getattr(b, "type", None) == "tool_use" and getattr(b, "id", None) in owned_ids)
                    ]
                except (AttributeError, TypeError):
                    logger.warning(
                        "Anthropic-mixed: could not filter content on response; client will see tool_use "
                        "blocks the gateway already executed (no-op on the client side).",
                    )
            _fold_usage(result, acc_input, acc_output)
            return result

        if not owned or result.stop_reason != "tool_use":
            _fold_usage(result, acc_input, acc_output)
            return result

        # All-owned: continue the loop. Append the assistant turn (so the model
        # sees its own tool_use blocks) and a user turn carrying tool_result.
        messages.append({"role": "assistant", "content": _content_to_dicts(content)})
        messages.append({"role": "user", "content": await _execute_tool_uses(pool, owned)})

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")


def _fold_usage(result: MessageResponse, input_total: int, output_total: int) -> None:
    """Replace ``result.usage`` token counts with the loop's running totals.

    Mirrors :func:`gateway.services.mcp_loop._fold_usage` but in Anthropic
    field naming (``input_tokens`` / ``output_tokens`` instead of
    ``prompt_tokens`` / ``completion_tokens``).
    """
    if result.usage is None:
        return
    result.usage.input_tokens = input_total
    result.usage.output_tokens = output_total


async def anthropic_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: MCPClientPool,
    max_iterations: int,
) -> AsyncIterator[MessageStreamEvent]:
    """Streaming Anthropic Messages tool-use loop.

    Forwards every Anthropic event downstream **except** the terminal
    ``message_delta`` / ``message_stop`` of an iteration that's about to
    continue (a new ``message_start`` after the client thought the message
    ended would confuse most SDK consumers).

    Per iteration:
      1. Set ``stream=True`` on ``amessages`` and iterate the event stream.
      2. Track tool_use content blocks by ``index`` from ``content_block_start``
         (when ``content_block.type == "tool_use"``). Buffer their
         ``input_json_delta`` chunks until ``content_block_stop``.
      3. Yield every event as it arrives (including the tool_use events — the
         client sees the model's tool intent even mid-loop). Defer
         ``message_delta`` and ``message_stop`` until we know whether the loop
         will continue.
      4. On ``message_stop``: if any buffered tool_use blocks exist AND all
         owned by the pool, execute them, append messages, drop the terminal
         events, and continue. If foreign blocks exist OR no tool_use blocks
         were buffered, forward the terminal events and exit.

    Re-emitting a synthetic ``message_start`` for the next iteration is not
    needed because ``amessages`` produces a fresh stream — the next call's
    natural ``message_start`` arrives downstream as if nothing had happened.
    """
    messages = list(completion_kwargs.get("messages") or [])
    user_tools = list(completion_kwargs.get("tools") or [])
    merged_tools = user_tools + openai_to_anthropic_tools(pool.openai_tools)

    base = {k: v for k, v in completion_kwargs.items() if k not in {"messages", "tools"}}
    base["stream"] = True

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, "messages": messages}
        if merged_tools:
            kwargs["tools"] = merged_tools

        stream: AsyncIterator[MessageStreamEvent] = await amessages(**kwargs)  # type: ignore[assignment]

        tool_use_blocks: dict[int, dict[str, Any]] = {}  # index -> {"id", "name", "json_buf"}
        text_buffers: dict[int, list[str]] = {}  # index -> running text fragments
        stop_reason: str | None = None
        deferred_terminal: list[MessageStreamEvent] = []

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_start":
                block = event.content_block  # type: ignore[union-attr]
                idx = event.index  # type: ignore[union-attr]
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    tool_use_blocks[idx] = {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "json_buf": "",
                    }
                elif btype == "text":
                    text_buffers[idx] = [getattr(block, "text", "") or ""]

            elif event_type == "content_block_delta":
                idx = event.index  # type: ignore[union-attr]
                delta = event.delta  # type: ignore[union-attr]
                dtype = getattr(delta, "type", None)
                if dtype == "input_json_delta" and idx in tool_use_blocks:
                    tool_use_blocks[idx]["json_buf"] += getattr(delta, "partial_json", "") or ""
                elif dtype == "text_delta" and idx in text_buffers:
                    text_buffers[idx].append(getattr(delta, "text", "") or "")

            elif event_type == "message_delta":
                stop_reason = getattr(event.delta, "stop_reason", None) or stop_reason  # type: ignore[union-attr]
                deferred_terminal.append(event)
                continue

            elif event_type == "message_stop":
                deferred_terminal.append(event)
                # Fall through to the post-stream decision below.
                break

            yield event

        # Decide whether to loop or exit.
        if not tool_use_blocks or stop_reason != "tool_use":
            for term in deferred_terminal:
                yield term
            return

        # Build owned/foreign blocks for execution. We don't have the full
        # pydantic ToolUseBlock; we only need .id / .name / .input.
        owned_specs: list[dict[str, Any]] = []
        has_foreign = False
        for idx in sorted(tool_use_blocks):
            spec = tool_use_blocks[idx]
            name = spec["name"]
            if pool.owns_tool(name):
                owned_specs.append(spec)
            else:
                has_foreign = True

        if has_foreign or not owned_specs:
            # Caller dispatches the foreign blocks (or there's nothing to do
            # internally). Forward terminal events and exit. Mixed (owned +
            # foreign) is handled the same way as all-foreign in streaming
            # mode — rewriting deltas mid-stream to remove the owned blocks
            # would be too invasive; the client sees the full set.
            for term in deferred_terminal:
                yield term
            return

        # All-owned: execute and continue the loop. Drop the deferred terminal
        # events so the client doesn't see a premature "message ended" signal.
        assistant_content: list[dict[str, Any]] = []
        for idx in sorted(text_buffers):
            assistant_content.append({"type": "text", "text": "".join(text_buffers[idx])})
        for spec in owned_specs:
            try:
                parsed_input = json.loads(spec["json_buf"] or "{}")
            except json.JSONDecodeError:
                parsed_input = {}
            assistant_content.append(
                {"type": "tool_use", "id": spec["id"], "name": spec["name"], "input": parsed_input}
            )

        # Execute and build tool_result blocks for the next user message.
        tool_results: list[dict[str, Any]] = []
        for spec in owned_specs:
            try:
                parsed_input = json.loads(spec["json_buf"] or "{}")
            except json.JSONDecodeError:
                parsed_input = {}
            try:
                text = await pool.call_tool(spec["name"], parsed_input)
            except Exception as exc:  # noqa: BLE001 — same tool-error-as-message idiom as the non-stream loop
                logger.warning("MCP tool %s execution failed: %s", spec["name"], exc)
                text = f"[tool error] {exc}"
            tool_results.append({"type": "tool_result", "tool_use_id": spec["id"], "content": text})

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")
