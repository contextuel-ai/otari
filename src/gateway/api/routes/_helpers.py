from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Response, status

from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailsNotReachableError, run_input_guardrails

if TYPE_CHECKING:
    from gateway.db import APIKey


GUARDRAILS_RESULT_HEADER = "X-Otari-Guardrails"
"""Response header carrying a compact JSON summary of guardrail verdicts when a
``monitor``-mode (or otherwise non-blocking) check ran."""


def resolve_user_id(
    user_id_from_request: str | None,
    api_key: APIKey | None,
    is_master_key: bool,
    *,
    master_key_error: HTTPException,
    no_api_key_error: HTTPException,
    no_user_error: HTTPException,
) -> str:
    """Resolve the effective user_id from request context.

    The resolution order is:
    1. If master key is used, the request *must* supply a user_id.
    2. If the request supplies a user_id, use it.
    3. Fall back to the user_id associated with the API key.

    Args:
        user_id_from_request: User identifier extracted from the request body
        api_key: Authenticated API key object (None when using master key)
        is_master_key: Whether the request was authenticated with a master key
        master_key_error: Raised when master key is used but no user_id is provided
        no_api_key_error: Raised when no API key is available
        no_user_error: Raised when the API key has no associated user

    Returns:
        Resolved user_id string

    """
    if is_master_key:
        if not user_id_from_request:
            raise master_key_error
        return user_id_from_request

    if user_id_from_request:
        return user_id_from_request

    if api_key is None:
        raise no_api_key_error
    if not api_key.user_id:
        raise no_user_error
    return str(api_key.user_id)


def text_from_content(content: Any) -> str:
    """Flatten a message ``content`` value to plain text for guardrail checks.

    Handles the two wire shapes shared across the chat and Anthropic-messages
    formats: a bare string, or a list of content parts where text parts look
    like ``{"type": "text", "text": "..."}``. Non-text parts (images, tool
    results, etc.) are ignored — guardrails like prompt-injection detection
    operate on the textual prompt.

    Returns:
        The flattened text, or an empty string for unrecognised shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the most recent ``role == "user"`` message.

    Falls back to the last message of any role if no user message is present.
    Used to feed input-direction guardrails the prompt the model is about to
    see.

    Returns:
        The latest user message's text, or an empty string if ``messages`` is
        empty.
    """
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return text_from_content(message.get("content"))
    return text_from_content(messages[-1].get("content")) if messages else ""


async def apply_input_guardrails(
    guardrails: list[GuardrailConfig] | None,
    input_text: str,
    *,
    response: Response,
) -> None:
    """Enforce caller-requested input guardrails before the provider call.

    No-op when ``guardrails`` is empty/None (zero overhead for the common
    case). On a ``block``-mode flag, raises ``403`` and the provider is never
    called. On a non-blocking flag (``monitor`` mode), attaches a compact
    summary to the :data:`GUARDRAILS_RESULT_HEADER` response header and lets
    the request proceed.

    Service-failure handling is mode-dependent (see
    :func:`gateway.services.guardrails.run_input_guardrails`): a ``block``
    guardrail that can't be evaluated fails closed (``502``); a ``monitor``
    guardrail fails open (logged, request proceeds).

    Note:
        The header is set on the injected ``response``, so it reaches
        non-streaming responses. For streamed responses (where the route
        returns its own ``StreamingResponse``) the ``monitor`` annotation is
        not currently propagated; ``block`` still applies (it raises before any
        bytes are streamed).

    Raises:
        HTTPException: ``403`` when a ``block`` guardrail flags the input;
            ``502`` when a ``block`` guardrail can't be evaluated.
    """
    if not guardrails:
        return

    default_url = os.environ.get("GATEWAY_GUARDRAILS_URL") or None
    try:
        verdict = await run_input_guardrails(guardrails, input_text, default_url=default_url)
    except GuardrailsNotReachableError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if verdict.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Request blocked by guardrail policy.",
                "code": "guardrail_violation",
                "guardrails": [
                    {
                        "profile": r.profile,
                        "explanation": r.explanation,
                        "score": r.score,
                    }
                    for r in verdict.flagged
                    if r.mode == "block"
                ],
            },
        )

    if verdict.results:
        # Non-blocking: surface the verdict for observability (monitor mode, or
        # a passing block-mode check). Header value is kept compact and free of
        # the freeform `explanation` to avoid oversized / non-ASCII headers.
        summary = [
            {"profile": r.profile, "mode": r.mode, "valid": r.valid, "score": r.score}
            for r in verdict.results
        ]
        response.headers[GUARDRAILS_RESULT_HEADER] = json.dumps(summary, separators=(",", ":"))
