"""
ChatGPT Responses API adapter.

Translates standard OpenAI Chat Completions requests into the ChatGPT
backend Responses API format, sends them to chatgpt.com, collects the
streamed response, and returns a standard Chat Completion dict.

Ported from ChatMock (https://github.com/RayBytes/ChatMock), adapted to
async httpx and stripped down to the essentials.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHATGPT_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

_MODEL_MAP = {
    "gpt5": "gpt-5",
    "gpt-5-latest": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt5.2": "gpt-5.2",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-latest": "gpt-5.2",
    "gpt5.2-codex": "gpt-5.2-codex",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.2-codex-latest": "gpt-5.2-codex",
    "gpt5-codex": "gpt-5-codex",
    "gpt-5-codex": "gpt-5-codex",
    "gpt-5-codex-latest": "gpt-5-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "codex": "codex-mini-latest",
    "codex-mini": "codex-mini-latest",
    "codex-mini-latest": "codex-mini-latest",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
}


# ---------------------------------------------------------------------------
# Model name normalisation
# ---------------------------------------------------------------------------


def normalize_model_name(name: str | None) -> str:
    """Normalize a model name to the canonical ChatGPT Responses API form."""
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"
    base = name.split(":", 1)[0].strip()
    # Strip reasoning-effort suffixes
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    return _MODEL_MAP.get(base, base)


# ---------------------------------------------------------------------------
# Message / tool conversion helpers
# ---------------------------------------------------------------------------


def convert_chat_messages_to_responses_input(
    messages: List[Dict[str, Any]],
) -> tuple[str | None, List[Dict[str, Any]]]:
    """
    Convert Chat Completions messages to Responses API ``input`` items.

    System messages are extracted and returned separately as *instructions*.

    Returns:
        (instructions, input_items)
    """
    instructions: str | None = None
    input_items: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        # System → instructions (take last system message)
        if role == "system":
            content = message.get("content", "")
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text") or part.get("content")
                        if isinstance(t, str) and t:
                            texts.append(t)
                content = "\n".join(texts)
            if isinstance(content, str) and content.strip():
                instructions = content
            continue

        # Tool results
        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("id")
            if isinstance(call_id, str) and call_id:
                content = message.get("content", "")
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("content")
                            if isinstance(t, str) and t:
                                texts.append(t)
                    content = "\n".join(texts)
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": content,
                        }
                    )
            continue

        # Assistant tool calls → function_call items
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                if tc.get("type", "function") != "function":
                    continue
                call_id = tc.get("id") or tc.get("call_id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") if isinstance(fn, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if (
                    isinstance(call_id, str)
                    and isinstance(name, str)
                    and isinstance(args, str)
                ):
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": name,
                            "arguments": args,
                            "call_id": call_id,
                        }
                    )

        # Build content items for user / assistant messages
        content = message.get("content", "")
        content_items: List[Dict[str, Any]] = []

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text") or part.get("content") or ""
                    if isinstance(text, str) and text:
                        kind = "output_text" if role == "assistant" else "input_text"
                        content_items.append({"type": kind, "text": text})
                elif ptype == "image_url":
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else image
                    if isinstance(url, str) and url:
                        content_items.append({"type": "input_image", "image_url": url})
        elif isinstance(content, str) and content:
            kind = "output_text" if role == "assistant" else "input_text"
            content_items.append({"type": kind, "text": content})

        if not content_items:
            continue

        role_out = "assistant" if role == "assistant" else "user"
        input_items.append(
            {"type": "message", "role": role_out, "content": content_items}
        )

    return instructions, input_items


def convert_tools_chat_to_responses(
    tools: Any,
) -> List[Dict[str, Any]]:
    """Convert Chat Completions tool definitions to Responses API format."""
    out: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "name": name,
                "description": desc or "",
                "strict": False,
                "parameters": params,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main adapter function
# ---------------------------------------------------------------------------


async def chatgpt_chat_completion(
    messages: list[dict],
    model: str,
    access_token: str,
    account_id: str,
    *,
    tools: list[dict] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Send a Chat Completions-style request through the ChatGPT backend.

    Translates messages to Responses API format, sends to chatgpt.com,
    collects the streamed response, and returns a standard Chat Completion
    response dict.
    """
    normalised_model = normalize_model_name(model)
    instructions, input_items = convert_chat_messages_to_responses_input(messages)
    responses_tools = convert_tools_chat_to_responses(tools) if tools else []

    payload = {
        "model": normalised_model,
        "instructions": instructions,
        "input": input_items,
        "tools": responses_tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,  # ChatGPT backend REQUIRES streaming
        "include": ["reasoning.encrypted_content"],
        "reasoning": {
            "effort": "low",
            "summary": "auto",
        },
    }

    # Note: ChatGPT Responses API does NOT support temperature parameter

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
    }

    logger.debug(
        f"ChatGPT adapter: sending request to Responses API "
        f"(model={normalised_model}, input_items={len(input_items)}, "
        f"tools={len(responses_tools)})"
    )

    # --- Send streamed request and collect the full response ----------------
    full_text = ""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    tool_calls: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    usage_obj: Optional[Dict[str, int]] = None
    created = int(time.time())

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream(
                "POST",
                CHATGPT_RESPONSES_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    body_str = body.decode("utf-8", errors="ignore")
                    logger.error(f"ChatGPT adapter: raw error body: {body_str[:2000]}")
                    logger.error(f"ChatGPT adapter: sent payload model={normalised_model}, input_items={len(input_items)}, instructions_len={len(instructions) if instructions else 0}")
                    try:
                        err_body = json.loads(body_str)
                        err_msg = (
                            err_body.get("error", {}).get("message")
                            or f"Upstream error (HTTP {response.status_code})"
                        )
                    except Exception:
                        err_msg = f"Upstream error (HTTP {response.status_code})"
                    logger.error(f"ChatGPT adapter: upstream returned {response.status_code}: {err_msg}")
                    raise RuntimeError(err_msg)

                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    try:
                        evt = json.loads(data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    kind = evt.get("type")

                    # Capture response ID
                    if (
                        isinstance(evt.get("response"), dict)
                        and isinstance(evt["response"].get("id"), str)
                    ):
                        response_id = evt["response"]["id"]

                    # Collect output text
                    if kind == "response.output_text.delta":
                        full_text += evt.get("delta") or ""

                    # Collect tool calls
                    elif kind == "response.output_item.done":
                        item = evt.get("item") or {}
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id") or item.get("id") or ""
                            name = item.get("name") or ""
                            args = item.get("arguments") or ""
                            if (
                                isinstance(call_id, str)
                                and isinstance(name, str)
                                and isinstance(args, str)
                            ):
                                tool_calls.append(
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": args,
                                        },
                                    }
                                )

                    # Capture usage from completed event
                    elif kind == "response.completed":
                        usage_obj = _extract_usage(evt)
                        break

                    # Handle failures
                    elif kind == "response.failed":
                        error_message = (
                            evt.get("response", {})
                            .get("error", {})
                            .get("message", "response.failed")
                        )
                        break

    except httpx.HTTPError as exc:
        logger.error(f"ChatGPT adapter: HTTP error: {exc}")
        raise RuntimeError(f"ChatGPT backend request failed: {exc}") from exc

    if error_message:
        logger.error(f"ChatGPT adapter: response failed: {error_message}")
        raise RuntimeError(f"ChatGPT response failed: {error_message}")

    # --- Build standard Chat Completion response ----------------------------
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": full_text if full_text else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"

    completion: Dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,  # echo back the requested model name
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage_obj:
        completion["usage"] = usage_obj

    logger.debug(
        f"ChatGPT adapter: response collected "
        f"(content_len={len(full_text)}, tool_calls={len(tool_calls)})"
    )

    return completion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_usage(evt: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract usage information from a response.completed event."""
    try:
        usage = (evt.get("response") or {}).get("usage")
        if not isinstance(usage, dict):
            return None
        pt = int(usage.get("input_tokens") or 0)
        ct = int(usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except Exception:
        return None
