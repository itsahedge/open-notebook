"""
LangChain-compatible ChatModel backed by the ChatGPT Responses API adapter.

This module provides:
- ``ChatGPTChatModel`` — a LangChain ``BaseChatModel`` that translates
  ``invoke`` / ``ainvoke`` calls into ``chatgpt_chat_completion()`` calls.
- ``ChatGPTLanguageModelWrapper`` — a thin object that quacks like an
  Esperanto ``LanguageModel`` just enough for Open Notebook's provisioning
  pipeline (``to_langchain()`` and the isinstance check).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger


# ---------------------------------------------------------------------------
# LangChain-compatible chat model
# ---------------------------------------------------------------------------


class ChatGPTChatModel(BaseChatModel):
    """
    LangChain ``BaseChatModel`` backed by the ChatGPT Responses API.

    All calls are routed through ``chatgpt_chat_completion()`` from the
    adapter module. The ChatGPT backend requires streaming, which the
    adapter handles internally while returning a collected result.
    """

    model_name: str = "gpt-5"
    access_token: str = ""
    account_id: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "chatgpt-responses-api"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "account_id": self.account_id[:8] + "..." if self.account_id else "",
        }

    # -- helpers to convert LangChain messages → Chat Completions dicts -----

    @staticmethod
    def _messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain ``BaseMessage`` objects to plain dicts."""
        result: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                d: Dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["args"]
                                if isinstance(tc["args"], str)
                                else __import__("json").dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                # Fallback: treat as user message
                result.append({"role": "user", "content": str(msg.content)})
        return result

    # -- required LangChain interface ---------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation — runs the async adapter in a loop."""
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via the ChatGPT adapter."""
        from open_notebook.auth.chatgpt_adapter import chatgpt_chat_completion

        msg_dicts = self._messages_to_dicts(messages)

        completion = await chatgpt_chat_completion(
            messages=msg_dicts,
            model=self.model_name,
            access_token=self.access_token,
            account_id=self.account_id,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = completion["choices"][0]
        message_data = choice["message"]

        # Build AIMessage from the response
        content = message_data.get("content") or ""
        ai_msg_kwargs: Dict[str, Any] = {"content": content}

        # Handle tool calls
        if message_data.get("tool_calls"):
            tool_calls = []
            for tc in message_data["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    parsed_args = __import__("json").loads(args_str)
                except Exception:
                    parsed_args = {}
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "args": parsed_args,
                    }
                )
            ai_msg_kwargs["tool_calls"] = tool_calls

        ai_message = AIMessage(**ai_msg_kwargs)

        generation = ChatGeneration(message=ai_message)
        llm_output: Dict[str, Any] = {}
        if "usage" in completion:
            llm_output["token_usage"] = completion["usage"]
        llm_output["model_name"] = completion.get("model", self.model_name)

        return ChatResult(generations=[generation], llm_output=llm_output)


# ---------------------------------------------------------------------------
# Wrapper that looks like an Esperanto LanguageModel
# ---------------------------------------------------------------------------


class ChatGPTLanguageModelWrapper:
    """
    Thin wrapper around ``ChatGPTChatModel`` that provides a
    ``to_langchain()`` method, matching the interface expected by
    ``open_notebook.ai.provision.provision_langchain_model()``.

    This is **not** a real Esperanto ``LanguageModel`` subclass but
    carries enough of the interface for the provisioning pipeline.
    """

    def __init__(
        self,
        model_name: str,
        access_token: str,
        account_id: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ):
        self.model_name = model_name
        self._access_token = access_token
        self._account_id = account_id
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info(
            f"ChatGPT adapter wrapper created for model={model_name} "
            f"(account_id={account_id[:8]}...)"
        )

    def to_langchain(self) -> ChatGPTChatModel:
        """Return a LangChain-compatible chat model."""
        return ChatGPTChatModel(
            model_name=self.model_name,
            access_token=self._access_token,
            account_id=self._account_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    def get_model_name(self) -> str:
        return self.model_name

    async def achat_complete(self, messages: list[dict], **kwargs) -> Any:
        """
        Async chat completion matching Esperanto's ``LanguageModel`` interface.

        Used by the connection tester. Returns a simple object with a
        ``.content`` attribute.
        """
        from open_notebook.auth.chatgpt_adapter import chatgpt_chat_completion

        result = await chatgpt_chat_completion(
            messages=messages,
            model=self.model_name,
            access_token=self._access_token,
            account_id=self._account_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        # Return a simple namespace with .content like Esperanto's ChatCompletion
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""

        class _Response:
            pass

        resp = _Response()
        resp.content = content  # type: ignore[attr-defined]
        return resp

    # Provide a provider property so logging/debugging works
    @property
    def provider(self) -> str:
        return "chatgpt-backend"
