"""
LangChain-compatible ChatModel for Anthropic OAuth/setup-token authentication.

Claude Code setup-tokens (sk-ant-oat01-...) require:
1. Authorization: Bearer (not x-api-key)
2. anthropic-beta: claude-code-20250219,oauth-2025-04-20,...
3. Claude Code identity headers

This module provides:
- ``ChatAnthropicOAuth`` — subclass of ``ChatAnthropic`` that overrides
  client creation to use ``authToken`` (Bearer) instead of ``apiKey``.
- ``AnthropicOAuthLanguageModelWrapper`` — Esperanto-shaped wrapper for
  Open Notebook's model provisioning pipeline.

Reference: OpenClaw's pi-ai Anthropic provider (createClient function).
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from loguru import logger

from open_notebook.auth.anthropic_adapter import (
    CLAUDE_CODE_VERSION,
    OAUTH_BETA_FEATURES,
    is_anthropic_oauth_token,
)

# Headers required for OAuth token authentication
OAUTH_DEFAULT_HEADERS: Dict[str, str] = {
    "anthropic-beta": OAUTH_BETA_FEATURES,
    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
    "x-app": "cli",
    "accept": "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
}


class ChatAnthropicOAuth(ChatAnthropic):
    """
    LangChain ``ChatAnthropic`` subclass that uses Bearer auth for OAuth tokens.

    The standard ``ChatAnthropic`` passes the key as ``api_key`` to the
    ``anthropic.Client``, which sends it as ``x-api-key`` header.

    For OAuth/setup-tokens, we need ``authToken`` (``Authorization: Bearer``).
    This subclass overrides ``_client_params`` to swap the auth mechanism.
    """

    @property
    def _client_params(self) -> dict[str, Any]:
        """
        Build client params with authToken instead of api_key for OAuth tokens.
        """
        api_key_value = self.anthropic_api_key.get_secret_value()

        # Merge OAuth headers with any user-supplied headers
        merged_headers = {**OAUTH_DEFAULT_HEADERS}
        if self.default_headers:
            merged_headers.update(self.default_headers)

        client_params: dict[str, Any] = {
            # OAuth token → use authToken (Bearer) instead of api_key (x-api-key)
            "api_key": None,
            "auth_token": api_key_value,
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": merged_headers,
        }

        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout

        return client_params


class AnthropicOAuthLanguageModelWrapper:
    """
    Wrapper that makes an Anthropic OAuth model compatible with
    Open Notebook's Esperanto-based provisioning pipeline.

    Calling ``to_langchain()`` returns a ``ChatAnthropicOAuth`` instance
    configured with Bearer auth + Claude Code identity headers.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self._kwargs = kwargs

    def to_langchain(self):
        """Return a LangChain ChatAnthropicOAuth for Bearer-based auth."""
        logger.debug(
            f"Creating ChatAnthropicOAuth for model={self.model_name}"
        )
        return ChatAnthropicOAuth(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            anthropic_api_url="https://api.anthropic.com",
        )
