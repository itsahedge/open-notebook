"""
Anthropic OAuth adapter for Claude Code setup-tokens.

Claude Code setup-tokens (sk-ant-oat01-...) require special handling:
1. Authorization: Bearer (not x-api-key)
2. anthropic-beta: claude-code-20250219,oauth-2025-04-20
3. Claude Code identity headers (user-agent, x-app)
4. System prompt prefix: "You are Claude Code..."

This module provides:
- ``AnthropicOAuthLanguageModel`` — subclass of Esperanto's
  ``AnthropicLanguageModel`` that overrides auth headers for OAuth tokens.
- ``is_anthropic_oauth_token()`` — helper to detect setup-token format.

Reference: OpenClaw's pi-ai Anthropic provider (createClient function).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from esperanto.providers.llm.anthropic import AnthropicLanguageModel

# Claude Code identity constants (must match for OAuth token auth to work)
CLAUDE_CODE_VERSION = "2.1.2"
OAUTH_BETA_FEATURES = (
    "claude-code-20250219,"
    "oauth-2025-04-20,"
    "fine-grained-tool-streaming-2025-05-14,"
    "interleaved-thinking-2025-05-14"
)


def is_anthropic_oauth_token(key: str | None) -> bool:
    """Check if a key is an Anthropic OAuth/setup-token (vs standard API key)."""
    return bool(key and "sk-ant-oat" in key)


@dataclass
class AnthropicOAuthLanguageModel(AnthropicLanguageModel):
    """
    Anthropic language model using OAuth/setup-token authentication.

    Overrides header generation to use Bearer auth + Claude Code identity,
    which is required for setup-tokens to authenticate against the
    Anthropic Messages API.
    """

    def _get_headers(self) -> Dict[str, str]:
        """
        Build headers for OAuth-authenticated Anthropic API requests.

        Uses Authorization: Bearer instead of x-api-key, and includes
        the required anthropic-beta flags for OAuth token acceptance.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": OAUTH_BETA_FEATURES,
            "content-type": "application/json",
            "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
            "x-app": "cli",
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }
