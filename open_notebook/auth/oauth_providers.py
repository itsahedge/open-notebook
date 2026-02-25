"""
OAuth provider abstraction for Open Notebook.

Provides a provider-agnostic OAuth system with PKCE support for secure
token exchange. Currently supports OpenAI and Anthropic, designed for
easy extension to additional providers.

OpenAI OAuth has TWO paths after authentication:

1. **API key path**: After getting id_token via OAuth, attempt a token-exchange
   grant to obtain a real API key. If successful, use the API key against
   api.openai.com as normal.

2. **ChatGPT backend path** (fallback for ChatGPT-only accounts): Use the
   OAuth access_token as Bearer auth against a completely different endpoint
   (https://chatgpt.com/backend-api/). Requires a chatgpt-account-id header
   extracted from JWT claims.

The constants (client_id, scopes, URLs) match the Codex CLI / EchoNotes
reference implementations.

Usage:
    from open_notebook.auth.oauth_providers import (
        get_oauth_provider,
        build_authorize_url,
        exchange_code_for_tokens,
        refresh_access_token,
        exchange_token_for_api_key,
        extract_account_id_from_jwt,
    )

    provider = get_oauth_provider("openai")
    url = build_authorize_url("openai", redirect_uri, state, code_challenge)
"""

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from loguru import logger


# =============================================================================
# OpenAI OAuth Constants (matches Codex CLI)
# =============================================================================

OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_ISSUER = "https://auth.openai.com"
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_SCOPES = ["openid", "profile", "email", "offline_access"]
OPENAI_CALLBACK_PORT = 1455
OPENAI_REDIRECT_URI = f"http://localhost:{OPENAI_CALLBACK_PORT}/auth/callback"

# The standard API base URL (used when API key exchange succeeds)
OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# The ChatGPT backend URL (used when API key exchange fails — ChatGPT-only accounts)
OPENAI_CHATGPT_BASE_URL = "https://chatgpt.com/backend-api"

# Extra authorize params that match Codex CLI behavior
OPENAI_EXTRA_AUTH_PARAMS = {
    "id_token_add_organizations": "true",
    "codex_cli_simplified_flow": "true",
    "originator": "codex_cli_rs",
}


@dataclass
class OAuthProviderConfig:
    """Configuration for an OAuth provider."""

    provider: str
    auth_url: str
    token_url: str
    scopes: list[str]
    api_base_url: str  # Default base URL for API calls
    chatgpt_base_url: str = ""  # ChatGPT backend URL (OpenAI fallback path)
    use_pkce: bool = True
    client_id_env: str = ""
    client_secret_env: str = ""
    default_client_id: str = ""  # Hard-coded client_id (e.g., Codex CLI public ID)
    extra_auth_params: Dict[str, str] = field(default_factory=dict)
    supports_api_key_exchange: bool = False  # Whether provider supports token→API key

    @property
    def client_id(self) -> Optional[str]:
        """Client ID: env var takes precedence, then default (public) client_id."""
        env_val = os.environ.get(self.client_id_env) if self.client_id_env else None
        return env_val or self.default_client_id or None

    @property
    def client_secret(self) -> Optional[str]:
        return os.environ.get(self.client_secret_env) if self.client_secret_env else None


OAUTH_PROVIDERS: Dict[str, OAuthProviderConfig] = {
    "openai": OAuthProviderConfig(
        provider="openai",
        auth_url=OPENAI_AUTH_URL,
        token_url=OPENAI_TOKEN_URL,
        scopes=OPENAI_SCOPES,
        api_base_url=OPENAI_API_BASE_URL,
        chatgpt_base_url=OPENAI_CHATGPT_BASE_URL,
        use_pkce=True,
        client_id_env="OPENAI_OAUTH_CLIENT_ID",
        client_secret_env="OPENAI_OAUTH_CLIENT_SECRET",
        default_client_id=OPENAI_CLIENT_ID,
        extra_auth_params=OPENAI_EXTRA_AUTH_PARAMS,
        supports_api_key_exchange=True,
    ),
    "anthropic": OAuthProviderConfig(
        provider="anthropic",
        # Placeholder URLs — Anthropic OAuth is not yet publicly available
        auth_url="https://auth.anthropic.com/oauth/authorize",
        token_url="https://auth.anthropic.com/oauth/token",
        scopes=["claude.chat"],
        api_base_url="https://api.anthropic.com/v1",
        use_pkce=True,
        client_id_env="ANTHROPIC_OAUTH_CLIENT_ID",
        client_secret_env="ANTHROPIC_OAUTH_CLIENT_SECRET",
    ),
}


def get_oauth_provider(provider: str) -> OAuthProviderConfig:
    """
    Get OAuth configuration for a provider.

    Args:
        provider: Provider name (openai, anthropic)

    Returns:
        OAuthProviderConfig for the provider

    Raises:
        ValueError: If provider is not supported or not configured
    """
    config = OAUTH_PROVIDERS.get(provider.lower())
    if not config:
        supported = ", ".join(OAUTH_PROVIDERS.keys())
        raise ValueError(
            f"OAuth provider '{provider}' is not supported. "
            f"Supported providers: {supported}"
        )
    if not config.client_id:
        raise ValueError(
            f"OAuth client ID not configured for {provider}. "
            f"Set the {config.client_id_env} environment variable or "
            f"ensure a default_client_id is set."
        )
    return config


def generate_pkce_pair() -> tuple[str, str]:
    """
    Generate a PKCE code verifier and code challenge pair (S256).

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_authorize_url(
    provider: str,
    redirect_uri: str,
    state: str,
    code_challenge: Optional[str] = None,
) -> str:
    """
    Build the OAuth authorization URL for a provider.

    Args:
        provider: Provider name
        redirect_uri: OAuth callback URL
        state: CSRF protection state parameter
        code_challenge: PKCE code challenge (required if provider uses PKCE)

    Returns:
        Full authorization URL to redirect the user to
    """
    config = get_oauth_provider(provider)

    params: Dict[str, str] = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(config.scopes),
        "state": state,
    }

    if config.use_pkce and code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    params.update(config.extra_auth_params)

    return f"{config.auth_url}?{urlencode(params)}"


async def exchange_code_for_tokens(
    provider: str,
    code: str,
    redirect_uri: str,
    code_verifier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exchange an authorization code for access and refresh tokens.

    Args:
        provider: Provider name
        code: Authorization code from OAuth callback
        redirect_uri: Same redirect URI used in the authorization request
        code_verifier: PKCE code verifier (required if provider uses PKCE)

    Returns:
        Dict with keys: access_token, refresh_token, expires_in, token_type,
        id_token (for OpenAI/OIDC), and any additional provider-specific fields

    Raises:
        ValueError: If the token exchange fails
    """
    config = get_oauth_provider(provider)

    data: Dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": config.client_id,
    }

    if config.client_secret:
        data["client_secret"] = config.client_secret

    if config.use_pkce and code_verifier:
        data["code_verifier"] = code_verifier

    logger.info(f"Exchanging OAuth code for tokens with {provider}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config.token_url,
            data=data,
            headers={"Accept": "application/json"},
        )

    if response.status_code != 200:
        error_detail = response.text
        logger.error(
            f"OAuth token exchange failed for {provider}: "
            f"{response.status_code} - {error_detail}"
        )
        raise ValueError(
            f"OAuth token exchange failed for {provider}: {error_detail}"
        )

    token_data = response.json()
    logger.info(f"Successfully obtained OAuth tokens for {provider}")
    return token_data


async def exchange_token_for_api_key(
    provider: str,
    id_token: str,
) -> Optional[str]:
    """
    Attempt to exchange an id_token for a real API key via token-exchange grant.

    This is specific to OpenAI: after the initial OAuth code exchange returns
    an id_token, we can try a second token-exchange grant to get a real API key
    that works against api.openai.com. This succeeds for accounts with API
    access (pay-as-you-go) but fails for ChatGPT-only subscriptions.

    Args:
        provider: Provider name (only "openai" is supported)
        id_token: The id_token from the initial OAuth code exchange

    Returns:
        The API key string if exchange succeeded, or None if it failed
        (indicating a ChatGPT-only account that must use the backend path)
    """
    config = get_oauth_provider(provider)

    if not config.supports_api_key_exchange:
        logger.debug(f"Provider {provider} does not support API key exchange")
        return None

    data: Dict[str, str] = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "requested_token_type": "openai-api-key",
        "subject_token": id_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        "client_id": config.client_id,
    }

    if config.client_secret:
        data["client_secret"] = config.client_secret

    logger.info(f"Attempting id_token → API key exchange for {provider}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )

        if response.status_code == 200:
            result = response.json()
            api_key = result.get("access_token")
            if api_key:
                logger.info(
                    f"Successfully exchanged id_token for API key ({provider})"
                )
                return api_key
            logger.warning(
                f"Token exchange succeeded but no access_token in response ({provider})"
            )
            return None

        # Non-200 = account doesn't have API access (ChatGPT-only)
        logger.info(
            f"API key exchange failed for {provider} "
            f"(status {response.status_code}): likely a ChatGPT-only account. "
            f"Will use ChatGPT backend path."
        )
        return None

    except Exception as e:
        logger.warning(
            f"API key exchange request failed for {provider}: {e}. "
            f"Will fall back to ChatGPT backend path."
        )
        return None


def extract_account_id_from_jwt(id_token: str) -> Optional[str]:
    """
    Extract the chatgpt_account_id from an OpenAI id_token JWT.

    The id_token is a standard JWT. We decode the payload (middle segment)
    without signature verification (we trust it — it came from the token
    endpoint over TLS). The claim we need is in the organization info
    added by id_token_add_organizations=true.

    The JWT payload may contain:
    - "https://api.openai.com/auth": {"chatgpt_account_id": "..."}
    - or organizations list with account IDs

    Args:
        id_token: The raw JWT id_token string

    Returns:
        The chatgpt_account_id string, or None if not found
    """
    try:
        # JWT is header.payload.signature — we only need the payload
        parts = id_token.split(".")
        if len(parts) < 2:
            logger.warning("Invalid JWT format: not enough segments")
            return None

        # Base64url decode the payload (add padding)
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes)

        # Try the auth claims namespace first
        auth_claims = claims.get("https://api.openai.com/auth", {})
        if isinstance(auth_claims, dict):
            account_id = auth_claims.get("chatgpt_account_id")
            if account_id:
                logger.debug(f"Found chatgpt_account_id in auth claims: {account_id[:8]}...")
                return account_id

        # Fall back to organizations array
        orgs = claims.get("https://api.openai.com/organizations", [])
        if isinstance(orgs, list):
            for org in orgs:
                if isinstance(org, dict):
                    # Look for a personal org or first org with an account ID
                    account_id = org.get("chatgpt_account_id")
                    if account_id:
                        logger.debug(
                            f"Found chatgpt_account_id in organizations: {account_id[:8]}..."
                        )
                        return account_id

        logger.warning(
            "Could not find chatgpt_account_id in JWT claims. "
            "Available top-level keys: " + ", ".join(claims.keys())
        )
        return None

    except Exception as e:
        logger.error(f"Failed to decode JWT for account ID extraction: {e}")
        return None


async def refresh_access_token(
    provider: str,
    refresh_token: str,
) -> Dict[str, Any]:
    """
    Refresh an access token using a refresh token.

    Args:
        provider: Provider name
        refresh_token: The refresh token to use

    Returns:
        Dict with keys: access_token, refresh_token (if rotated),
        expires_in, token_type

    Raises:
        ValueError: If the token refresh fails
    """
    config = get_oauth_provider(provider)

    data: Dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": config.client_id,
    }

    if config.client_secret:
        data["client_secret"] = config.client_secret

    logger.info(f"Refreshing OAuth access token for {provider}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config.token_url,
            data=data,
            headers={"Accept": "application/json"},
        )

    if response.status_code != 200:
        error_detail = response.text
        logger.error(
            f"OAuth token refresh failed for {provider}: "
            f"{response.status_code} - {error_detail}"
        )
        raise ValueError(
            f"OAuth token refresh failed for {provider}: {error_detail}"
        )

    token_data = response.json()
    logger.info(f"Successfully refreshed OAuth access token for {provider}")
    return token_data
