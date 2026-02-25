"""
OAuth provider abstraction for Open Notebook.

Provides a provider-agnostic OAuth system with PKCE support for secure
token exchange. Currently supports OpenAI and Anthropic, designed for
easy extension to additional providers.

Usage:
    from open_notebook.auth.oauth_providers import (
        get_oauth_provider,
        build_authorize_url,
        exchange_code_for_tokens,
        refresh_access_token,
    )

    provider = get_oauth_provider("openai")
    url = build_authorize_url("openai", redirect_uri, state, code_challenge)
"""

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from loguru import logger


@dataclass
class OAuthProviderConfig:
    """Configuration for an OAuth provider."""

    provider: str
    auth_url: str
    token_url: str
    scopes: list[str]
    api_base_url: str
    use_pkce: bool = True
    client_id_env: str = ""
    client_secret_env: str = ""
    extra_auth_params: Dict[str, str] = field(default_factory=dict)

    @property
    def client_id(self) -> Optional[str]:
        return os.environ.get(self.client_id_env)

    @property
    def client_secret(self) -> Optional[str]:
        return os.environ.get(self.client_secret_env)


OAUTH_PROVIDERS: Dict[str, OAuthProviderConfig] = {
    "openai": OAuthProviderConfig(
        provider="openai",
        auth_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        scopes=["openai.chat", "openai.models"],
        api_base_url="https://api.openai.com/v1",
        use_pkce=True,
        client_id_env="OPENAI_OAUTH_CLIENT_ID",
        client_secret_env="OPENAI_OAUTH_CLIENT_SECRET",
    ),
    "anthropic": OAuthProviderConfig(
        provider="anthropic",
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
            f"Set the {config.client_id_env} environment variable."
        )
    return config


def generate_pkce_pair() -> tuple[str, str]:
    """
    Generate a PKCE code verifier and code challenge pair (S256).

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    import base64

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
        and any additional provider-specific fields

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
