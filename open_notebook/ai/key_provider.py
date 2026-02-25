"""
API Key Provider - Database-first with environment fallback.

This module provides a unified interface for retrieving API keys and provider
configuration. It reads from Credential records (individual per-provider
credentials) and falls back to environment variables for backward compatibility.

For OAuth credentials, it automatically handles token refresh when tokens
are expired or near expiry.

Usage:
    from open_notebook.ai.key_provider import provision_provider_keys

    # Call before model provisioning to set env vars from DB
    await provision_provider_keys("openai")
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger
from pydantic import SecretStr

from open_notebook.domain.credential import Credential


# =============================================================================
# Provider Configuration Mapping
# =============================================================================
# Maps provider names to their environment variable names.
# This is the single source of truth for provider-to-env-var mapping.

PROVIDER_CONFIG = {
    # Simple providers (just API key)
    "openai": {
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
    },
    "google": {
        "env_var": "GOOGLE_API_KEY",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
    },
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
    },
    "xai": {
        "env_var": "XAI_API_KEY",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
    },
    "voyage": {
        "env_var": "VOYAGE_API_KEY",
    },
    "elevenlabs": {
        "env_var": "ELEVENLABS_API_KEY",
    },
    # URL-based providers
    "ollama": {
        "env_var": "OLLAMA_API_BASE",
    },
}


async def _get_default_credential(provider: str) -> Optional[Credential]:
    """Get the first credential for a provider from the database."""
    try:
        credentials = await Credential.get_by_provider(provider)
        if credentials:
            return credentials[0]
    except Exception as e:
        logger.debug(f"Could not load credential from database for {provider}: {e}")
    return None


async def _refresh_oauth_token_if_needed(cred: Credential) -> Credential:
    """
    Check if an OAuth credential's token is expired and refresh if needed.

    Args:
        cred: A Credential with auth_type == "oauth"

    Returns:
        The credential (updated in-place and saved if refreshed)
    """
    if not cred.is_oauth_token_expired():
        return cred

    if not cred.oauth_refresh_token:
        logger.warning(
            f"OAuth token expired for credential {cred.id} but no refresh token available"
        )
        return cred

    if not cred.oauth_provider:
        logger.warning(
            f"OAuth token expired for credential {cred.id} but no oauth_provider set"
        )
        return cred

    try:
        from open_notebook.auth.oauth_providers import refresh_access_token

        token_data = await refresh_access_token(
            cred.oauth_provider,
            cred.oauth_refresh_token.get_secret_value(),
        )

        # Update credential with new tokens
        object.__setattr__(
            cred, "oauth_access_token", SecretStr(token_data["access_token"])
        )

        # Some providers rotate refresh tokens
        if "refresh_token" in token_data:
            object.__setattr__(
                cred, "oauth_refresh_token", SecretStr(token_data["refresh_token"])
            )

        # Update expiry
        expires_in = token_data.get("expires_in", 3600)
        new_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        object.__setattr__(cred, "oauth_token_expiry", new_expiry)

        # Persist the refreshed tokens
        await cred.save()
        logger.info(f"Successfully refreshed OAuth token for credential {cred.id}")

    except Exception as e:
        logger.error(f"Failed to refresh OAuth token for credential {cred.id}: {e}")

    return cred


async def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a provider. Checks database first, then env var.

    For OAuth credentials, prefers the exchanged API key (oauth_api_key) if
    available (standard API path). Falls back to oauth_access_token for the
    ChatGPT backend path. Auto-refreshes expired tokens.

    Args:
        provider: Provider name (openai, anthropic, etc.)

    Returns:
        API key string or None if not configured
    """
    cred = await _get_default_credential(provider)
    if cred:
        # OAuth credential: two paths
        if cred.auth_type == "oauth":
            cred = await _refresh_oauth_token_if_needed(cred)

            # Path 1: exchanged API key (standard API endpoint)
            if cred.oauth_api_key:
                logger.debug(f"Using {provider} OAuth exchanged API key from Credential")
                return cred.oauth_api_key.get_secret_value()

            # Path 2: access token (ChatGPT backend or other OAuth provider)
            if cred.oauth_access_token:
                logger.debug(f"Using {provider} OAuth access token from Credential")
                return cred.oauth_access_token.get_secret_value()

        # API key credential
        if cred.api_key:
            logger.debug(f"Using {provider} API key from Credential")
            return cred.api_key.get_secret_value()

    # Fall back to environment variable
    config_info = PROVIDER_CONFIG.get(provider.lower())
    if config_info:
        env_value = os.environ.get(config_info["env_var"])
        if env_value:
            logger.debug(f"Using {provider} API key from environment variable")
        return env_value

    return None


async def _provision_simple_provider(provider: str) -> bool:
    """
    Set environment variable for a simple provider from DB config.

    Handles both API key and OAuth credentials.

    Returns:
        True if key was set from database, False otherwise
    """
    provider_lower = provider.lower()
    config_info = PROVIDER_CONFIG.get(provider_lower)
    if not config_info:
        return False

    env_var = config_info["env_var"]

    cred = await _get_default_credential(provider_lower)
    if not cred:
        return False

    # OAuth credential: two paths for API key resolution
    if cred.auth_type == "oauth":
        cred = await _refresh_oauth_token_if_needed(cred)

        if cred.oauth_api_key:
            # Path 1: exchanged API key → standard API endpoint
            os.environ[env_var] = cred.oauth_api_key.get_secret_value()
            logger.debug(f"Set {env_var} from OAuth exchanged API key")
            # Don't set a custom base URL — use the provider's default
        elif cred.oauth_access_token:
            # Path 2: access token → ChatGPT backend (or provider-specific URL)
            os.environ[env_var] = cred.oauth_access_token.get_secret_value()
            logger.debug(f"Set {env_var} from OAuth access token")
            if cred.oauth_base_url:
                provider_upper = provider_lower.upper()
                os.environ[f"{provider_upper}_API_BASE"] = cred.oauth_base_url
                logger.debug(f"Set {provider_upper}_API_BASE from OAuth Credential")
        return True

    # API key credential
    if cred.api_key:
        os.environ[env_var] = cred.api_key.get_secret_value()
        logger.debug(f"Set {env_var} from Credential")

    # Set base URL if present
    if cred.base_url:
        provider_upper = provider_lower.upper()
        os.environ[f"{provider_upper}_API_BASE"] = cred.base_url
        logger.debug(f"Set {provider_upper}_API_BASE from Credential")

    return True


async def _provision_vertex() -> bool:
    """
    Set environment variables for Google Vertex AI from DB config.

    Returns:
        True if any keys were set from database
    """
    any_set = False

    cred = await _get_default_credential("vertex")
    if not cred:
        return False

    if cred.project:
        os.environ["VERTEX_PROJECT"] = cred.project
        logger.debug("Set VERTEX_PROJECT from Credential")
        any_set = True
    if cred.location:
        os.environ["VERTEX_LOCATION"] = cred.location
        logger.debug("Set VERTEX_LOCATION from Credential")
        any_set = True
    if cred.credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred.credentials_path
        logger.debug("Set GOOGLE_APPLICATION_CREDENTIALS from Credential")
        any_set = True

    return any_set


async def _provision_azure() -> bool:
    """
    Set environment variables for Azure OpenAI from DB config.

    Returns:
        True if any keys were set from database
    """
    any_set = False

    cred = await _get_default_credential("azure")
    if not cred:
        return False

    if cred.api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = cred.api_key.get_secret_value()
        logger.debug("Set AZURE_OPENAI_API_KEY from Credential")
        any_set = True
    if cred.api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = cred.api_version
        logger.debug("Set AZURE_OPENAI_API_VERSION from Credential")
        any_set = True
    if cred.endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = cred.endpoint
        logger.debug("Set AZURE_OPENAI_ENDPOINT from Credential")
        any_set = True
    if cred.endpoint_llm:
        os.environ["AZURE_OPENAI_ENDPOINT_LLM"] = cred.endpoint_llm
        logger.debug("Set AZURE_OPENAI_ENDPOINT_LLM from Credential")
        any_set = True
    if cred.endpoint_embedding:
        os.environ["AZURE_OPENAI_ENDPOINT_EMBEDDING"] = cred.endpoint_embedding
        logger.debug("Set AZURE_OPENAI_ENDPOINT_EMBEDDING from Credential")
        any_set = True
    if cred.endpoint_stt:
        os.environ["AZURE_OPENAI_ENDPOINT_STT"] = cred.endpoint_stt
        logger.debug("Set AZURE_OPENAI_ENDPOINT_STT from Credential")
        any_set = True
    if cred.endpoint_tts:
        os.environ["AZURE_OPENAI_ENDPOINT_TTS"] = cred.endpoint_tts
        logger.debug("Set AZURE_OPENAI_ENDPOINT_TTS from Credential")
        any_set = True

    return any_set


async def _provision_openai_compatible() -> bool:
    """
    Set environment variables for OpenAI-Compatible providers from DB config.

    Returns:
        True if any keys were set from database
    """
    any_set = False

    cred = await _get_default_credential("openai_compatible")
    if not cred:
        return False

    if cred.api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = cred.api_key.get_secret_value()
        logger.debug("Set OPENAI_COMPATIBLE_API_KEY from Credential")
        any_set = True
    if cred.base_url:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = cred.base_url
        logger.debug("Set OPENAI_COMPATIBLE_BASE_URL from Credential")
        any_set = True

    return any_set


async def provision_provider_keys(provider: str) -> bool:
    """
    Provision environment variables from database for a specific provider.

    This function checks if the provider has a Credential record stored in the
    database and sets the corresponding environment variables. If the database
    doesn't have the configuration, existing environment variables remain unchanged.

    This is the main entry point for the DB->Env fallback mechanism.

    Args:
        provider: Provider name (openai, anthropic, azure, vertex,
                  openai-compatible, etc.)

    Returns:
        True if any keys were set from database, False otherwise

    Example:
        # Before provisioning a model, ensure DB keys are in env vars
        await provision_provider_keys("openai")
        model = AIFactory.create_language(model_name="gpt-4", provider="openai")
    """
    # Normalize provider name
    provider_lower = provider.lower()

    # Handle complex providers with multiple config fields
    if provider_lower == "vertex":
        return await _provision_vertex()
    elif provider_lower == "azure":
        return await _provision_azure()
    elif provider_lower in ("openai-compatible", "openai_compatible"):
        return await _provision_openai_compatible()

    # Handle simple providers
    return await _provision_simple_provider(provider_lower)


async def provision_all_keys() -> dict[str, bool]:
    """
    Provision environment variables from database for all providers.

    NOTE: This function is deprecated for request-time use because it can leave
    stale env vars after key deletion. Keys should only be provisioned at startup
    or via provision_provider_keys() for specific providers.

    Useful at application startup to load all DB-stored keys into environment.

    Returns:
        Dict mapping provider names to whether keys were set from DB
    """
    results: dict[str, bool] = {}

    # Simple providers
    for provider in PROVIDER_CONFIG.keys():
        results[provider] = await provision_provider_keys(provider)

    # Complex providers
    results["vertex"] = await provision_provider_keys("vertex")
    results["azure"] = await provision_provider_keys("azure")
    results["openai_compatible"] = await provision_provider_keys("openai_compatible")

    return results
