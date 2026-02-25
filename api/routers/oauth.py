"""
OAuth Router

Handles OAuth authorization flows for AI providers (OpenAI, Anthropic, etc.).
Creates/updates Credential records with OAuth tokens.

OpenAI has TWO paths after authentication:

1. **API key path**: After getting id_token via OAuth, try token-exchange
   grant (urn:ietf:params:oauth:grant-type:token-exchange) to get a real
   API key. If successful → store as oauth_api_key, use against
   api.openai.com just like a normal key.

2. **ChatGPT backend path** (fallback for ChatGPT-only subscriptions):
   When API key exchange fails, use the OAuth access_token as Bearer auth
   against https://chatgpt.com/backend-api/. Requires chatgpt-account-id
   header extracted from the JWT id_token.

Endpoints:
- GET  /oauth/{provider}/authorize          - Get authorization URL
- POST /oauth/{provider}/callback           - Exchange code for tokens
- POST /oauth/{provider}/refresh/{cred_id}  - Manually refresh a token

NEVER returns actual tokens in API responses.
"""

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import SecretStr

from api.credentials_service import (
    credential_to_response,
    get_default_modalities,
    require_encryption_key,
)
from api.models import (
    CredentialResponse,
    OAuthAuthorizeResponse,
    OAuthCallbackRequest,
    OAuthTokenRefreshResponse,
)
from open_notebook.auth.oauth_providers import (
    build_authorize_url,
    exchange_code_for_tokens,
    exchange_token_for_api_key,
    extract_account_id_from_jwt,
    generate_pkce_pair,
    get_oauth_provider,
    refresh_access_token,
)
from open_notebook.domain.credential import Credential

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.get("/{provider}/authorize", response_model=OAuthAuthorizeResponse)
async def oauth_authorize(provider: str, redirect_uri: str):
    """
    Generate an OAuth authorization URL for a provider.

    The frontend should redirect the user to the returned URL. After the user
    authorizes, the provider will redirect back to redirect_uri with a code.
    """
    try:
        require_encryption_key()
        config = get_oauth_provider(provider)

        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = generate_pkce_pair()

        authorize_url = build_authorize_url(
            provider=provider,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge if config.use_pkce else None,
        )

        # The frontend must store code_verifier and send it back in the callback
        logger.info(f"Generated OAuth authorization URL for {provider}")

        return OAuthAuthorizeResponse(
            authorize_url=authorize_url,
            state=state,
            code_verifier=code_verifier if config.use_pkce else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating OAuth authorize URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate authorization URL")


@router.post("/{provider}/callback", response_model=CredentialResponse, status_code=201)
async def oauth_callback(provider: str, request: OAuthCallbackRequest):
    """
    Handle the OAuth callback by exchanging the authorization code for tokens.

    For OpenAI, this implements the two-path flow:

    1. Exchange authorization code for tokens (access_token, id_token, etc.)
    2. Try to exchange the id_token for a real API key (token-exchange grant)
       - If successful → store oauth_api_key, use standard api.openai.com
       - If fails → extract chatgpt_account_id from JWT, use ChatGPT backend

    Creates a new Credential record with auth_type="oauth" storing the
    encrypted tokens.
    """
    try:
        require_encryption_key()

        # Step 1: Exchange authorization code for tokens
        token_data = await exchange_code_for_tokens(
            provider=provider,
            code=request.code,
            redirect_uri=request.redirect_uri,
            code_verifier=request.code_verifier,
        )

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        id_token = token_data.get("id_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            raise ValueError("No access_token in OAuth response")

        # Compute token expiry
        token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        # Get the OAuth provider config
        config = get_oauth_provider(provider)

        # Step 2: For providers that support it, try API key exchange
        exchanged_api_key: str | None = None
        account_id: str | None = None
        oauth_base_url: str | None = None
        credential_name_suffix = "OAuth"

        if config.supports_api_key_exchange and id_token:
            exchanged_api_key = await exchange_token_for_api_key(
                provider=provider,
                id_token=id_token,
            )

        if exchanged_api_key:
            # Path 1: API key exchange succeeded → standard API path
            # No special base_url needed — uses default api.openai.com/v1
            credential_name_suffix = "OAuth - API Key"
            logger.info(
                f"OpenAI OAuth: API key exchange succeeded. "
                f"Using standard API path."
            )
        else:
            # Path 2: ChatGPT backend path (or non-OpenAI provider)
            oauth_base_url = config.chatgpt_base_url or config.api_base_url

            # Extract account_id from JWT for ChatGPT backend auth
            if id_token:
                account_id = extract_account_id_from_jwt(id_token)
                if account_id:
                    credential_name_suffix = "OAuth - ChatGPT"
                    logger.info(
                        f"OpenAI OAuth: using ChatGPT backend path "
                        f"(account_id={account_id[:8]}...)"
                    )
                else:
                    credential_name_suffix = "OAuth - ChatGPT (no account ID)"
                    logger.warning(
                        "OpenAI OAuth: ChatGPT backend path but "
                        "could not extract account_id from JWT"
                    )

        # Create a new Credential with OAuth tokens
        cred = Credential(
            name=f"{provider.title()} ({credential_name_suffix})",
            provider=provider,
            modalities=get_default_modalities(provider),
            auth_type="oauth",
            oauth_access_token=SecretStr(access_token),
            oauth_refresh_token=SecretStr(refresh_token) if refresh_token else None,
            oauth_token_expiry=token_expiry,
            oauth_provider=provider,
            oauth_client_id=config.client_id,
            oauth_base_url=oauth_base_url,
            oauth_api_key=SecretStr(exchanged_api_key) if exchanged_api_key else None,
            oauth_account_id=account_id,
        )

        await cred.save()
        logger.info(f"Created OAuth credential for {provider}: {cred.id}")

        return credential_to_response(cred)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OAuth callback error for {provider}: {e}")
        raise HTTPException(status_code=500, detail="OAuth token exchange failed")


@router.post(
    "/{provider}/refresh/{credential_id}",
    response_model=OAuthTokenRefreshResponse,
)
async def oauth_refresh(provider: str, credential_id: str):
    """
    Manually refresh an OAuth access token for a credential.

    This is typically not needed as token refresh happens automatically
    when using the credential, but can be useful for testing or
    pre-emptive refresh.

    After refreshing, re-attempts the API key exchange for providers that
    support it (the new id_token may grant API key access if the account
    was upgraded).
    """
    try:
        require_encryption_key()

        cred = await Credential.get(credential_id)
        if cred.auth_type != "oauth":
            raise ValueError("Credential is not an OAuth credential")
        if cred.provider.lower() != provider.lower():
            raise ValueError(f"Credential provider mismatch: {cred.provider} != {provider}")
        if not cred.oauth_refresh_token:
            raise ValueError("No refresh token available for this credential")

        token_data = await refresh_access_token(
            provider=provider,
            refresh_token=cred.oauth_refresh_token.get_secret_value(),
        )

        # Update credential with new tokens
        object.__setattr__(
            cred, "oauth_access_token", SecretStr(token_data["access_token"])
        )
        if "refresh_token" in token_data:
            object.__setattr__(
                cred, "oauth_refresh_token", SecretStr(token_data["refresh_token"])
            )

        expires_in = token_data.get("expires_in", 3600)
        new_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        object.__setattr__(cred, "oauth_token_expiry", new_expiry)

        # If refresh returned a new id_token, re-attempt API key exchange
        new_id_token = token_data.get("id_token")
        if new_id_token:
            config = get_oauth_provider(provider)
            if config.supports_api_key_exchange:
                new_api_key = await exchange_token_for_api_key(
                    provider=provider,
                    id_token=new_id_token,
                )
                if new_api_key:
                    object.__setattr__(cred, "oauth_api_key", SecretStr(new_api_key))
                    # Clear ChatGPT-specific base URL since we now have an API key
                    object.__setattr__(cred, "oauth_base_url", None)
                    logger.info(
                        f"API key exchange succeeded on refresh for {cred.id}"
                    )

            # Also re-extract account_id in case it changed
            account_id = extract_account_id_from_jwt(new_id_token)
            if account_id:
                object.__setattr__(cred, "oauth_account_id", account_id)

        await cred.save()

        return OAuthTokenRefreshResponse(
            success=True,
            expires_at=new_expiry.isoformat(),
            message="Token refreshed successfully",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OAuth refresh error for {provider}/{credential_id}: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")
