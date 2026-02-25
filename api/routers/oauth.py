"""
OAuth Router

Handles OAuth authorization flows for AI providers (OpenAI, Anthropic, etc.).
Creates/updates Credential records with OAuth tokens.

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

        # Store the code_verifier in the state for the callback
        # The frontend must send it back in the callback request
        logger.info(f"Generated OAuth authorization URL for {provider}")

        return OAuthAuthorizeResponse(
            authorize_url=authorize_url,
            state=state,
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

    Creates a new Credential record with auth_type="oauth" storing the
    encrypted tokens.
    """
    try:
        require_encryption_key()

        # Exchange code for tokens
        token_data = await exchange_code_for_tokens(
            provider=provider,
            code=request.code,
            redirect_uri=request.redirect_uri,
            code_verifier=request.code_verifier,
        )

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            raise ValueError("No access_token in OAuth response")

        # Compute token expiry
        token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        # Get the OAuth provider config for the base URL
        config = get_oauth_provider(provider)

        # Create a new Credential with OAuth tokens
        cred = Credential(
            name=f"{provider.title()} (OAuth)",
            provider=provider,
            modalities=get_default_modalities(provider),
            auth_type="oauth",
            oauth_access_token=SecretStr(access_token),
            oauth_refresh_token=SecretStr(refresh_token) if refresh_token else None,
            oauth_token_expiry=token_expiry,
            oauth_provider=provider,
            oauth_client_id=config.client_id,
            oauth_base_url=config.api_base_url,
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
