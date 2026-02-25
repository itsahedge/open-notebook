"""
Credential domain model for storing individual provider credentials.

Each credential is a standalone record in the 'credential' table, replacing
the old ProviderConfig singleton. Credentials store API keys (encrypted at
rest) and provider-specific configuration fields.

Usage:
    cred = Credential(
        name="Production",
        provider="openai",
        modalities=["language", "embedding"],
        api_key=SecretStr("sk-..."),
    )
    await cred.save()
"""

from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Dict, List, Optional

from loguru import logger
from pydantic import SecretStr

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel
from open_notebook.utils.encryption import decrypt_value, encrypt_value


class Credential(ObjectModel):
    """
    Individual credential record for an AI provider.

    Each record stores authentication and configuration for a single provider
    account. Models link to credentials via the credential field.
    """

    table_name: ClassVar[str] = "credential"
    nullable_fields: ClassVar[set[str]] = {
        "api_key",
        "base_url",
        "endpoint",
        "api_version",
        "endpoint_llm",
        "endpoint_embedding",
        "endpoint_stt",
        "endpoint_tts",
        "project",
        "location",
        "credentials_path",
        "oauth_access_token",
        "oauth_refresh_token",
        "oauth_token_expiry",
        "oauth_provider",
        "oauth_client_id",
        "oauth_base_url",
    }

    name: str
    provider: str
    modalities: List[str] = []
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    endpoint_llm: Optional[str] = None
    endpoint_embedding: Optional[str] = None
    endpoint_stt: Optional[str] = None
    endpoint_tts: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None
    credentials_path: Optional[str] = None

    # OAuth fields
    auth_type: str = "api_key"  # "api_key" or "oauth"
    oauth_access_token: Optional[SecretStr] = None
    oauth_refresh_token: Optional[SecretStr] = None
    oauth_token_expiry: Optional[datetime] = None
    oauth_provider: Optional[str] = None  # "openai", "anthropic", etc.
    oauth_client_id: Optional[str] = None
    oauth_base_url: Optional[str] = None  # base URL for OAuth-authenticated requests

    def to_esperanto_config(self) -> Dict[str, Any]:
        """
        Build config dict for AIFactory.create_*() calls.

        Returns a dict that can be passed as the 'config' parameter to
        Esperanto's AIFactory methods, overriding env var lookup.

        For OAuth credentials, uses the OAuth access token as api_key and
        sets base_url to the OAuth-specific endpoint.
        """
        config: Dict[str, Any] = {}

        if self.auth_type == "oauth" and self.oauth_access_token:
            # OAuth: use access token as API key, use OAuth base URL
            config["api_key"] = self.oauth_access_token.get_secret_value()
            if self.oauth_base_url:
                config["base_url"] = self.oauth_base_url
        elif self.api_key:
            config["api_key"] = self.api_key.get_secret_value()

        if self.base_url and "base_url" not in config:
            config["base_url"] = self.base_url
        if self.endpoint:
            config["endpoint"] = self.endpoint
        if self.api_version:
            config["api_version"] = self.api_version
        if self.endpoint_llm:
            config["endpoint_llm"] = self.endpoint_llm
        if self.endpoint_embedding:
            config["endpoint_embedding"] = self.endpoint_embedding
        if self.endpoint_stt:
            config["endpoint_stt"] = self.endpoint_stt
        if self.endpoint_tts:
            config["endpoint_tts"] = self.endpoint_tts
        if self.project:
            config["project"] = self.project
        if self.location:
            config["location"] = self.location
        if self.credentials_path:
            config["credentials_path"] = self.credentials_path
        return config

    @classmethod
    async def get_by_provider(cls, provider: str) -> List["Credential"]:
        """Get all credentials for a provider."""
        results = await repo_query(
            "SELECT * FROM credential WHERE string::lowercase(provider) = string::lowercase($provider) ORDER BY created ASC",
            {"provider": provider},
        )
        credentials = []
        for row in results:
            try:
                cred = cls._from_db_row(row)
                credentials.append(cred)
            except Exception as e:
                logger.warning(f"Skipping invalid credential: {e}")
        return credentials

    @classmethod
    async def get(cls, id: str) -> "Credential":
        """Override get() to handle api_key and OAuth token decryption."""
        instance = await super().get(id)
        # Decrypt api_key
        if instance.api_key:
            raw = (
                instance.api_key.get_secret_value()
                if isinstance(instance.api_key, SecretStr)
                else instance.api_key
            )
            decrypted = decrypt_value(raw)
            object.__setattr__(instance, "api_key", SecretStr(decrypted))
        # Decrypt OAuth tokens
        for token_field in ("oauth_access_token", "oauth_refresh_token"):
            token_val = getattr(instance, token_field, None)
            if token_val:
                raw = (
                    token_val.get_secret_value()
                    if isinstance(token_val, SecretStr)
                    else token_val
                )
                decrypted = decrypt_value(raw)
                object.__setattr__(instance, token_field, SecretStr(decrypted))
        return instance

    @classmethod
    async def get_all(cls, order_by=None) -> List["Credential"]:
        """Override get_all() to handle api_key and OAuth token decryption."""
        instances = await super().get_all(order_by=order_by)
        for instance in instances:
            if instance.api_key:
                raw = (
                    instance.api_key.get_secret_value()
                    if isinstance(instance.api_key, SecretStr)
                    else instance.api_key
                )
                decrypted = decrypt_value(raw)
                object.__setattr__(instance, "api_key", SecretStr(decrypted))
            # Decrypt OAuth tokens
            for token_field in ("oauth_access_token", "oauth_refresh_token"):
                token_val = getattr(instance, token_field, None)
                if token_val:
                    raw = (
                        token_val.get_secret_value()
                        if isinstance(token_val, SecretStr)
                        else token_val
                    )
                    decrypted = decrypt_value(raw)
                    object.__setattr__(instance, token_field, SecretStr(decrypted))
        return instances

    async def get_linked_models(self) -> list:
        """Get all models linked to this credential."""
        if not self.id:
            return []
        from open_notebook.ai.models import Model

        results = await repo_query(
            "SELECT * FROM model WHERE credential = $cred_id",
            {"cred_id": ensure_record_id(self.id)},
        )
        return [Model(**row) for row in results]

    _SECRET_FIELDS = {"api_key", "oauth_access_token", "oauth_refresh_token"}

    def _prepare_save_data(self) -> Dict[str, Any]:
        """Override to encrypt api_key and OAuth tokens before storage."""
        data = {}
        for key, value in self.model_dump().items():
            if key in self._SECRET_FIELDS:
                # Handle SecretStr fields: extract, encrypt, store
                field_val = getattr(self, key, None)
                if field_val:
                    secret_value = field_val.get_secret_value()
                    data[key] = encrypt_value(secret_value)
                else:
                    data[key] = None
            elif value is not None or key in self.__class__.nullable_fields:
                data[key] = value

        return data

    async def save(self) -> None:
        """Save credential, handling secret field re-hydration after DB round-trip."""
        # Remember the original SecretStr values before save
        originals = {
            field: getattr(self, field, None)
            for field in self._SECRET_FIELDS
        }

        await super().save()

        # After save, secret fields may be set to encrypted strings from
        # the DB result. Restore the original SecretStr values.
        for field, original in originals.items():
            if original:
                object.__setattr__(self, field, original)
            else:
                current = getattr(self, field, None)
                if current and isinstance(current, str):
                    decrypted = decrypt_value(current)
                    object.__setattr__(self, field, SecretStr(decrypted))

    @classmethod
    def _from_db_row(cls, row: dict) -> "Credential":
        """Create a Credential from a database row, decrypting secret fields."""
        for field in ("api_key", "oauth_access_token", "oauth_refresh_token"):
            val = row.get(field)
            if val and isinstance(val, str):
                decrypted = decrypt_value(val)
                row[field] = SecretStr(decrypted)
            elif val is None:
                row[field] = None
        return cls(**row)

    def is_oauth_token_expired(self) -> bool:
        """Check if the OAuth access token has expired or is near expiry."""
        if self.auth_type != "oauth" or not self.oauth_token_expiry:
            return False
        # Consider expired if within 5 minutes of expiry
        buffer = timedelta(minutes=5)
        return datetime.now(timezone.utc) >= self.oauth_token_expiry - buffer
