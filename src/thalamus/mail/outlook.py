"""Microsoft sign-in (OAuth device code flow) for Outlook, Hotmail and Live mail over IMAP.

Microsoft no longer accepts passwords for mail apps. The user registers a free app once (Azure
portal, personal Microsoft accounts, public client flows allowed) and signs in here with a short
code. MSAL keeps a refresh token, stored with the other secrets, and renews access silently.
"""

from __future__ import annotations

from thalamus.mail.secrets import Secrets

SCOPES = ["https://outlook.office.com/IMAP.AccessAsUser.All"]
AUTHORITY = "https://login.microsoftonline.com/common"


class OutlookAuth:
    def __init__(self, secrets: Secrets) -> None:
        self._secrets = secrets

    def _app(self, client_id: str, cache_name: str):
        import msal

        cache = msal.SerializableTokenCache()
        saved = self._secrets.get(cache_name)
        if saved:
            cache.deserialize(saved)
        return msal.PublicClientApplication(client_id, authority=AUTHORITY, token_cache=cache), cache

    def start(self, client_id: str) -> dict:
        app, _ = self._app(client_id, "outlook:pending")
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(flow.get("error_description", "Microsoft sign-in couldn't start. Check the client ID."))
        return flow

    def finish(self, client_id: str, flow: dict, cache_name: str) -> str:
        """Blocks until the user signs in (or the code expires). Returns the signed-in address."""
        app, cache = self._app(client_id, cache_name)
        result = app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(result.get("error_description", "Microsoft sign-in didn't complete."))
        self._secrets.set(cache_name, cache.serialize())
        claims = result.get("id_token_claims") or {}
        return claims.get("preferred_username") or claims.get("email") or ""

    def token(self, client_id: str, cache_name: str) -> str:
        app, cache = self._app(client_id, cache_name)
        accounts = app.get_accounts()
        result = app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        if not result or "access_token" not in result:
            raise RuntimeError("Microsoft sign-in has expired. Sign in again in Settings → Email accounts.")
        if cache.has_state_changed:
            self._secrets.set(cache_name, cache.serialize())
        return result["access_token"]
