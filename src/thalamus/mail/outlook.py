"""Microsoft sign-in (OAuth device code flow) for Outlook, Hotmail and Live mail over IMAP.

Microsoft no longer accepts passwords for mail apps. The user registers a free app once (Azure
portal, public client flows allowed) and signs in here with a short code. MSAL keeps a refresh
token, stored with the other secrets, and renews access silently.

The sign-in authority comes from the address: personal accounts (outlook.com, hotmail, live, msn)
use /consumers, and work or school accounts use their own domain as the tenant. The generic
/common endpoint can't do device-code sign-in for a work account (AADSTS50059).
"""

from __future__ import annotations

from thalamus.mail.secrets import Secrets

SCOPES = ["https://outlook.office.com/IMAP.AccessAsUser.All"]
LOGIN = "https://login.microsoftonline.com"
PERSONAL_DOMAINS = ("outlook.", "hotmail.", "live.", "msn.com", "passport.com")

# Sign-in errors people actually hit, in words they can act on.
HINTS = {
    "AADSTS50059": "Microsoft couldn't tell which organization this account belongs to. Check the email address.",
    "AADSTS700016": (
        "This client ID isn't available to your account. In Azure, open the app → Authentication and set "
        "Supported account types to \"Accounts in any organizational directory and personal Microsoft accounts\"."
    ),
    "AADSTS50194": (
        "The app is set to one organization only. In Azure, open the app → Authentication and choose "
        "\"Accounts in any organizational directory and personal Microsoft accounts\"."
    ),
    "AADSTS7000218": "In Azure, open the app → Authentication and set Allow public client flows to Yes.",
    "AADSTS65001": "The app needs permission. In Azure, open API permissions and add IMAP.AccessAsUser.All.",
    "AADSTS90094": (
        "Your organization requires an administrator to approve new apps. Ask your IT admin to grant consent "
        "for THALAMUS (IMAP.AccessAsUser.All), or use a personal account."
    ),
    "AADSTS65004": "Sign-in was declined. Start again and click Accept on Microsoft's permission page.",
}


def authority_for(address: str) -> str:
    domain = address.rpartition("@")[2].strip().lower()
    if not domain or domain.startswith(PERSONAL_DOMAINS):
        return f"{LOGIN}/consumers"
    return f"{LOGIN}/{domain}"


def explain(result: dict, fallback: str) -> str:
    text = result.get("error_description") or fallback
    for code, hint in HINTS.items():
        if code in text:
            return f"{hint} ({code})"
    return text.split(" Trace ID:")[0]


class OutlookAuth:
    def __init__(self, secrets: Secrets) -> None:
        self._secrets = secrets

    def _app(self, client_id: str, cache_name: str, address: str):
        import msal

        cache = msal.SerializableTokenCache()
        saved = self._secrets.get(cache_name)
        if saved:
            cache.deserialize(saved)
        return msal.PublicClientApplication(client_id, authority=authority_for(address), token_cache=cache), cache

    def start(self, client_id: str, address: str) -> dict:
        app, _ = self._app(client_id, "outlook:pending", address)
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(explain(flow, "Microsoft sign-in couldn't start. Check the client ID."))
        return flow

    def finish(self, client_id: str, flow: dict, cache_name: str, address: str) -> str:
        """Blocks until the user signs in (or the code expires). Returns the signed-in address."""
        app, cache = self._app(client_id, cache_name, address)
        result = app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(explain(result, "Microsoft sign-in didn't complete."))
        self._secrets.set(cache_name, cache.serialize())
        claims = result.get("id_token_claims") or {}
        return claims.get("preferred_username") or claims.get("email") or ""

    def token(self, client_id: str, cache_name: str, address: str) -> str:
        app, cache = self._app(client_id, cache_name, address)
        accounts = app.get_accounts()
        result = app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        if not result or "access_token" not in result:
            raise RuntimeError("Microsoft sign-in has expired. Sign in again in Settings → Email accounts.")
        if cache.has_state_changed:
            self._secrets.set(cache_name, cache.serialize())
        return result["access_token"]
