import base64
import logging
import textwrap
import time
import uuid
from urllib.parse import urlsplit

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

log = logging.getLogger(__name__)


class KalshiClient:
    """Synchronous Kalshi HTTP API client with RSA-PSS auth."""

    def __init__(self, key_id: str, private_key_path: str):
        self.key_id = key_id
        self.api_base = config.KALSHI_API_BASE

        with open(private_key_path, "rb") as f:
            key_bytes = f.read()
        self.private_key = self._load_private_key(key_bytes)
        log.info("KalshiClient initialised (key_id=%s)", key_id)

    # ── private-key loading (PEM / DER / base64) ──────────────────────

    @staticmethod
    def _load_private_key(key_bytes):
        key_text = key_bytes.decode("utf-8", errors="ignore").strip().replace("\\n", "\n")

        if "BEGIN" in key_text:
            return serialization.load_pem_private_key(key_text.encode("utf-8"), password=None)

        b64_body = "".join(key_text.split())
        der_bytes = base64.b64decode(b64_body)

        # Try direct DER first (works for PKCS8/PKCS1 DER material).
        try:
            return serialization.load_der_private_key(der_bytes, password=None)
        except ValueError:
            pass

        # Fallback to wrapped PEM variants.
        for begin_tag, end_tag in [
            ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"),
            ("-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"),
        ]:
            pem = f"{begin_tag}\n" + "\n".join(textwrap.wrap(b64_body, 64)) + f"\n{end_tag}\n"
            try:
                return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
            except ValueError:
                continue

        raise ValueError("Unable to parse Kalshi private key. Provide a valid PEM or DER/base64 RSA private key.")

    # ── auth headers ──────────────────────────────────────────────────

    def _signed_headers(self, method: str, endpoint: str) -> dict:
        ts = str(int(time.time() * 1000))
        path = urlsplit(self.api_base).path + endpoint.split("?")[0]
        msg = f"{ts}{method.upper()}{path}"
        sig = self.private_key.sign(
            msg.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }

    # ── HTTP verbs ────────────────────────────────────────────────────

    def get(self, endpoint: str, params: dict = None) -> dict:
        url = self.api_base + endpoint
        headers = self._signed_headers("GET", endpoint)
        log.debug("GET %s params=%s", url, params)
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint: str, data: dict) -> dict:
        url = self.api_base + endpoint
        headers = self._signed_headers("POST", endpoint)
        log.debug("POST %s data=%s", url, data)
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()

    # ── domain helpers ────────────────────────────────────────────────

    def get_markets(self, event_ticker: str) -> list[dict]:
        data = self.get("/markets", params={"event_ticker": event_ticker, "limit": 200})
        return data["markets"]

    def get_market(self, ticker: str) -> dict:
        data = self.get(f"/markets/{ticker}")
        return data["market"]

    def post_market_order(self, ticker: str, side: str, count: int) -> dict:
        body = {
            "action": "buy",
            "ticker": ticker,
            "side": side,
            "count": count,
            "type": "market",
            "client_order_id": uuid.uuid4().hex,
        }
        return self.post("/portfolio/orders", body)

    def get_balance(self) -> dict:
        return self.get("/portfolio/balance")

    def get_positions(self) -> dict:
        return self.get("/portfolio/positions")
