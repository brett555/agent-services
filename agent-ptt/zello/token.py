"""Zello JWT token generation for server-side use.

For development, use the 30-day token from https://developers.zello.com/
For production, generate tokens here using your Issuer ID and RSA private key.
"""
import time

import jwt


def generate_token(issuer: str, private_key_pem: str, ttl_seconds: int = 3600) -> str:
    """Generate a Zello RS256 JWT token."""
    payload = {
        "iss": issuer,
        "exp": int(time.time()) + ttl_seconds,
    }
    return jwt.encode(payload, private_key_pem, algorithm="RS256")
