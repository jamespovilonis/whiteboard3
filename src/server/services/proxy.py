"""HTTP proxy helpers for forwarding OCR/model requests."""

from __future__ import annotations

from typing import Mapping
import urllib.error
import urllib.request


HOP_BY_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding"}


class UpstreamNotConfigured(RuntimeError):
    """Raised when OCR proxying is requested without an upstream API URL."""


class UpstreamResponse:
    def __init__(self, status: int, headers: Mapping[str, str], body: bytes):
        self.status = int(status)
        self.headers = headers
        self.body = body


def proxy_target_url(upstream_api_url: str, path: str) -> str:
    return f"{str(upstream_api_url or '').rstrip('/')}{path}"


def proxy_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def proxy_request(
    *,
    upstream_api_url: str,
    path: str,
    method: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> UpstreamResponse:
    upstream = str(upstream_api_url or "").rstrip("/")
    if not upstream:
        raise UpstreamNotConfigured(
            "No upstream API configured. Start with --upstream-api-url to proxy CoMER/DBNet endpoints."
        )

    request = urllib.request.Request(
        proxy_target_url(upstream, path),
        data=body if body else None,
        headers=proxy_request_headers(headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return UpstreamResponse(response.status, response.headers, response.read())
    except urllib.error.HTTPError as exc:
        return UpstreamResponse(exc.code, exc.headers, exc.read())
