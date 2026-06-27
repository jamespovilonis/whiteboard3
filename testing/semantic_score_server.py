#!/usr/bin/env python3
"""Tiny local recognition gateway for recognition-pipeline experiments.

This uses only the standard library plus ``latex_semantics`` so it can be run
next to the Vite app without introducing a backend framework. It serves the
semantic endpoint locally and can proxy CoMER/DBNet requests to an existing
model API, giving the browser one ``VITE_OCR_API_URL`` for the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from latex_semantics import score_semantic_payload


class SemanticScoringTimeout(TimeoutError):
    """Raised when semantic scoring exceeds the gateway request budget."""


def proxy_target_url(upstream_api_url, path):
    return f"{str(upstream_api_url or '').rstrip('/')}{path}"


def proxy_request_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }


class SemanticScoreHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/gateway/health":
            self._send_json({
                "status": "ok",
                "semantic": True,
                "upstreamApiUrl": self._upstream_api_url(),
            })
            return
        if self._is_proxy_path():
            self._proxy_request()
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        if self.path != "/score-latex-candidates":
            if self._is_proxy_path():
                self._proxy_request()
                return
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json({"detail": f"Invalid JSON: {exc}"}, status=400)
            return

        started = time.monotonic()
        try:
            result = score_semantic_payload_with_timeout(
                payload,
                timeout_seconds=getattr(self.server, "semantic_timeout", 2.5),
                scorer=getattr(self.server, "semantic_scorer", score_semantic_payload),
            )
            result["elapsedSeconds"] = round(time.monotonic() - started, 3)
        except SemanticScoringTimeout as exc:
            self._send_json({"detail": str(exc)}, status=504)
            return
        except Exception as exc:
            self._send_json({"detail": f"Semantic scoring failed: {exc}"}, status=500)
            return

        self._send_json(result)

    def log_message(self, format, *args):
        return

    def _is_proxy_path(self):
        return (
            self.path.startswith("/recognize") or
            self.path.startswith("/segment-lines") or
            self.path.startswith("/health")
        )

    def _upstream_api_url(self):
        return getattr(self.server, "upstream_api_url", "") or ""

    def _proxy_request(self):
        upstream = self._upstream_api_url().rstrip("/")
        if not upstream:
            self._send_json({
                "detail": "No upstream API configured. Start with --upstream-api-url to proxy CoMER/DBNet endpoints."
            }, status=502)
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else None
        target = proxy_target_url(upstream, self.path)
        headers = proxy_request_headers(self.headers)
        request = urllib.request.Request(
            target,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=getattr(self.server, "upstream_timeout", 25.0)) as response:
                payload = response.read()
                self._send_proxy_response(response.status, response.headers, payload)
        except urllib.error.HTTPError as exc:
            self._send_proxy_response(exc.code, exc.headers, exc.read())
        except Exception as exc:
            self._send_json({"detail": f"Upstream request failed: {exc}"}, status=502)

    def _send_proxy_response(self, status, headers, payload):
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", headers.get("Content-Type", "application/octet-stream"))
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def score_semantic_payload_with_timeout(payload, timeout_seconds=2.5, scorer=score_semantic_payload):
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return scorer(payload)

    if "fork" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context()
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_score_semantic_payload_worker,
        args=(payload, queue, scorer),
        daemon=True,
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        raise SemanticScoringTimeout(f"Semantic scoring exceeded {timeout:.2f}s")

    try:
        status, result = queue.get(timeout=1)
    except queue.Empty as exc:
        raise RuntimeError("semantic scoring worker exited without a result") from exc

    if status == "ok":
        return result
    raise RuntimeError(result)


def _score_semantic_payload_worker(payload, queue, scorer):
    try:
        queue.put(("ok", scorer(payload)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--upstream-api-url",
        default="",
        help="Optional CoMER/DBNet API base URL to proxy /recognize, /segment-lines, and /health.",
    )
    parser.add_argument("--upstream-timeout", type=float, default=25.0)
    parser.add_argument(
        "--semantic-timeout",
        type=float,
        default=2.5,
        help="Maximum seconds to spend scoring one /score-latex-candidates request.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SemanticScoreHandler)
    server.upstream_api_url = args.upstream_api_url
    server.upstream_timeout = args.upstream_timeout
    server.semantic_timeout = args.semantic_timeout
    print(f"Recognition gateway listening on http://{args.host}:{args.port}")
    if args.upstream_api_url:
        print(f"Proxying CoMER/DBNet endpoints to {args.upstream_api_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
