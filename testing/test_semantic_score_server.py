#!/usr/bin/env python3
"""Tests for the local recognition gateway server."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from semantic_score_server import (
    SemanticScoringTimeout,
    SemanticScoreHandler,
    proxy_request_headers,
    proxy_target_url,
    score_semantic_payload_with_timeout,
)


def slow_semantic_payload(_payload):
    time.sleep(2)
    return {"candidateScores": []}


class GatewayPureTests(unittest.TestCase):
    def test_proxy_target_url_preserves_path_and_query(self):
        self.assertEqual(
            proxy_target_url("http://127.0.0.1:8000/", "/recognize?model=comer"),
            "http://127.0.0.1:8000/recognize?model=comer",
        )

    def test_proxy_request_headers_removes_hop_by_hop_headers(self):
        headers = proxy_request_headers({
            "Host": "localhost",
            "Content-Type": "image/png",
            "Content-Length": "10",
            "Connection": "keep-alive",
            "Accept": "application/json",
        })
        self.assertEqual(headers, {
            "Content-Type": "image/png",
            "Accept": "application/json",
        })

    def test_semantic_scoring_timeout_stops_slow_worker(self):
        started = time.monotonic()
        with self.assertRaises(SemanticScoringTimeout):
            score_semantic_payload_with_timeout({}, timeout_seconds=0.05, scorer=slow_semantic_payload)
        self.assertLess(time.monotonic() - started, 1.0)


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send_json({
            "path": self.path,
            "method": "GET",
            "status": "ok",
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        self._send_json({
            "path": self.path,
            "method": "POST",
            "body": body,
            "contentType": self.headers.get("Content-Type"),
            "candidates": [{"latex": "x = 4", "score": -0.1}],
            "top": {"latex": "x = 4", "score": -0.1, "confidence": 0.9},
        })

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class GatewayServerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.upstream = start_server(FakeUpstreamHandler)
            self.gateway = start_server(
                SemanticScoreHandler,
                upstream_api_url=server_url(self.upstream),
                upstream_timeout=5.0,
            )
        except PermissionError as exc:
            if getattr(exc, "errno", None) == 1:
                raise unittest.SkipTest("sandbox does not permit binding localhost sockets") from exc
            raise

    def tearDown(self):
        if hasattr(self, "gateway"):
            stop_server(self.gateway)
        if hasattr(self, "upstream"):
            stop_server(self.upstream)

    def test_gateway_health_reports_semantic_and_upstream(self):
        payload = get_json(f"{server_url(self.gateway)}/gateway/health")
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["semantic"])
        self.assertEqual(payload["upstreamApiUrl"], server_url(self.upstream))

    def test_scores_latex_candidates_locally(self):
        request = urllib.request.Request(
            f"{server_url(self.gateway)}/score-latex-candidates",
            data=json.dumps({
                "problemLatex": r"2 x + 3 = 11",
                "candidateGroups": [{
                    "candidateId": "line-a",
                    "candidates": [{"latex": r"2 x = 8", "score": -1.0}],
                }],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = read_json(request)
        self.assertEqual(payload["candidateScores"][0]["candidateId"], "line-a")
        self.assertEqual(payload["candidateScores"][0]["bestLatex"], r"2 x = 8")

    def test_semantic_timeout_returns_gateway_error(self):
        stop_server(self.gateway)
        self.gateway = start_server(
            SemanticScoreHandler,
            upstream_api_url=server_url(self.upstream),
            upstream_timeout=5.0,
            semantic_timeout=0.05,
            semantic_scorer=slow_semantic_payload,
        )
        request = urllib.request.Request(
            f"{server_url(self.gateway)}/score-latex-candidates",
            data=json.dumps({"candidateGroups": []}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            read_json(request)
        self.assertEqual(context.exception.code, 504)
        payload = json.loads(context.exception.read().decode("utf-8"))
        self.assertIn("exceeded", payload["detail"])

    def test_proxies_recognition_requests_to_upstream(self):
        request = urllib.request.Request(
            f"{server_url(self.gateway)}/recognize?model=comer&timeout_seconds=1",
            data=b"fake-image-bytes",
            headers={"Content-Type": "image/png"},
            method="POST",
        )
        payload = read_json(request)
        self.assertEqual(payload["path"], "/recognize?model=comer&timeout_seconds=1")
        self.assertEqual(payload["method"], "POST")
        self.assertEqual(payload["body"], "fake-image-bytes")
        self.assertEqual(payload["top"]["latex"], "x = 4")

    def test_proxies_health_requests_to_upstream(self):
        payload = get_json(f"{server_url(self.gateway)}/health?model=comer")
        self.assertEqual(payload["path"], "/health?model=comer")
        self.assertEqual(payload["status"], "ok")


def start_server(handler, **attrs):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    for key, value in attrs.items():
        setattr(server, key, value)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._thread = thread
    return server


def stop_server(server):
    server.shutdown()
    server.server_close()
    server._thread.join(timeout=2)


def server_url(server):
    host, port = server.server_address
    return f"http://{host}:{port}"


def get_json(url):
    return read_json(urllib.request.Request(url, method="GET"))


def read_json(request):
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
