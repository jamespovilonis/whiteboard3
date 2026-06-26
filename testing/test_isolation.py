#!/usr/bin/env python3
"""Guardrails for keeping fixture generation isolated from app/model code."""

from __future__ import annotations

from pathlib import Path
import unittest


TESTING_DIR = Path(__file__).resolve().parent
ISOLATED_SOURCES = [
    TESTING_DIR / "synthetic_handwriting.py",
    TESTING_DIR / "fixture_catalog.py",
    TESTING_DIR / "render_math_fixture.py",
]
FORBIDDEN_TEXT = [
    "CanvasSegmentation",
    "DBNet",
    "CoMER",
    "whiteboard_2",
    "from server",
    "import server",
    "from src",
    "import src",
]


class IsolationTests(unittest.TestCase):
    def test_fixture_sources_do_not_import_app_or_model_code(self):
        for path in ISOLATED_SOURCES:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                for forbidden in FORBIDDEN_TEXT:
                    self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
