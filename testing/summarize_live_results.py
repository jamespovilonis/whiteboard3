#!/usr/bin/env python3
"""Summarize live DBNet/CoMER recognition matrix result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from .fixture_catalog import RESULTS_DIR
except ImportError:
    from fixture_catalog import RESULTS_DIR


LIVE_RESULTS = RESULTS_DIR / "live_recognition"


def load_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append({
            "path": str(path),
            "status": payload.get("status"),
            "totals": payload.get("totals") or {},
        })
    return summaries


def aggregate_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "fixtures",
        "segmentationExact",
        "segmentationFailures",
        "pipelineSelectionExact",
        "pipelineSelectionFailures",
        "ocrLines",
        "ocrTopStrictMatches",
        "ocrTopLooseMatches",
        "ocrSemanticStrictMatches",
        "ocrSemanticLooseMatches",
        "ocrAcceptedMatches",
        "ocrMisses",
    ]
    totals = {key: 0 for key in numeric_keys}
    for summary in summaries:
        for key in numeric_keys:
            totals[key] += int((summary.get("totals") or {}).get(key) or 0)
    return {
        "summaryFiles": len(summaries),
        "totals": totals,
        "files": list(summaries),
    }


def default_summary_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*summary.json"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Summary JSON files. Defaults to testing/results/live_recognition/*summary.json.",
    )
    parser.add_argument("--results-dir", type=Path, default=LIVE_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.paths or default_summary_paths(args.results_dir)
    summary = aggregate_summaries(load_summaries(paths))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
