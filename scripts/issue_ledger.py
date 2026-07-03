#!/usr/bin/env python3
"""Maintain the repo-native issue ledger used by Codex automations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_DIR = REPO_ROOT / "ops" / "issue-ledger"
ISSUES_FILE = "issues.json"
EVENTS_FILE = "events.jsonl"
DASHBOARD_FILE = "ISSUES.md"

OPEN_STATUSES = {
    "new",
    "triaged",
    "fixture_needed",
    "ready_for_fix",
    "fixing",
    "fixed_unverified",
    "verified_by_replay",
    "verified_by_live_audit",
    "reopened",
}
TERMINAL_STATUSES = {"closed", "deferred", "duplicate", "wont_fix"}
ALL_STATUSES = OPEN_STATUSES | TERMINAL_STATUSES
VERIFIED_STATUSES = {"verified_by_replay", "verified_by_live_audit"}
SEVERITIES = ["P0", "P1", "P2", "P3"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


@dataclass
class Ledger:
    root: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "Ledger":
        root.mkdir(parents=True, exist_ok=True)
        data = read_json(root / ISSUES_FILE, {"schema_version": 1, "next_id": 1, "issues": []})
        data.setdefault("schema_version", 1)
        data.setdefault("next_id", 1)
        data.setdefault("issues", [])
        return cls(root=root, data=data)

    def save(self) -> None:
        write_json(self.root / ISSUES_FILE, self.data)

    def append_event(self, event_type: str, issue_id: str | None, summary: str, **extra: Any) -> None:
        payload = {
            "timestamp": now_iso(),
            "type": event_type,
            "issue_id": issue_id,
            "summary": summary,
        }
        payload.update({key: value for key, value in extra.items() if value is not None})
        with (self.root / EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def new_id(self) -> str:
        next_id = int(self.data.get("next_id", 1))
        self.data["next_id"] = next_id + 1
        return f"WB3-{next_id:04d}"

    def issue_by_id(self, issue_id: str) -> dict[str, Any] | None:
        issue_id = issue_id.upper()
        for issue in self.data["issues"]:
            if issue.get("id", "").upper() == issue_id:
                return issue
        return None

    def issue_by_key(self, key: str) -> dict[str, Any] | None:
        for issue in self.data["issues"]:
            if issue.get("key") == key:
                return issue
        return None

    def require_issue(self, issue_id: str) -> dict[str, Any]:
        issue = self.issue_by_id(issue_id)
        if issue is None:
            raise SystemExit(f"Unknown issue id: {issue_id}")
        return issue


def load_audit_evidence(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = read_json(path, {})
    audit_id = next((part for part in path.parts if part.startswith("audit_")), path.parent.name)
    discrepancy_types = []
    for item in comparison.get("discrepancies", []) or []:
        if item.get("type"):
            discrepancy_types.append(item["type"])
    for item in comparison.get("observation_only_discrepancies", []) or []:
        if item.get("type"):
            discrepancy_types.append(item["type"])
    discrepancy_types = sorted(set(discrepancy_types))
    summary_parts = []
    if comparison.get("description"):
        summary_parts.append(comparison["description"])
    if comparison.get("fastProblemStatus") or comparison.get("vlmProblemStatus"):
        summary_parts.append(
            f"fast={comparison.get('fastProblemStatus')} vlm={comparison.get('vlmProblemStatus')}"
        )
    if discrepancy_types:
        summary_parts.append("types=" + ",".join(discrepancy_types))
    evidence = {
        "audit_id": audit_id,
        "path": str(path.resolve()),
        "summary": "; ".join(summary_parts) or "Audit comparison evidence",
        "discrepancy_types": discrepancy_types,
    }
    return comparison, evidence


def default_issue(args: argparse.Namespace, ledger: Ledger, key: str, evidence: dict[str, Any] | None) -> dict[str, Any]:
    created = now_iso()
    title = args.title
    if not title and evidence:
        title = f"Audit discrepancy: {', '.join(evidence.get('discrepancy_types') or ['needs triage'])}"
    if not title:
        title = key.replace("-", " ").title()
    return {
        "id": ledger.new_id(),
        "key": key,
        "title": title,
        "severity": args.severity,
        "status": args.status or "new",
        "category": args.category,
        "area": args.area or [],
        "first_seen": created,
        "last_seen": created,
        "evidence": [],
        "acceptance_criteria": args.acceptance_criteria or [],
        "fixtures": [],
        "suspected_files": args.suspected_file or [],
        "fix_commits": [],
        "verification": {"replay": "missing", "live_audit": "missing", "commands": []},
        "next_action": args.next_action or "Triage issue and define acceptance criteria.",
        "notes": [],
    }


def add_unique_dict(items: list[dict[str, Any]], new_item: dict[str, Any], identity_key: str) -> bool:
    identity = new_item.get(identity_key)
    for item in items:
        if item.get(identity_key) == identity:
            item.update({key: value for key, value in new_item.items() if value not in (None, [], "")})
            return False
    items.append(new_item)
    return True


def cmd_upsert(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    comparison = None
    evidence = None
    if args.from_audit:
        comparison, evidence = load_audit_evidence(args.from_audit)
    key = args.key or (args.issue.lower() if args.issue else None)
    if not key:
        if evidence and evidence.get("discrepancy_types"):
            key = slugify(" ".join(evidence["discrepancy_types"]))
        else:
            key = slugify(args.title or "untriaged-audit-discrepancy")

    issue = ledger.issue_by_id(args.issue) if args.issue else ledger.issue_by_key(key)
    created = False
    if issue is None:
        issue = default_issue(args, ledger, key, evidence)
        ledger.data["issues"].append(issue)
        created = True
    else:
        if args.title:
            issue["title"] = args.title
        if args.severity:
            issue["severity"] = args.severity
        if args.status:
            issue["status"] = args.status
        if args.category:
            issue["category"] = args.category
        if args.next_action:
            issue["next_action"] = args.next_action
        for area in args.area or []:
            issue.setdefault("area", [])
            if area not in issue["area"]:
                issue["area"].append(area)
        for file_path in args.suspected_file or []:
            issue.setdefault("suspected_files", [])
            if file_path not in issue["suspected_files"]:
                issue["suspected_files"].append(file_path)
        for criterion in args.acceptance_criteria or []:
            issue.setdefault("acceptance_criteria", [])
            if criterion not in issue["acceptance_criteria"]:
                issue["acceptance_criteria"].append(criterion)

    if evidence:
        added = add_unique_dict(issue.setdefault("evidence", []), evidence, "audit_id")
        issue["last_seen"] = now_iso()
        if comparison and comparison.get("status"):
            issue.setdefault("observed_statuses", [])
            if comparison["status"] not in issue["observed_statuses"]:
                issue["observed_statuses"].append(comparison["status"])
    else:
        added = False

    ledger.save()
    event_type = "issue_created" if created else "issue_updated"
    summary = f"{'Created' if created else 'Updated'} {issue['id']}: {issue['title']}"
    ledger.append_event(event_type, issue["id"], summary, evidence_added=added)
    render_dashboard(ledger)
    print(summary)


def cmd_transition(args: argparse.Namespace) -> None:
    if args.status not in ALL_STATUSES:
        raise SystemExit(f"Invalid status {args.status}. Valid statuses: {', '.join(sorted(ALL_STATUSES))}")
    ledger = Ledger.load(args.ledger_dir)
    issue = ledger.require_issue(args.issue)
    old_status = issue.get("status")
    issue["status"] = args.status
    issue["updated_at"] = now_iso()
    if args.next_action is not None:
        issue["next_action"] = args.next_action
    if args.note:
        issue.setdefault("notes", []).append({"timestamp": now_iso(), "text": args.note})
    ledger.save()
    ledger.append_event("status_changed", issue["id"], f"{issue['id']} {old_status} -> {args.status}", note=args.note)
    render_dashboard(ledger)
    print(f"{issue['id']} {old_status} -> {args.status}")


def cmd_add_fixture(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    issue = ledger.require_issue(args.issue)
    fixture = {"path": args.path, "added_at": now_iso()}
    if args.note:
        fixture["note"] = args.note
    add_unique_dict(issue.setdefault("fixtures", []), fixture, "path")
    if issue.get("status") in {"new", "triaged", "fixture_needed"}:
        issue["status"] = "ready_for_fix"
    ledger.save()
    ledger.append_event("fixture_added", issue["id"], f"Added fixture to {issue['id']}: {args.path}", note=args.note)
    render_dashboard(ledger)
    print(f"Added fixture to {issue['id']}: {args.path}")


def cmd_add_verification(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    issue = ledger.require_issue(args.issue)
    verification = issue.setdefault("verification", {"replay": "missing", "live_audit": "missing", "commands": []})
    command = {"command": args.command, "result": args.result, "timestamp": now_iso()}
    if args.note:
        command["note"] = args.note
    verification.setdefault("commands", []).append(command)
    if args.kind == "replay" and args.result == "passed":
        verification["replay"] = "passed"
        if issue.get("status") in {"fixture_needed", "ready_for_fix", "fixing", "fixed_unverified"}:
            issue["status"] = "verified_by_replay"
    if args.kind == "live_audit" and args.result == "passed":
        verification["live_audit"] = "passed"
        if issue.get("status") in {"verified_by_replay", "fixed_unverified"}:
            issue["status"] = "verified_by_live_audit"
    ledger.save()
    ledger.append_event(
        "verification_added",
        issue["id"],
        f"Added {args.kind} verification to {issue['id']}: {args.result}",
        command=args.command,
        note=args.note,
    )
    render_dashboard(ledger)
    print(f"Added verification to {issue['id']}: {args.result}")


def cmd_next(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    issues = ledger.data.get("issues", [])
    if args.severity:
        issues = [issue for issue in issues if issue.get("severity") == args.severity]
    if args.status:
        issues = [issue for issue in issues if issue.get("status") == args.status]
    else:
        issues = [
            issue
            for issue in issues
            if issue.get("status") not in TERMINAL_STATUSES and issue.get("status") not in VERIFIED_STATUSES
        ]
    severity_rank = {severity: index for index, severity in enumerate(SEVERITIES)}
    status_rank = {
        "reopened": 0,
        "new": 1,
        "triaged": 2,
        "fixture_needed": 3,
        "ready_for_fix": 4,
        "fixing": 5,
        "fixed_unverified": 6,
        "verified_by_replay": 7,
        "verified_by_live_audit": 8,
    }
    issues.sort(key=lambda issue: (severity_rank.get(issue.get("severity"), 99), status_rank.get(issue.get("status"), 99), issue.get("id", "")))
    for issue in issues[: args.limit]:
        print(f"{issue['id']} [{issue.get('severity')}] {issue.get('status')}: {issue.get('title')}")
        print(f"  next: {issue.get('next_action', '')}")


def severity_from_diagnostic(issue: dict[str, Any]) -> str:
    status = " ".join(str(issue.get(key, "")) for key in ("status", "last_trend", "trend", "title")).lower()
    if "p0" in status or "regression" in status or "worsening" in status:
        return "P0"
    if "persistent" in status or "active_new" in status:
        return "P1"
    return "P2"


def status_from_diagnostic(issue: dict[str, Any]) -> str:
    raw = str(issue.get("status", "")).lower()
    trend = " ".join(str(issue.get(key, "")) for key in ("last_trend", "trend")).lower()
    combined = f"{raw} {trend}"
    if "verified_by_live" in combined or "live_audit" in combined:
        return "verified_by_live_audit"
    if "verified_by_replay" in combined or "replay_verification" in combined:
        return "verified_by_replay"
    if "deferred" in combined:
        return "deferred"
    if "recurred_after" in combined or "reopened" in combined:
        return "reopened"
    if "resolved" in combined:
        return "verified_by_live_audit"
    if "implemented" in combined or "awaiting_comparable_post_fix" in combined:
        return "fixed_unverified"
    if "new_regression" in combined or "worsening" in combined:
        return "fixture_needed"
    if "new" in combined:
        return "triaged"
    return "fixture_needed"


def evidence_from_diagnostic(issue: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for audit_id in issue.get("representative_audit_ids", []) or []:
        evidence.append(
            {
                "audit_id": audit_id,
                "path": "",
                "summary": issue.get("title") or issue.get("issue_key") or "Diagnostic representative audit",
                "source": "app_discrepancy_diagnostic",
            }
        )
    return evidence


def cmd_import_diagnostic_state(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    state = read_json(args.path, {})
    imported = 0
    updated = 0
    for diagnostic_issue in state.get("ongoing_issues", []) or []:
        key = diagnostic_issue.get("issue_key") or diagnostic_issue.get("key")
        if not key:
            continue
        issue = ledger.issue_by_key(key)
        created = False
        if issue is None:
            issue = {
                "id": ledger.new_id(),
                "key": key,
                "title": diagnostic_issue.get("title") or key.replace("-", " ").title(),
                "severity": args.severity or severity_from_diagnostic(diagnostic_issue),
                "status": status_from_diagnostic(diagnostic_issue),
                "category": args.category,
                "area": [diagnostic_issue.get("likely_app_area")] if diagnostic_issue.get("likely_app_area") else [],
                "first_seen": diagnostic_issue.get("first_seen") or now_iso(),
                "last_seen": diagnostic_issue.get("last_seen") or now_iso(),
                "evidence": [],
                "acceptance_criteria": [],
                "fixtures": [],
                "suspected_files": [],
                "fix_commits": [],
                "verification": {"replay": "missing", "live_audit": "missing", "commands": []},
                "next_action": diagnostic_issue.get("recommended_fix") or "Triage issue and define next action.",
                "notes": [],
            }
            ledger.data["issues"].append(issue)
            created = True
            imported += 1
        else:
            updated += 1
            issue["title"] = diagnostic_issue.get("title") or issue.get("title")
            issue["severity"] = args.severity or issue.get("severity") or severity_from_diagnostic(diagnostic_issue)
            if issue.get("status") not in {"closed", "verified_by_live_audit"}:
                issue["status"] = status_from_diagnostic(diagnostic_issue)
            issue["last_seen"] = diagnostic_issue.get("last_seen") or issue.get("last_seen")
            issue["next_action"] = diagnostic_issue.get("recommended_fix") or issue.get("next_action")
            if diagnostic_issue.get("likely_app_area"):
                issue.setdefault("area", [])
                if diagnostic_issue["likely_app_area"] not in issue["area"]:
                    issue["area"].append(diagnostic_issue["likely_app_area"])
        issue["diagnostic_status"] = diagnostic_issue.get("status")
        issue["diagnostic_occurrence_count"] = diagnostic_issue.get("occurrence_count")
        issue["affected_problem_ids"] = diagnostic_issue.get("affected_problem_ids", [])
        for evidence in evidence_from_diagnostic(diagnostic_issue):
            add_unique_dict(issue.setdefault("evidence", []), evidence, "audit_id")
        ledger.append_event(
            "diagnostic_issue_imported" if created else "diagnostic_issue_updated",
            issue["id"],
            f"{'Imported' if created else 'Updated'} diagnostic issue {key}",
            diagnostic_status=diagnostic_issue.get("status"),
        )
    ledger.save()
    render_dashboard(ledger)
    print(f"Imported {imported}, updated {updated} issues from {args.path}")


def issue_markdown(issue: dict[str, Any]) -> str:
    evidence = issue.get("evidence", [])
    fixtures = issue.get("fixtures", [])
    verifications = issue.get("verification", {}).get("commands", [])
    lines = [
        f"### {issue.get('id')} - {issue.get('title')}",
        "",
        f"- Severity: `{issue.get('severity')}`",
        f"- Status: `{issue.get('status')}`",
        f"- Category: `{issue.get('category')}`",
        f"- Key: `{issue.get('key')}`",
        f"- Last seen: `{issue.get('last_seen', 'unknown')}`",
        f"- Next action: {issue.get('next_action', '')}",
    ]
    if issue.get("acceptance_criteria"):
        lines.append("- Acceptance criteria:")
        for criterion in issue["acceptance_criteria"]:
            lines.append(f"  - {criterion}")
    if evidence:
        lines.append("- Evidence:")
        for item in evidence[-5:]:
            lines.append(f"  - `{item.get('audit_id')}`: {item.get('summary')} ({item.get('path')})")
    if fixtures:
        lines.append("- Fixtures:")
        for fixture in fixtures:
            lines.append(f"  - `{fixture.get('path')}`")
    if verifications:
        lines.append("- Verification:")
        for command in verifications[-5:]:
            lines.append(f"  - `{command.get('result')}` `{command.get('command')}`")
    lines.append("")
    return "\n".join(lines)


def render_dashboard(ledger: Ledger) -> None:
    issues = ledger.data.get("issues", [])
    open_issues = [
        issue
        for issue in issues
        if issue.get("status") not in TERMINAL_STATUSES and issue.get("status") not in VERIFIED_STATUSES
    ]
    verified = [issue for issue in issues if issue.get("status") in VERIFIED_STATUSES | {"closed"}]
    severity_rank = {severity: index for index, severity in enumerate(SEVERITIES)}
    open_issues.sort(key=lambda issue: (severity_rank.get(issue.get("severity"), 99), issue.get("id", "")))
    verified.sort(key=lambda issue: issue.get("updated_at") or issue.get("last_seen") or "", reverse=True)
    lines = [
        "# Whiteboard 3 Issue Ledger",
        "",
        f"Generated from `issues.json` at `{now_iso()}`.",
        "",
        "## Actionable Issues",
        "",
    ]
    if open_issues:
        for issue in open_issues:
            lines.append(issue_markdown(issue))
    else:
        lines.append("No actionable issues.\n")
    lines.extend(["## Recently Verified", ""])
    if verified:
        for issue in verified[:10]:
            lines.append(issue_markdown(issue))
    else:
        lines.append("No verified issues.\n")
    lines.extend(
        [
            "## Workflow",
            "",
            "Use `python3 scripts/issue_ledger.py --help` to create, update, transition, and render ledger issues. Do not hand-edit this file; run `python3 scripts/issue_ledger.py render` after changing issue state.",
        ]
    )
    (ledger.root / DASHBOARD_FILE).write_text("\n".join(lines), encoding="utf-8")


def cmd_render(args: argparse.Namespace) -> None:
    ledger = Ledger.load(args.ledger_dir)
    render_dashboard(ledger)
    print(str((ledger.root / DASHBOARD_FILE).resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upsert = subparsers.add_parser("upsert", help="Create or update an issue")
    upsert.add_argument("--issue", help="Existing issue id to update")
    upsert.add_argument("--key", help="Stable issue key for recurring bug grouping")
    upsert.add_argument("--title")
    upsert.add_argument("--severity", choices=SEVERITIES, default="P1")
    upsert.add_argument("--status", choices=sorted(ALL_STATUSES))
    upsert.add_argument("--category", default="recognition")
    upsert.add_argument("--area", action="append")
    upsert.add_argument("--suspected-file", action="append")
    upsert.add_argument("--acceptance-criteria", action="append")
    upsert.add_argument("--next-action")
    upsert.add_argument("--from-audit", type=Path)
    upsert.set_defaults(func=cmd_upsert)

    transition = subparsers.add_parser("transition", help="Move an issue to a new status")
    transition.add_argument("issue")
    transition.add_argument("status")
    transition.add_argument("--note")
    transition.add_argument("--next-action")
    transition.set_defaults(func=cmd_transition)

    add_fixture = subparsers.add_parser("add-fixture", help="Attach a replay fixture to an issue")
    add_fixture.add_argument("issue")
    add_fixture.add_argument("path")
    add_fixture.add_argument("--note")
    add_fixture.set_defaults(func=cmd_add_fixture)

    add_verification = subparsers.add_parser("add-verification", help="Record a verification command")
    add_verification.add_argument("issue")
    add_verification.add_argument("command")
    add_verification.add_argument("--kind", choices=["replay", "live_audit", "manual"], default="replay")
    add_verification.add_argument("--result", choices=["passed", "failed", "skipped"], default="passed")
    add_verification.add_argument("--note")
    add_verification.set_defaults(func=cmd_add_verification)

    next_cmd = subparsers.add_parser("next", help="List highest-priority open issues")
    next_cmd.add_argument("--severity", choices=SEVERITIES)
    next_cmd.add_argument("--status", choices=sorted(ALL_STATUSES))
    next_cmd.add_argument("--limit", type=int, default=10)
    next_cmd.set_defaults(func=cmd_next)

    import_state = subparsers.add_parser("import-diagnostic-state", help="Import app discrepancy diagnostic state.json")
    import_state.add_argument("path", type=Path)
    import_state.add_argument("--category", default="recognition")
    import_state.add_argument("--severity", choices=SEVERITIES)
    import_state.set_defaults(func=cmd_import_diagnostic_state)

    render = subparsers.add_parser("render", help="Regenerate ISSUES.md")
    render.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
