# Issue Ledger

This directory is the canonical bug handoff between audit automations, diagnostic runs, implementation threads, verification threads, and weekly status summaries.

## Files

- `issues.json`: source of truth for issue state.
- `events.jsonl`: append-only history written by `scripts/issue_ledger.py`.
- `ISSUES.md`: generated human dashboard. Do not hand-edit it; run `python3 scripts/issue_ledger.py render`.

## Status Lifecycle

```text
new -> triaged -> fixture_needed -> ready_for_fix -> fixing -> fixed_unverified -> verified_by_replay -> verified_by_live_audit -> closed
```

Other terminal or special statuses:

```text
reopened, deferred, duplicate, wont_fix
```

Rules:

- Do not mark an issue closed just because code changed.
- Move to `fixed_unverified` after implementation if deterministic replay evidence is missing.
- Move to `verified_by_replay` only after replay tests cover the issue.
- Move to `verified_by_live_audit` only after comparable live-audit evidence appears.
- For P0/P1 bug work, start from an issue ID, add or confirm a replay fixture before broad source edits, record verification commands with `scripts/issue_ledger.py`, and keep the issue open until replay or comparable live-audit evidence supports closure.
- When running the daily VLM audit overview, use Codex model `gpt-5.5` with `xhigh` reasoning effort, corresponding to the user-facing "extra high" setting.

## `issues.json` Shape

```json
{
  "schema_version": 1,
  "next_id": 2,
  "issues": [
    {
      "id": "WB3-0001",
      "key": "stable-recurring-bug-key",
      "title": "Human-readable issue title",
      "severity": "P1",
      "status": "fixture_needed",
      "category": "recognition",
      "area": ["src/recognition/studentWritingPipeline.js"],
      "first_seen": "2026-07-05T12:00:00Z",
      "last_seen": "2026-07-05T12:00:00Z",
      "evidence": [
        {
          "audit_id": "audit_...",
          "path": "/absolute/path/to/comparison.json",
          "summary": "Short evidence summary",
          "discrepancy_types": ["line_latex_mismatch"]
        }
      ],
      "acceptance_criteria": [
        "Replay fixture reproduces the original failure.",
        "Targeted fix passes the replay and no broad segmentation regression."
      ],
      "fixtures": [
        {
          "path": "testing/fixtures/real_handwriting/example.json",
          "added_at": "2026-07-05T12:10:00Z",
          "note": "Replay fixture for the audit discrepancy."
        }
      ],
      "suspected_files": ["src/recognition/lineSegmentation.js"],
      "fix_commits": [],
      "verification": {
        "replay": "missing",
        "live_audit": "missing",
        "commands": [
          {
            "command": "node --test testing/test_recognition_pipeline.mjs",
            "result": "passed",
            "timestamp": "2026-07-05T12:20:00Z",
            "note": "Targeted replay passed."
          }
        ]
      },
      "next_action": "Add a replay fixture before implementing.",
      "notes": [
        {
          "timestamp": "2026-07-05T12:05:00Z",
          "text": "Triage note."
        }
      ]
    }
  ]
}
```

The script may add diagnostic fields such as `diagnostic_status`, `diagnostic_occurrence_count`, `affected_problem_ids`, `observed_statuses`, or `updated_at` when importing automated diagnostic state.

## Typical Commands

Create or update an issue from audit evidence:

```bash
python3 scripts/issue_ledger.py upsert \
  --key problem-input-fraction-repair-overstrips \
  --title "Problem-input fraction repair strips equation tails" \
  --severity P0 \
  --category recognition \
  --from-audit /path/to/comparison.json
```

Move an issue through the lifecycle:

```bash
python3 scripts/issue_ledger.py transition WB3-0001 fixture_needed --note "Needs replay fixture before implementation"
python3 scripts/issue_ledger.py transition WB3-0001 fixing --note "Starting targeted implementation"
python3 scripts/issue_ledger.py transition WB3-0001 fixed_unverified --note "Code changed; waiting on replay evidence"
```

Attach replay fixtures and verification:

```bash
python3 scripts/issue_ledger.py add-fixture WB3-0001 testing/fixtures/real_handwriting/example.json
python3 scripts/issue_ledger.py add-verification WB3-0001 "node --test testing/test_recognition_pipeline.mjs" --kind replay --result passed
python3 scripts/issue_ledger.py add-verification WB3-0001 "python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8010" --kind live_audit --result passed
```

Import diagnostic state, render the dashboard, and choose next work:

```bash
python3 scripts/issue_ledger.py import-diagnostic-state /Users/jpovj/Documents/dev/log_whiteboard_3/app_discrepancy_diagnostics/state.json
python3 scripts/issue_ledger.py render
python3 scripts/issue_ledger.py next --severity P0
```
