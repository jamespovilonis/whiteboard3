# Issue Ledger

This directory is the canonical bug handoff between audit automations, implementation threads, verification threads, and weekly status summaries.

- `issues.json` is the source of truth.
- `events.jsonl` is append-only history.
- `ISSUES.md` is a generated human dashboard.

Status lifecycle:

```text
new -> triaged -> fixture_needed -> ready_for_fix -> fixing -> fixed_unverified -> verified_by_replay -> verified_by_live_audit -> closed
```

Other statuses:

```text
reopened, deferred, duplicate, wont_fix
```

Rules:

- Do not mark an issue closed just because code changed.
- Move to `fixed_unverified` after implementation if deterministic replay evidence is missing.
- Move to `verified_by_replay` only after replay tests cover the issue.
- Move to `verified_by_live_audit` only after comparable live audit evidence appears.
- Every P0/P1 implementation thread should name the issue ID, acceptance criteria, and verification commands before editing source code.

Typical commands:

```bash
python3 scripts/issue_ledger.py upsert --key problem-input-fraction-repair-overstrips --title "Problem-input fraction repair strips equation tails" --severity P0 --category recognition --from-audit /path/to/comparison.json
python3 scripts/issue_ledger.py transition WB3-0001 fixture_needed --note "Needs replay fixture before implementation"
python3 scripts/issue_ledger.py add-fixture WB3-0001 testing/fixtures/real_handwriting/example.json
python3 scripts/issue_ledger.py add-verification WB3-0001 "node --test testing/test_recognition_pipeline.mjs" --result passed
python3 scripts/issue_ledger.py import-diagnostic-state /Users/jpovj/Documents/dev/log_whiteboard_3/app_discrepancy_diagnostics/state.json
python3 scripts/issue_ledger.py render
python3 scripts/issue_ledger.py next --severity P0
```
