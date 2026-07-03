# Whiteboard Codex Notes

- When running the daily VLM audit overview, use Codex model `gpt-5.5` with `xhigh` reasoning effort, corresponding to the user-facing "extra high" setting.
- Use `ops/issue-ledger/issues.json` as the canonical bug handoff between audit, diagnostic, implementation, verification, and weekly-status work. For P0/P1 bug work, start from an issue ID, add or confirm a replay fixture before broad source edits, record verification commands with `scripts/issue_ledger.py`, and do not mark issues closed until replay or comparable live-audit evidence supports the status.
