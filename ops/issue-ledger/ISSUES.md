# Whiteboard 3 Issue Ledger

Generated from `issues.json` at `2026-07-04T13:14:38Z`.

## Actionable Issues

No actionable issues.

## Recently Verified

### WB3-0023 - Audit overlay boxes are reported as visual marks

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `audit-process`
- Key: `audit-overlay-visual-mark-false-positive`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Monitor live audit artifacts for diagnostic overlay false positives before reopening.
- Evidence:
  - `audit_20260703T125227496485Z_5e0ddd9d04`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T125227496485Z_5e0ddd9d04/comparison.json)
  - `audit_20260703T130912395585Z_bbbd388b6e`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T130912395585Z_bbbd388b6e/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_audit_service`

### WB3-0005 - Inverse-trig manifest parse failure turns partial work into incorrect

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `inverse-trig-manifest-parse-failure-misgrades-partial-work`
- Last seen: `2026-07-02T14:28:58.288847Z`
- Next action: Monitor live inverse-trig audits for comparable recurrence.
- Evidence:
  - `audit_20260702T142451952142Z_3f3e9f3660`: Inverse-trig manifest parse failure turns partial work into incorrect ()
  - `audit_20260702T142858288579Z_d3a66ff397`: Inverse-trig manifest parse failure turns partial work into incorrect ()
- Fixtures:
  - `testing/fixtures/real_handwriting/inverse-trig-partial-expression.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0006 - Two-fraction subtraction row is split into numerator/denominator fragments

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `fraction-subtraction-row-split-into-numerator-denominator-fragments`
- Last seen: `2026-07-02T14:30:19.930534Z`
- Next action: Monitor live compact-fraction subtraction audits for comparable recurrence.
- Evidence:
  - `audit_20260702T143019928891Z_14e2f278b4`: Two-fraction subtraction row is split into numerator/denominator fragments ()
- Fixtures:
  - `testing/fixtures/real_handwriting/two-fraction-subtraction-shared-baseline.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0007 - Intermediate algebra row dropped between recognized lines

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `intermediate-algebra-row-dropped-between-recognized-lines`
- Last seen: `2026-07-02T19:05:33.049158Z`
- Next action: Monitor live algebra-row audits for comparable recurrence.
- Evidence:
  - `audit_20260702T190533049012Z_1dc4b772e2`: Intermediate algebra row dropped between recognized lines ()
- Fixtures:
  - `testing/fixtures/real_handwriting/intermediate-algebra-row-retained.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0012 - Fractional log bases lose their subscript braces and become multiplied arguments

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `log-fraction-base-subscript-lost`
- Last seen: `2026-07-02T21:54:12.999529Z`
- Next action: Monitor live fractional-log-base audits for comparable recurrence.
- Evidence:
  - `audit_20260702T215209128186Z_2c6498c760`: Fractional log bases lose their subscript braces and become multiplied arguments ()
- Fixtures:
  - `testing/fixtures/real_handwriting/fractional-log-base-solve-context.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0024 - Tutor feedback leaks prompt text or gives literal fixes

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `tutor-feedback`
- Key: `tutor-feedback-prompt-leak-and-literal-feedback`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Monitor live feedback artifacts for prompt leakage or non-actionable literal feedback recurrence.
- Evidence:
  - `audit_20260703T213955419107Z_1abc4a8529`: Fast pipeline and VLM audit agreed on the checked signals.; fast=incorrect vlm=incorrect; types=visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T213955419107Z_1abc4a8529/comparison.json)
  - `audit_20260703T174233152665Z_de476ff2a5`: VLM JSON must include latexLines as a list; fast=incorrect vlm=None; types=vlm_schema_error (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174233152665Z_de476ff2a5/comparison.json)
  - `audit_20260703T174901022130Z_232e0fa7c8`: timed out; fast=incorrect vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174901022130Z_232e0fa7c8/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_feedback_service`

### WB3-0025 - Realtime segmentation status flips during grading

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition-ux`
- Key: `realtime-segmentation-status-flips`
- Last seen: `2026-07-04T12:13:55Z`
- Next action: Monitor live recognition telemetry for status flips after correct submissions.
- Evidence:
  - `audit_20260703T144210547539Z_fe7e799b0c`: timed out; fast=correct vlm=None; types=vlm_unavailable (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T144210547539Z_fe7e799b0c/comparison.json)
- Verification:
  - `passed` `node --test --test-name-pattern "realtime|segmentation|submitted|status" testing/test_recognition_pipeline.mjs`

### WB3-0021 - VLM audit timeouts and circuit-open skips

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `audit-infra`
- Key: `vlm-audit-timeouts-and-circuit-open`
- Last seen: `2026-07-04T12:13:30Z`
- Next action: Monitor comparable live audit evidence for timeout/circuit-open recurrence before reopening.
- Evidence:
  - `audit_20260703T223159069590Z_8543e95e26`: timed out; fast=correct vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T223159069590Z_8543e95e26/comparison.json)
  - `audit_20260703T144456547842Z_a8394ed135`: VLM audit circuit open after repeated timeouts; retry in 300s; types=vlm_unavailable (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T144456547842Z_a8394ed135/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_audit_service`

### WB3-0022 - VLM audit schema errors return empty JSON

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `audit-infra`
- Key: `vlm-audit-schema-empty-json`
- Last seen: `2026-07-04T12:13:30Z`
- Next action: Monitor live audits for malformed or empty VLM JSON recurrence.
- Evidence:
  - `audit_20260703T125032301196Z_334c75e45e`: VLM JSON must include latexLines as a list; fast=correct vlm=None; types=vlm_schema_error (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T125032301196Z_334c75e45e/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_audit_service`

### WB3-0004 - Detached/circled annotation is selected and graded as a final answer line

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `detached-operation-annotations-selected-as-answer-lines`
- Last seen: `2026-07-04T12:12:59Z`
- Next action: Monitor daily VLM audits for recurrence of unlabeled circle/box annotations being graded as final answers.
- Evidence:
  - `audit_20260702T133425158949Z_a1e84136a1`: Detached/circled annotation is selected and graded as a final answer line ()
  - `audit_20260703T141214622367Z_00171da20d`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch, equation_side_operation_annotation_mismatch; fast=correct vlm=incomplete; types=equation_side_operation_annotation_mismatch,line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T141214622367Z_00171da20d/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/detached-circled-zero-annotation.json`
- Verification:
  - `passed` `node --test --test-name-pattern "segmentation infers visual-only annotations|isolated circled annotation|real standalone zero" testing/test_recognition_pipeline.mjs`

## Workflow

Use `python3 scripts/issue_ledger.py --help` to create, update, transition, and render ledger issues. Do not hand-edit this file; run `python3 scripts/issue_ledger.py render` after changing issue state.