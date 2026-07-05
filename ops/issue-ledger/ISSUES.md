# Whiteboard 3 Issue Ledger

Generated from `issues.json` at `2026-07-05T12:00:50Z`.

## Actionable Issues

No actionable issues.

## Recently Verified

### WB3-0001 - Candidate/grading path overtrusts ambiguous final-answer OCR

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `candidate-selection-ambiguous-final-answer-overtrust`
- Last seen: `2026-07-05T11:42:45Z`
- Next action: Use the July 4 replay fixture to design a targeted fraction/final-answer selection fix.
- Evidence:
  - `audit_20260702T132349514705Z_0da52d6b90`: Candidate/grading path overtrusts ambiguous final-answer OCR ()
  - `audit_20260702T133425158949Z_a1e84136a1`: Candidate/grading path overtrusts ambiguous final-answer OCR ()
  - `audit_20260704T124709059757Z_800c947a21`: VLM audit raised: problem_status_mismatch, line_latex_mismatch; fast=incomplete vlm=incorrect; types=line_latex_mismatch,problem_status_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-04/audit_20260704T124709059757Z_800c947a21/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/20260704t124709059757z-800c947a21.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures|decimal|VLM audit" testing/test_recognition_pipeline.mjs`
  - `passed` `npm run test:segmentation`
  - `passed` `npm run build`

### WB3-0010 - Valid semantic OCR candidates are present but not promoted to accepted/final grading lines

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `valid-semantic-alternate-not-promoted`
- Last seen: `2026-07-05T11:42:50Z`
- Next action: Watch next live handwriting audit for decimal 9.5/2-style corrections before closing.
- Evidence:
  - `audit_20260702T220020765884Z_71983a33ea`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T125227496485Z_5e0ddd9d04`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T130912395585Z_bbbd388b6e`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T142416290747Z_ff29f8b9d9`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260704T131146086859Z_a3afacc0c0`: VLM audit raised: line_latex_mismatch, vlm_low_confidence; fast=correct vlm=correct; types=line_latex_mismatch,visual_intent_observed,vlm_low_confidence (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-04/audit_20260704T131146086859Z_a3afacc0c0/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/20260704t131146086859z-a3afacc0c0.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures|decimal|VLM audit" testing/test_recognition_pipeline.mjs`
  - `passed` `npm run test:segmentation`
  - `passed` `npm run build`

### WB3-0021 - VLM audit timeouts and circuit-open skips

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `vlm-audit-timeouts-and-circuit-open`
- Last seen: `2026-07-05T11:42:13Z`
- Next action: Watch next daily audit for live circuit-open telemetry before closing.
- Evidence:
  - `audit_20260703T223159069590Z_8543e95e26`: timed out; fast=correct vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T223159069590Z_8543e95e26/comparison.json)
  - `audit_20260703T144456547842Z_a8394ed135`: VLM audit circuit open after repeated timeouts; retry in 300s; types=vlm_unavailable (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T144456547842Z_a8394ed135/comparison.json)
  - `audit_20260704T132042116096Z_47e6686c30`: timed out; fast=incomplete vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-04/audit_20260704T132042116096Z_47e6686c30/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_audit_service`
  - `passed` `python3 -m unittest testing.test_audit_service testing.test_feedback_service testing.test_issue_ledger`
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures|decimal|VLM audit" testing/test_recognition_pipeline.mjs`
  - `passed` `npm run build`

### WB3-0023 - Audit overlay boxes are reported as visual marks

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `audit-process`
- Key: `audit-overlay-visual-mark-false-positive`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Watch next daily audit for absence of overlay visual-mark false positives before closing.
- Evidence:
  - `audit_20260703T125227496485Z_5e0ddd9d04`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T125227496485Z_5e0ddd9d04/comparison.json)
  - `audit_20260703T130912395585Z_bbbd388b6e`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T130912395585Z_bbbd388b6e/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_audit_service`
  - `passed` `python3 -m unittest testing.test_audit_service testing.test_feedback_service testing.test_issue_ledger`
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures|decimal|VLM audit" testing/test_recognition_pipeline.mjs`
  - `passed` `npm run build`

### WB3-0024 - Tutor feedback leaks prompt text or gives literal fixes

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `tutor-feedback`
- Key: `tutor-feedback-prompt-leak-and-literal-feedback`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Watch live feedback artifacts for missing-target and decimal-format behavior before closing.
- Evidence:
  - `audit_20260703T213955419107Z_1abc4a8529`: Fast pipeline and VLM audit agreed on the checked signals.; fast=incorrect vlm=incorrect; types=visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T213955419107Z_1abc4a8529/comparison.json)
  - `audit_20260703T174233152665Z_de476ff2a5`: VLM JSON must include latexLines as a list; fast=incorrect vlm=None; types=vlm_schema_error (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174233152665Z_de476ff2a5/comparison.json)
  - `audit_20260703T174901022130Z_232e0fa7c8`: timed out; fast=incorrect vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174901022130Z_232e0fa7c8/comparison.json)
- Verification:
  - `passed` `python3 -m unittest testing.test_feedback_service`
  - `passed` `python3 -m unittest testing.test_audit_service testing.test_feedback_service testing.test_issue_ledger`
  - `passed` `npm run build`

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

## Workflow

Use `python3 scripts/issue_ledger.py --help` to create, update, transition, and render ledger issues. Do not hand-edit this file; run `python3 scripts/issue_ledger.py render` after changing issue state.