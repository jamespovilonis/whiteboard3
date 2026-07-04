# Whiteboard 3 Issue Ledger

Generated from `issues.json` at `2026-07-04T12:50:44Z`.

## Actionable Issues

### WB3-0021 - VLM audit timeouts and circuit-open skips

- Severity: `P1`
- Status: `triaged`
- Category: `audit-infra`
- Key: `vlm-audit-timeouts-and-circuit-open`
- Last seen: `2026-07-04T12:13:30Z`
- Next action: Add VLM request metadata, queue/inference timing, retry behavior, and circuit-open accounting. Keep timeout/circuit-open audits grouped here until daily failure volume drops.
- Evidence:
  - `audit_20260703T223159069590Z_8543e95e26`: timed out; fast=correct vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T223159069590Z_8543e95e26/comparison.json)
  - `audit_20260703T144456547842Z_a8394ed135`: VLM audit circuit open after repeated timeouts; retry in 300s; types=vlm_unavailable (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T144456547842Z_a8394ed135/comparison.json)

### WB3-0022 - VLM audit schema errors return empty JSON

- Severity: `P1`
- Status: `triaged`
- Category: `audit-infra`
- Key: `vlm-audit-schema-empty-json`
- Last seen: `2026-07-04T12:13:30Z`
- Next action: Add schema-repair coverage and retry handling for VLM responses that omit latexLines, including empty JSON responses. Preserve raw response and parser error telemetry for every schema failure.
- Evidence:
  - `audit_20260703T125032301196Z_334c75e45e`: VLM JSON must include latexLines as a list; fast=correct vlm=None; types=vlm_schema_error (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T125032301196Z_334c75e45e/comparison.json)

### WB3-0023 - Audit overlay boxes are reported as visual marks

- Severity: `P2`
- Status: `fixture_needed`
- Category: `audit-process`
- Key: `audit-overlay-visual-mark-false-positive`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Add clean VLM crops or explicit overlay-ignore metadata/instructions so diagnostic red boxes are not interpreted as boxed-answer visual intent. Add fixture coverage before changing visual-intent comparison behavior.
- Evidence:
  - `audit_20260703T125227496485Z_5e0ddd9d04`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T125227496485Z_5e0ddd9d04/comparison.json)
  - `audit_20260703T130912395585Z_bbbd388b6e`: VLM audit raised: problem_status_mismatch, solution_set_mismatch, line_latex_mismatch; fast=incorrect vlm=correct; types=line_latex_mismatch,problem_status_mismatch,solution_set_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T130912395585Z_bbbd388b6e/comparison.json)

### WB3-0024 - Tutor feedback leaks prompt text or gives literal fixes

- Severity: `P2`
- Status: `triaged`
- Category: `tutor-feedback`
- Key: `tutor-feedback-prompt-leak-and-literal-feedback`
- Last seen: `2026-07-04T12:13:48Z`
- Next action: Add tutor-feedback prompt/output contract tests that reject raw prompt instruction leakage and require actionable, line-specific feedback instead of literal phrases like fix the error.
- Evidence:
  - `audit_20260703T213955419107Z_1abc4a8529`: Fast pipeline and VLM audit agreed on the checked signals.; fast=incorrect vlm=incorrect; types=visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T213955419107Z_1abc4a8529/comparison.json)
  - `audit_20260703T174233152665Z_de476ff2a5`: VLM JSON must include latexLines as a list; fast=incorrect vlm=None; types=vlm_schema_error (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174233152665Z_de476ff2a5/comparison.json)
  - `audit_20260703T174901022130Z_232e0fa7c8`: timed out; fast=incorrect vlm=None; types=vlm_timeout (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T174901022130Z_232e0fa7c8/comparison.json)

### WB3-0025 - Realtime segmentation status flips during grading

- Severity: `P2`
- Status: `triaged`
- Category: `recognition-ux`
- Key: `realtime-segmentation-status-flips`
- Last seen: `2026-07-04T12:13:55Z`
- Next action: Add instrumentation and tests for live segmentation status stability so a problem that has reached correct does not repeatedly flip to incomplete during late segmentation/resegmentation unless new evidence invalidates it.
- Evidence:
  - `audit_20260703T144210547539Z_fe7e799b0c`: timed out; fast=correct vlm=None; types=vlm_unavailable (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T144210547539Z_fe7e799b0c/comparison.json)

## Recently Verified

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

### WB3-0011 - Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `radical-index-root-structure-flattened`
- Last seen: `2026-07-04T12:13:12Z`
- Next action: Monitor live problem-input sqrt-fraction audits for x^{10}/x^3 recurrence.
- Evidence:
  - `audit_20260703T150500335747Z_76031958ee`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T160259082821Z_65b2bed88a`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T162040163106Z_29aa7258fe`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T162055470646Z_3ef1226787`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T214322728771Z_95c77652c8`: VLM audit raised: line_latex_mismatch, problem_input_ocr_mismatch; types=line_latex_mismatch,problem_input_ocr_mismatch (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T214322728771Z_95c77652c8/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/problem-input-sqrt-fraction-x10-over-x-cubed-brace-regression.json`
- Verification:
  - `passed` `node --test --test-name-pattern "distilled real handwriting trace fixtures|problem-input sqrt fraction" testing/test_recognition_pipeline.mjs`

### WB3-0020 - Problem-input multiplication dot is promoted to equals between fractions

- Severity: `P1`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `problem-input-multiplication-dot-promoted-to-equals`
- Last seen: `2026-07-04T12:13:12Z`
- Next action: Add a replay fixture for audit_20260703T213932311256Z_4580bfe0d6 before broad problem-input edits, then prevent evaluate-expression fraction inputs from promoting low-confidence equals candidates over centered dot/multiply candidates unless equation-mode geometry is strong.
- Evidence:
  - `audit_20260703T213932311256Z_4580bfe0d6`: VLM audit raised: line_latex_mismatch, problem_input_ocr_mismatch; types=line_latex_mismatch,problem_input_ocr_mismatch (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T213932311256Z_4580bfe0d6/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/problem-input-fraction-multiply-dot.json`
- Verification:
  - `passed` `node --test --test-name-pattern "evaluate problem-input fraction keeps|distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0017 - Equivalent radical/power final answer is not accepted as complete

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `equivalent-radical-decimal-final-marked-unsimplified`
- Last seen: `2026-07-04T12:13:12Z`
- Next action: Add a replay fixture for audit_20260703T162152180907Z_1815d9d8cb before broad grader edits, then accept exact equivalent power forms such as 2^{0.5} for sqrt(2) evaluate-expression finals.
- Evidence:
  - `audit_20260703T162152180907Z_1815d9d8cb`: VLM audit raised: line_latex_mismatch; fast=incorrect vlm=incorrect; types=line_latex_mismatch,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T162152180907Z_1815d9d8cb/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/sqrt-two-power-half-final.json`
- Verification:
  - `passed` `python3 -m unittest testing.test_equation_grader && node --test --test-name-pattern "distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0019 - Superscript 1/2 exponent is read as a square-root exponent in simplification work

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `superscript-half-exponent-read-as-radical`
- Last seen: `2026-07-04T12:13:12Z`
- Next action: Add a replay fixture for audit_20260703T211024481616Z_9ac973d64b before broad layout repair edits, then conservatively repair compact superscript 1/2 patterns in simplify contexts while preserving lower rows as separate lines.
- Evidence:
  - `audit_20260703T211024481616Z_9ac973d64b`: VLM audit raised: problem_status_mismatch, simplification_policy_disagreement, line_count_mismatch, line_latex_mismatch; fast=incorrect vlm=incomplete; types=line_count_mismatch,line_latex_mismatch,problem_status_mismatch,simplification_policy_disagreement,visual_intent_observed (/Users/jpovj/Documents/dev/log_whiteboard_3/2026-07-03/audit_20260703T211024481616Z_9ac973d64b/comparison.json)
- Fixtures:
  - `testing/fixtures/real_handwriting/sqrt-x10-half-exponent-simplification.json`
- Verification:
  - `passed` `node --test --test-name-pattern "radical simplification half exponent|distilled real handwriting trace fixtures" testing/test_recognition_pipeline.mjs`

### WB3-0001 - Candidate/grading path overtrusts ambiguous final-answer OCR

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `candidate-selection-ambiguous-final-answer-overtrust`
- Last seen: `2026-07-02T13:34:25.159495Z`
- Next action: Keep candidateSelectionSafeForGrading and restricted Python candidate payload behavior. Maintain replay tests for the ambiguous final-answer audits before marking resolved.
- Evidence:
  - `audit_20260702T132349514705Z_0da52d6b90`: Candidate/grading path overtrusts ambiguous final-answer OCR ()
  - `audit_20260702T133425158949Z_a1e84136a1`: Candidate/grading path overtrusts ambiguous final-answer OCR ()

### WB3-0002 - Standalone operation repair rewrites numeric literals

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `standalone-operation-repair-overfires-on-numeric-literal`
- Last seen: `2026-07-02T13:28:23.822421Z`
- Next action: Keep the plain numeric literal guard in repairStandaloneOperationLatex and exact fixture coverage for 2 / spaced decimals.
- Evidence:
  - `audit_20260702T132601015622Z_44d23b674d`: Standalone operation repair rewrites numeric literals ()
  - `audit_20260702T132606546543Z_9cc8f69a51`: Standalone operation repair rewrites numeric literals ()
  - `audit_20260702T132819834470Z_44f1a28ce5`: Standalone operation repair rewrites numeric literals ()

### WB3-0003 - Superscript layout is flattened into row text

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `exponent-layout-flattened-to-rows`
- Last seen: `2026-07-02T13:28:23.822421Z`
- Next action: Keep the superscript-line candidate family and compact/equation-base guards; add post-fix replay evidence for the original exponent audit before marking resolved.
- Evidence:
  - `audit_20260702T132606546543Z_9cc8f69a51`: Superscript layout is flattened into row text ()
  - `audit_20260702T132819834470Z_44f1a28ce5`: Superscript layout is flattened into row text ()

### WB3-0008 - Compact monomial exponent is read as multiplication or a baseline numeral

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `compact-monomial-exponent-omitted-after-variable`
- Last seen: `2026-07-02T19:44:43.536772Z`
- Next action: Implemented in current worktree; keep regression fixture and wait for comparable post-fix audit evidence before marking likely resolved.
- Evidence:
  - `audit_20260702T194443536522Z_07b99ef859`: Compact monomial exponent is read as multiplication or a baseline numeral ()

### WB3-0009 - Change-of-base log denominators lose their base numerals

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `log-base-denominator-ocr-collapses-base-numerals`
- Last seen: `2026-07-02T19:57:17.639094Z`
- Next action: Implemented in current worktree; keep regression fixture and wait for comparable post-fix audit evidence before marking likely resolved.
- Evidence:
  - `audit_20260702T195717638774Z_b8112da14e`: Change-of-base log denominators lose their base numerals ()

## Workflow

Use `python3 scripts/issue_ledger.py --help` to create, update, transition, and render ledger issues. Do not hand-edit this file; run `python3 scripts/issue_ledger.py render` after changing issue state.