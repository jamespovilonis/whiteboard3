# Whiteboard 3 Issue Ledger

Generated from `issues.json` at `2026-07-03T22:02:46Z`.

## Actionable Issues

### WB3-0011 - Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions

- Severity: `P1`
- Status: `reopened`
- Category: `recognition`
- Key: `radical-index-root-structure-flattened`
- Last seen: `2026-07-03T21:43:52.351771Z`
- Next action: Strengthen problem-input sqrt-fraction candidate ranking so exponent-preserving candidates such as x^{1 0} beat brace-repaired x^{0} candidates even when the latter has synthetic confidence 1; add replay fixture for audit_20260703T214322728771Z_95c77652c8.
- Evidence:
  - `audit_20260703T150500335747Z_76031958ee`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T160259082821Z_65b2bed88a`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T162040163106Z_29aa7258fe`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T162055470646Z_3ef1226787`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()
  - `audit_20260703T214322728771Z_95c77652c8`: Indexed radicals and root wrappers are flattened, malformed, or discarded in problem-input fractions ()

### WB3-0020 - Problem-input multiplication dot is promoted to equals between fractions

- Severity: `P1`
- Status: `triaged`
- Category: `recognition`
- Key: `problem-input-multiplication-dot-promoted-to-equals`
- Last seen: `2026-07-03T21:40:01.238215Z`
- Next action: Do not allow full-parent problem-input repair to promote a lower-confidence equals candidate over dot/multiply candidates for evaluate-expression input unless equation mode or strong equals geometry is present; prefer \cdot/. candidates between adjacent fractions when the problem input mode is evaluate-expression and the visual mark is a small centered dot.
- Evidence:
  - `audit_20260703T213932311256Z_4580bfe0d6`: Problem-input multiplication dot is promoted to equals between fractions ()

## Recently Verified

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

### WB3-0004 - Detached/circled annotation is selected and graded as a final answer line

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `detached-operation-annotations-selected-as-answer-lines`
- Last seen: `2026-07-03T14:13:05.020876Z`
- Next action: Treat isolated circle/box-like strokes and detached operation marks as annotation candidates first, not answer-line candidates, unless they contain interior glyph strokes or are textually connected to an equation baseline. Exclude such annotation candidates from grading and preserve them only as annotationAttachments/visual marks. Add a regression fixture for audit_20260703T141214622367Z_00171da20d asserting latexLines stop at the unsimplified algebra rows and problemStatus remains incomplete, not correct.
- Evidence:
  - `audit_20260702T133425158949Z_a1e84136a1`: Detached/circled annotation is selected and graded as a final answer line ()
  - `audit_20260703T141214622367Z_00171da20d`: Detached/circled annotation is selected and graded as a final answer line ()

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

### WB3-0010 - Valid semantic OCR candidates are present but not promoted to accepted/final grading lines

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `valid-semantic-alternate-not-promoted`
- Last seen: `2026-07-03T14:24:16.291076Z`
- Next action: Current worktree now promotes safe grader-selected candidates consistently; add or keep a regression asserting a top-five `9` candidate replaces an OCR top `q` when gradeWork selects the numeric solution, then wait for post-fix audit evidence before marking resolved.
- Evidence:
  - `audit_20260702T214417603075Z_2cc1598b2a`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260702T220020765884Z_71983a33ea`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T125227496485Z_5e0ddd9d04`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T130912395585Z_bbbd388b6e`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()
  - `audit_20260703T142416290747Z_ff29f8b9d9`: Valid semantic OCR candidates are present but not promoted to accepted/final grading lines ()

### WB3-0013 - Finalization budget can return blank selected lines after OCR timeout

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `finalization-budget-blank-output`
- Last seen: `2026-07-02T21:58:13.615624Z`
- Next action: When budget is exhausted and selected covers have strokes but no OCR candidates, run one bounded OCR fallback or return last non-empty partial; classify as incomplete/timeout, not not_started.
- Evidence:
  - `audit_20260702T215813615519Z_3622a67128`: Finalization budget can return blank selected lines after OCR timeout ()

### WB3-0014 - Problem-input fraction structure loses the numerator/denominator relationship

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `problem-input-fraction-structure-overwraps-numerator`
- Last seen: `2026-07-03T13:59:04.053650Z`
- Next action: Keep the structural fraction merge behavior, but narrow the follow-up latex repair path tracked by problem-input-fraction-repair-overstrips-tail-and-denominator so it cannot strip valid problem-input equation tails or multi-token denominators.
- Evidence:
  - `audit_20260702T220328773710Z_d0ddea889f`: Problem-input fraction structure loses the numerator/denominator relationship ()
  - `audit_20260703T124722130237Z_0c402355aa`: Problem-input fraction structure loses the numerator/denominator relationship ()
  - `audit_20260703T135904053464Z_fff4e46f6d`: Problem-input fraction structure loses the numerator/denominator relationship ()

### WB3-0015 - Problem-input fraction repair over-strips numerators/denominators and equation tails

- Severity: `P2`
- Status: `verified_by_replay`
- Category: `recognition`
- Key: `problem-input-fraction-repair-overstrips-tail-and-denominator`
- Last seen: `2026-07-03T15:03:43.943976Z`
- Next action: Keep the narrowed outer-fraction repair guard and replay tests; track the new merge-overwrite bug separately.
- Evidence:
  - `audit_20260703T145757428980Z_f097b2eb7b`: Problem-input fraction repair over-strips numerators/denominators and equation tails ()
  - `audit_20260703T145847809653Z_2c7ca1aee2`: Problem-input fraction repair over-strips numerators/denominators and equation tails ()
  - `audit_20260703T145944835679Z_f48367f5a6`: Problem-input fraction repair over-strips numerators/denominators and equation tails ()
  - `audit_20260703T150343943838Z_76819e33e8`: Problem-input fraction repair over-strips numerators/denominators and equation tails ()

## Workflow

Use `python3 scripts/issue_ledger.py --help` to create, update, transition, and render ledger issues. Do not hand-edit this file; run `python3 scripts/issue_ledger.py render` after changing issue state.