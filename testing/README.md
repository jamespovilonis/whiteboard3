# Testing And Fixture Harnesses

This folder contains isolated Python and Node test utilities for recognition, grading, audit, feedback, and handwriting fixtures. Runtime app code does not import fixture generators; tests and scripts import app modules where needed.

## Setup

```sh
python3 -m pip install -r testing/requirements.txt
npm install
```

## Common Checks

Run the broad Python test suite:

```sh
python3 -m unittest discover testing
```

Run the focused recognition/segmentation regression suite used before most recognition changes:

```sh
npm run test:segmentation
```

That command runs Python unit tests for real-handwriting fixtures, realistic handwriting, line segmentation, LaTeX semantics, and equation grading, then runs the Node recognition pipeline tests in `testing/test_recognition_pipeline.mjs`.

Other useful checks:

```sh
npm run build
npm run test:e2e
npm run test:e2e:real:smoke
npm run test:e2e:real:matrix
npm run test:e2e:real:traces
```

Real OCR E2E commands require the FastAPI gateway on `http://127.0.0.1:8010`.

## Synthetic Math Fixtures

Render a single mixed-spacing multi-step algebra board:

```sh
python3 testing/render_math_fixture.py --problem algebra_steps --spacing mixed
```

Render a derivative board with explicit non-uniform line gaps:

```sh
python3 testing/render_math_fixture.py --problem derivative_evaluate --line-gaps 64,12,48
```

Render all integral fixtures across every spacing profile:

```sh
python3 testing/render_math_fixture.py --family integral --all-spacings
```

Generated PNG and JSON files are written under `testing/results/` and are ignored by git, except for a few committed demo artifacts. Each fixture JSON stores the prompt equation as `fixture.problemLatex` and the student-written rows as `fixture.expectedLatexLines`. Most fixtures repeat the prompt as the first student row, but `algebra_prompt_context` keeps them separate to exercise semantic scoring against the actual problem context.

## Real Handwriting Trace Fixtures

The committed fixtures under `testing/fixtures/real_handwriting/` are distilled from local VLM audit logs. They preserve anonymized board-space strokes, relative timing, reviewed line groups, visual-mark metadata, and both the fast pipeline transcript and VLM transcript.

Regenerate curated seed drafts from the local audit log directory:

```sh
npm run fixtures:real-handwriting
```

The distiller reads `WHITEBOARD_AUDIT_LOG_DIR` when set, otherwise it uses the local default audit log path. Review `expectedLineGroups` before committing a new trace; those groups are the CI-gated segmentation expectation. The normal segmentation suite consumes these fixtures directly:

```sh
npm run test:segmentation
```

To replay committed traces through the browser with the real OCR gateway:

```sh
npm run test:e2e:real:traces
```

To inspect the local audit log corpus without treating stale logged OCR as current behavior:

```sh
npm run audit:real-handwriting
```

This command regrades logged fast OCR transcripts and separately runs the current JS segmenter on raw strokes, so mismatches can be sorted into stale OCR, grading-policy, and current-segmentation buckets.

## Realistic And Hybrid Handwriting

For generated stress fixtures calibrated to real user input, use the realistic stroke harness instead of the LaTeX-to-contour renderer:

```sh
npm run handwriting:calibration
```

This Phase 1 report scans local audit `input.json` records when available and falls back to committed distilled traces in CI. It summarizes stroke count, point count, pressure, timing gaps, stroke and line geometry, answer/board extents, visual-mark rates, and non-sequential writing behavior.

Save a calibration report:

```sh
python3 testing/realistic_handwriting.py \
  --calibration-only \
  --audit-limit 100 \
  --calibration-output testing/results/real-handwriting-calibration.json
```

Create a generated stress fixture from real strokes:

```sh
python3 testing/realistic_handwriting.py \
  --scenario mixed-marks \
  --seed 17 \
  --audit-limit 50 \
  --output testing/results/generated-mixed-marks.json
```

Generate a linear-equation trace from the curated real-stroke atom catalog:

```sh
npm run handwriting:hybrid-linear -- \
  --random-linear \
  --seed 31 \
  --include-crossout \
  --output testing/results/hybrid_handwriting/linear.json
```

Generate a broader complex-math trace from the same catalog:

```sh
npm run handwriting:hybrid-complex -- \
  --seed 41 \
  --include-crossout \
  --output testing/results/hybrid_handwriting/complex.json
```

Generate log-observed harness packs when a test needs real failure-mode shape rather than a newly solved equation:

```sh
npm run handwriting:hybrid-pack -- \
  --hybrid-pack failure-modes \
  --seed 5 \
  --output testing/results/hybrid_handwriting/failure-modes.json
```

Available packs cover known discrepancy templates, handwritten problem-input OCR, ambiguous fraction structures, visual-intent marks, non-sequential writing, and bad handwriting that still represents valid math. Each pack emits `oracleContracts`, so end-to-end tests can assert recognition, segmentation, visual-intent, and grading behavior independently.

Write a generated-vs-real dashboard report:

```sh
npm run handwriting:dashboard -- \
  --dashboard-output testing/results/hybrid_handwriting/generator-dashboard.json
```

## Offline Segmentation Matrix

Run the broader offline segmentation matrix:

```sh
npm run test:segmentation:matrix
```

The matrix covers every synthetic fixture, spacing profile, stroke order, and a small seed set. It is slower than the normal regression suite, but useful before changing line-splitting heuristics.

Stress line segmentation with custom uneven spacing:

```sh
python3 testing/run_segmentation_matrix.py --all \
  --spacing dense --spacing tight-steps \
  --gap-pattern accordion --gap-pattern pinched-middle --gap-pattern stair-step \
  --ink-style compact --ink-style messy \
  --order interleaved-lines --order reverse-lines \
  --seed 616 --seed 1720
```

Use `--line-gaps 4,64,2` for a single selected problem whose line count needs exactly three gaps.

## Live Recognition Matrix

When a CoMER/DBNet API is running, exercise the full detector-to-OCR path:

```sh
python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8000
```

To test through the FastAPI gateway:

```sh
python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8010
```

For a detector-only smoke run:

```sh
python3 testing/run_live_recognition_matrix.py --skip-comer --api-url http://127.0.0.1:8000
```

The live runner renders varied synthetic boards, posts each board to `/segment-lines`, feeds detections into the JS segmenter, posts candidate line alternatives to `/recognize`, then scores top-five OCR candidates with the SymPy semantic helper. By default it re-runs exact-cover selection with those candidate scores and the first geometry-selected cover as a conservative baseline before reporting final OCR lines.

Selected candidates whose OCR times out, returns no LaTeX, or looks like a suspicious operation annotation are sent at a normalized initial crop height and retried at alternate normalized heights. Wide selected lines that still fail are split into horizontal chunks, with isolated equals signs classified geometrically, before chunk LaTeX is concatenated.

Pass `--geometry-only-selection` to recognize only the first geometry-selected crops. Add `--progress` when diagnosing slow CoMER reads. For broader live sweeps that still use semantic rescoring, pass `--max-candidate-alternatives N` or `--candidate-time-budget-seconds N` to bound extra non-selected candidate OCR; add `--extra-candidate-timeout-seconds N` to shorten speculative CoMER requests while keeping selected geometry candidates on the full timeout. CoMER request timeout arguments are clamped to the API's accepted maximum of 20 seconds.

When CoMER is enabled, the process fails if any selected line lacks either a matching top prediction or a matching semantic-best prediction. Pass `--allow-ocr-misses` for exploratory data collection. Client-side CoMER socket timeouts are recorded as OCR misses so the suite can still report partial results. The live runner accepts the same `--gap-pattern` and `--line-gaps` layout options as the offline matrix. It writes PNG/JSON fixtures, crops, and a checkpointed summary under `testing/results/live_recognition/`.

For accuracy-gated sweeps:

```sh
python3 testing/run_live_recognition_matrix.py \
  --api-url http://127.0.0.1:8010 \
  --spacing dense \
  --gap-pattern pinched-middle \
  --ink-style messy \
  --order interleaved-lines \
  --max-candidate-alternatives 1 \
  --extra-candidate-timeout-seconds 8 \
  --min-segmentation-exact-rate 1 \
  --min-pipeline-exact-rate 1 \
  --min-accepted-strict-rate 1 \
  --min-accepted-match-rate 1
```

## Feedback, Audit, And Ledger Tests

- `testing/test_feedback_service.py` verifies prompt-context construction, target-line selection, Ollama response parsing, LLM rejection rules, and deterministic fallback text.
- `testing/test_audit_service.py` verifies audit artifacts, VLM normalization, feedback attachment, metadata, latency/circuit behavior, and discrepancy classification.
- `testing/test_issue_ledger.py` verifies `scripts/issue_ledger.py` behavior for the canonical ledger under `ops/issue-ledger/`.

For P0/P1 bug work, start from an issue ID in `ops/issue-ledger/issues.json`, add or confirm a replay fixture before broad source edits, and record verification commands through `scripts/issue_ledger.py`.

## Local Recognition API

To exercise OCR, segmentation, semantic scoring, grading, feedback, and audit through one browser API URL, start the CoMER/DBNet server, then run:

```sh
python3 -m src.server.app \
  --port 8010 \
  --upstream-api-url http://127.0.0.1:8000 \
  --semantic-timeout 2.5
```

Then start the app with:

```sh
VITE_API_URL=http://127.0.0.1:8010 npm run dev
```

The FastAPI backend handles `/score-latex-candidates`, `/grade-equation-work`, `/grade-math-work`, `/feedback/math-work`, `/audit-recognition`, `/audit-recognition-feedback`, `/audit-recognition-note`, and `/gateway/health` directly. It proxies `/recognize`, `/segment-lines`, `/segment-lines/*`, `/health`, and `/segment-lines/health` to the upstream model server. The semantic timeout keeps malformed or unusually complex CoMER candidates from blocking the browser pipeline.
