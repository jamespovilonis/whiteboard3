# Synthetic Math Fixtures

This folder is isolated from the whiteboard app runtime. It contains pure Python
fixture generation for handwritten-looking math boards and does not import
frontend, browser, OCR, DBNet, CoMER, server, or model code.

## Setup

```sh
python3 -m pip install -r testing/requirements.txt
```

## Render Fixtures

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

Generated PNG and JSON files are written only under `testing/results/` and are
ignored by git.

Each fixture JSON stores the prompt equation as `fixture.problemLatex` and the
student-written rows as `fixture.expectedLatexLines`. Most fixtures repeat the
prompt as the first student row, but `algebra_prompt_context` keeps them
separate to exercise semantic scoring against the actual problem context.

## Tests

```sh
python3 -m unittest discover testing
```

Run the broader offline segmentation matrix:

```sh
npm run test:segmentation:matrix
```

The matrix covers every synthetic fixture, spacing profile, stroke order, and a
small seed set. It is slower than the normal regression suite, but useful before
changing line-splitting heuristics.

To stress line segmentation with custom uneven spacing, add named gap patterns
or an explicit gap list:

```sh
python3 testing/run_segmentation_matrix.py --all \
  --spacing dense --spacing tight-steps \
  --gap-pattern accordion --gap-pattern pinched-middle --gap-pattern stair-step \
  --ink-style compact --ink-style messy \
  --order interleaved-lines --order reverse-lines \
  --seed 616 --seed 1720
```

Use `--line-gaps 4,64,2` for a single selected problem whose line count needs
exactly three gaps.

## Live Recognition Matrix

When a CoMER/DBNet API is running, exercise the full detector-to-OCR path:

```sh
python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8000
```

For a detector-only smoke run:

```sh
python3 testing/run_live_recognition_matrix.py --skip-comer --api-url http://127.0.0.1:8000
```

The live runner renders varied synthetic boards, posts each board to
`/segment-lines`, feeds those detections into the JS segmenter, posts candidate
line alternatives to `/recognize`, then scores the top-five OCR candidates with
the SymPy semantic helper. By default it re-runs the exact-cover selector with
those candidate scores and the first geometry-selected cover as a conservative
baseline before reporting the final OCR lines. Candidate rescoring includes
same-answer context from earlier line-like candidates, which lets later rows use
the problem statement and previous student rows during selection. Selected
candidates whose OCR times out, returns no LaTeX, or looks like a suspicious
operation annotation are sent at a normalized initial crop height and retried at
alternate normalized heights. Wide selected lines that still fail are split into
horizontal chunks, with isolated equals signs classified geometrically, before
the chunk LaTeX is concatenated. Pass `--geometry-only-selection` to recognize
only the first geometry-selected crops. Add `--progress` when diagnosing slow
CoMER reads; it prints each crop before sending it to the model. For broader live sweeps that
still use semantic rescoring, pass `--max-candidate-alternatives N` or
`--candidate-time-budget-seconds N` to bound extra non-selected candidate OCR;
add `--extra-candidate-timeout-seconds N` to shorten each speculative CoMER
request while keeping the selected geometry candidates on the full timeout. The
geometry-selected candidates are always recognized first. CoMER request timeout
arguments are clamped to the API's accepted maximum of 20 seconds, so an overly
large exploratory timeout does not turn the OCR sweep into validation errors.
When CoMER is
enabled, the process fails if any selected line lacks either a matching top
prediction or a matching semantic-best prediction. Pass `--allow-ocr-misses`
for exploratory data collection without that failure gate. Client-side CoMER
socket timeouts are recorded as OCR misses so the suite can still report
partial results. The live runner accepts the same `--gap-pattern` and
`--line-gaps` layout options as the offline matrix. It writes PNG/JSON fixtures,
crops, and a summary under `testing/results/live_recognition/`. During long
runs, the summary is checkpointed after each completed fixture record and
marked as `running` until the final `complete` summary is written.

For accuracy-gated sweeps, add explicit minimum rates. For example, this fails
unless segmentation, final selected-line coverage, accepted strict OCR, and
accepted OCR all stay perfect on the selected fixtures:

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

## Local Recognition API

To exercise OCR, segmentation, and semantic scoring through one browser API URL,
start the CoMER/DBNet server, then run:

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

The FastAPI backend handles `/score-latex-candidates` and
`/grade-equation-work` directly and proxies `/recognize`, `/segment-lines`,
`/health`, and `/segment-lines/health` to the upstream model server. The
semantic timeout keeps malformed or unusually complex CoMER candidates from
blocking the browser pipeline.
