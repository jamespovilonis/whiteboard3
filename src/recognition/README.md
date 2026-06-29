# Recognition Pipeline

The current pipeline starts with line segmentation. It accepts the strokes in a
dynamic answer box and emits overlapping hypotheses instead of committing to one
split too early.

## Line Segmentation

`lineSegmentation.js` builds several candidate families:

- `parent`: the original unsplit answer crop.
- `loose`, `strict`, `temporal`: geometric groupings inspired by the
  `whiteboard_2` segmenters.
- `row-line`: vector row clustering with structural merges for fractions,
  scripts, limits, and local underline/operation bars.
- `dbnet-parent`, `dbnet-line`: optional external detector evidence, if a
  DBNet-style endpoint supplies detections.
- `fallback-stroke`: a last-resort exact-cover candidate.

The selector then chooses the highest-scoring exact cover of all strokes. This
keeps fraction parents available while still preferring separate true solution
lines when child rows cover the same ink more plausibly. When OCR/semantic
scores are added later, the selector receives the initial geometry cover as a
baseline. Evidence can still merge or split lines, but a proposed cover is
rejected if it would explode the line count or fragment a structural
fraction-like line into tiny crops. The same baseline gate also rejects
coalescing multiple geometry-selected rows into an unread parent crop unless
the parent has strong OCR or semantic evidence.
Dense detector bands that accidentally merge adjacent worked rows are scored
against any independent child rows that cover the same ink, so DBNet evidence
can help without forcing a merged line.

## Semantic Scoring

`testing/latex_semantics.py` is the SymPy-backed sidecar for OCR candidate
scoring. It checks parse soundness, character continuity with the problem and
previous lines, and symbolic equivalence when the candidate and reference are
equations or expressions SymPy can understand.
It also handles a common worked-step pattern where a previous symbolic function
line is evaluated at a value, for example ranking `f'(2)=3/4` above nearby OCR
confusions when the previous line was `f'(x)=(x^2-1)/x^2`.
For structurally unsound top candidates, it can add narrow repair variants
before scoring; currently this includes replacing a trailing `1` with `)` only
when that change balances an otherwise unparseable expression.
For short final-value continuations such as `= q`, the scorer can also derive
a numeric `= ...` repair from the previous recognized expression when SymPy
proves the repaired value is equivalent to that previous line.

Operation annotations such as `/ 4 / 4`, `\times 6 \times 6`, and `- 3 - 3`
are classified as sound math-line annotations. They are not treated as
equivalent equations, but they also are not penalized as malformed OCR.

This is intentionally not an answer checker. It does not load hard-coded
solutions; it only uses the problem equation and recognized history as context
for ranking CoMER's top candidates and diagnosing likely bad segmentations.
Synthetic fixtures carry that equation explicitly as `problemLatex`, separate
from the expected handwritten rows, so a student's first visible line can be an
already-transformed equation.
In the browser app, previous-line context is scoped to the current submitted
answer as the selected lines are recognized in reading order. Completed answers
from earlier independent problems remain visible in the shell, but they are not
fed into the next problem's semantic context.

## Browser OCR Shell

`studentWritingPipeline.js` wires the first end-to-end path:

1. Rasterize the full dynamic answer box and optionally POST it to a
   DBNet-compatible `/segment-lines` endpoint.
2. Segment strokes inside the answer box using vector geometry plus any detector
   bands returned in board coordinates.
3. Rasterize candidate line crops with `lineRasterizer.js`.
4. POST each candidate crop to a CoMER-compatible `/recognize` endpoint via
   `ocrClient.js`.
5. Optionally POST CoMER candidate lists to `/score-latex-candidates` for
   SymPy-backed soundness/equivalence scoring.
6. Re-select the exact stroke cover with geometry plus OCR/semantic evidence,
   gated by the first geometry-selected cover.
7. Re-score competing candidates that have plausible earlier same-answer lines
   above them, then re-run the exact-cover selector once.
8. Retry selected lines at alternate normalized raster heights when CoMER times
   out, returns no LaTeX, or produces a suspicious operation annotation.
9. Re-score the selected lines in reading order, feeding earlier lines from the
   same answer back as semantic context.
10. For wide selected lines that still fail, split the line at large horizontal
   gaps, classify isolated equals signs geometrically, OCR the remaining chunks,
   and concatenate the chunk LaTeX.
11. Return ordered LaTeX lines plus the top candidate lists and elapsed times.

The first geometry pass gives us a candidate graph and a stable baseline. The
OCR pass can then let a single structural parent beat its child rows, or let
confident child rows beat a large multi-line parent. It cannot turn one
structural line into many tiny symbol crops just because CoMER dislikes the
whole-line image, and it cannot collapse several readable rows into a parent
crop when the parent OCR is missing or empty. Failed or timed-out OCR calls
degrade to geometry-only selection so the app can still preserve segmentation
metadata when CoMER is offline.

OCR crops are normalized to a CoMER-friendly height before the first request,
then selected-line retries try alternate heights only when the first result is
empty, timed out, structurally suspicious, or a low-semantic function/equation
read that may improve at another crop height. Selected-line retries are
intentionally narrower than candidate recognition. They do not influence the
segmentation cover. They only improve final LaTeX output for the chosen lines,
using normalized crop heights that CoMER handles better on tall fractions,
derivative quotient rows, and operation annotations. If all whole-line attempts
fail on a wide line, the chunk fallback is used only for final output; it does
not change the segmentation cover. Semantic-best LaTeX only overwrites the
current line when the evidence is strong: equivalence to the problem or
previous line, a high semantic score, or a repair from an unsound current top
candidate. The same conservative rule is used when deciding which recognized
line to feed into the next line's semantic context.

By default the browser posts OCR, detector, and semantic-scoring requests to the
current page hostname on port `8010`, matching the local recognition gateway. Set
`VITE_OCR_API_URL` to override this, for example
`VITE_OCR_API_URL=http://localhost:8010 npm run dev`.

The semantic endpoint is optional, but it is most useful when it sits beside the
model endpoints. During local experiments, run the CoMER/DBNet API on port
`8000`, then start the standard-library gateway:

```sh
python3 testing/semantic_score_server.py \
  --port 8010 \
  --upstream-api-url http://127.0.0.1:8000 \
  --semantic-timeout 2.5
```

Point the app at the gateway:

```sh
VITE_OCR_API_URL=http://127.0.0.1:8010 npm run dev
```

The gateway serves `/score-latex-candidates` locally and proxies `/recognize`,
`/segment-lines`, and health checks to the model API, so the browser can use one
base URL for the whole recognition pipeline. Semantic scoring runs behind a
request-level timeout so pathological SymPy comparisons fail closed instead of
blocking OCR.

## Live Evaluation

The deterministic tests use synthetic strokes and mocked model calls. To probe
the real server path, start the CoMER/DBNet API and run:

```sh
python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8000
```

The runner keeps the current broad candidate strategy intact: it asks DBNet for
bands on the full answer crop, lets the JS segmenter form parent and child line
hypotheses, recognizes candidate alternatives with CoMER, re-runs exact-cover
selection with semantic evidence and the conservative geometry baseline gate,
and reports whether top or semantic-best candidates match the fixture labels.
For selected candidates, failed or suspicious OCR is retried at normalized
heights to measure the same final-output path used by the browser. Wide
selected lines that still fail are split into horizontal chunks as a final OCR
fallback, matching the browser's conservative long-line behavior.
Use `--geometry-only-selection` when you want to compare against the initial
geometry-selected crops only. Semantic-rescoring sweeps include same-answer
context from earlier line-like candidates, matching the browser path that uses
previous student rows to score later rows. For broad semantic-rescoring sweeps, use
`--max-candidate-alternatives N` or `--candidate-time-budget-seconds N` to cap
extra non-selected candidate OCR, and add `--extra-candidate-timeout-seconds N`
when speculative parent/child crops should get a shorter per-request CoMER
timeout. The selected geometry cover is always recognized first with the full
line timeout. With CoMER enabled, OCR misses fail the run by default; use
`--allow-ocr-misses` only for exploratory collection where misses should be
reported but not gate the process. Client-side CoMER socket timeouts are
recorded as OCR misses instead of aborting the matrix. Add `--gap-pattern` or
`--line-gaps` to stress uneven student line spacing with the same layouts used
by the offline segmentation matrix. Long live runs checkpoint the summary after
each fixture record, so interrupted CoMER sweeps still leave the completed
records in the configured `--write-summary` file.

For accuracy-gated sweeps, add explicit minimum rates:

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
