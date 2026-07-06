# Whiteboard 3

Whiteboard 3 is a Vite + React math whiteboard. React owns application state and UI composition; plain JavaScript modules own canvas drawing, stroke smoothing, erasing, undo/redo, viewport math, recognition handoff, and problem-flow geometry. A local FastAPI gateway handles OCR/model proxying, semantic scoring, grading, VLM audit logging, and short math feedback.

## Project Layout

- `src/`: frontend app, whiteboard engine, recognition clients, grading clients, feedback client, and FastAPI backend.
- `src/whiteboard/`: canvas engine, renderer, stroke store, smoothing, geometry, and viewport helpers.
- `src/recognition/`: browser recognition pipeline, line segmentation, OCR/semantic/audit clients, rasterization, and latency telemetry.
- `testing/`: Python and Node regression suites, synthetic and real-handwriting fixtures, live OCR matrix runners, audit summarizers, and feedback/audit service tests.
- `e2e/`: Playwright flows for mocked recognition, custom problems, real OCR smoke/matrix runs, temporal writing, and real-handwriting traces.
- `ops/issue-ledger/`: canonical bug ledger used by audit, implementation, verification, and weekly-status work.
- `scripts/`: local server orchestration and issue-ledger maintenance commands.

## Run Locally

Install frontend dependencies once:

```sh
npm install
```

Install Python test/server dependencies when working on recognition, grading, audit, or feedback:

```sh
python3 -m pip install -r testing/requirements.txt
```

Start only the Vite app:

```sh
npm run dev
```

The app is usually served at `http://localhost:5500/`. Vite may choose another port if `5500` is busy.

Start the legacy OCR server from `../whiteboard_2`, the Whiteboard FastAPI gateway, and Vite together:

```sh
npm run dev:all
```

`dev:all` starts:

- OCR/DBNet/CoMER upstream on `http://127.0.0.1:8000`
- Whiteboard API gateway on `http://127.0.0.1:8010`
- Vite on `http://localhost:5500`

Set `WHITEBOARD_2_DIR=/path/to/whiteboard_2` if the OCR server does not live next to this repo. Stop old local servers with:

```sh
npm run dev:stop
```

## Recognition API

The browser defaults to the current page hostname on port `8010`. Override it with:

```sh
VITE_API_URL=http://127.0.0.1:8010 npm run dev
```

`VITE_OCR_API_URL` is still accepted as a compatibility fallback.

To run the gateway manually after starting a CoMER/DBNet API on port `8000`:

```sh
python3 -m src.server.app \
  --port 8010 \
  --upstream-api-url http://127.0.0.1:8000 \
  --semantic-timeout 2.5
```

The gateway provides local `/score-latex-candidates`, `/grade-equation-work`, `/grade-math-work`, `/feedback/math-work`, `/audit-recognition`, `/audit-recognition-feedback`, `/audit-recognition-note`, and `/gateway/health` routes. It proxies `/recognize`, `/segment-lines`, `/segment-lines/*`, and `/health` to the configured upstream OCR server.

Audit defaults are configured with `WHITEBOARD_AUDIT_*` and `VLM_AUDIT_*` environment variables. Math feedback defaults to Ollama at `http://127.0.0.1:11434` with model `qwen3:1.7b`; override with `WHITEBOARD_FEEDBACK_BASE_URL`, `WHITEBOARD_FEEDBACK_MODEL`, and `WHITEBOARD_FEEDBACK_TIMEOUT_SECONDS`.

## Math Feedback JSON

When submitted recognition and grading settle, the browser sends this JSON shape to `POST /feedback/math-work`:

```json
{
  "problemId": "problem-a",
  "problemLatex": "2x + 3 = 11",
  "problemMetadata": {
    "problemType": "equation-solving"
  },
  "inputSignature": "sig-a",
  "attemptId": "attempt_1234abcd",
  "grading": {
    "status": "complete",
    "failed": false,
    "problem": {},
    "steps": [],
    "result": {
      "problemStatus": "incorrect"
    }
  },
  "fastResult": {
    "latex": "2x = 8 \\\\ x = 5",
    "latexLines": ["2x = 8", "x = 5"],
    "grading": {},
    "timing": {},
    "detection": {},
    "semantic": {},
    "realtime": {},
    "lines": [],
    "candidatePredictions": [],
    "selectionSummary": {},
    "annotationAttachments": [],
    "segmentation": {
      "selected": [],
      "candidates": [],
      "partitions": {},
      "parentCandidateId": null,
      "ocrSelectedCandidateIds": []
    }
  }
}
```

The feedback service responds with `status`, `source`, `text`, `model`, `promptVersion`, `createdAt`, `targetLine`, `targetLineSource`, `targetLineReason`, and `promptContext`; fallback/error responses also include `error` and `rejectionReason`. Correct submissions are deterministic and skip the LLM.

## Testing

Common checks:

```sh
npm run build
npm run test:segmentation
python3 -m unittest discover testing
npm run test:e2e
```

Real OCR browser tests require the gateway on `8010`:

```sh
npm run test:e2e:real:smoke
npm run test:e2e:real:matrix
npm run test:e2e:real:traces
```

For live OCR pipeline sweeps:

```sh
python3 testing/run_live_recognition_matrix.py --api-url http://127.0.0.1:8010
```

See `testing/README.md` for fixture generation, real-handwriting harnesses, live matrix options, and audit-log summarizers.

## Chromebook Access On The Same Wi-Fi

The dev script binds Vite to `0.0.0.0`, so another device on the same network can open it.

1. Start the dev server with `npm run dev` or `npm run dev:all`.
2. Find the Mac's Wi-Fi IP address:

```sh
ipconfig getifaddr en0
```

3. On the Chromebook, open `http://<mac-ip>:<vite-port>/`.

Example: if the Mac IP is `192.168.1.156` and Vite is running on `5500`, open `http://192.168.1.156:5500/`.

## Keyboard Shortcuts

- `P`: pen tool
- `M`: mouse/pan tool
- `E`: eraser tool
- `Ctrl+Z` / `Cmd+Z`: undo
- `Ctrl+Shift+Z` / `Cmd+Shift+Z` or `Ctrl+Y` / `Cmd+Y`: redo
- `Shift+C`: clear strokes
- `Ctrl+B` / `Cmd+B`: toggle developer debug boxes

## Debug Boxes

`Ctrl+B` / `Cmd+B` toggles a developer-only overlay. Blue dashed boxes show the invisible problem catchment region. Green boxes show the active dynamic answer region. Purple boxes show submitted/frozen answer regions.

The overlay is non-interactive and pans with the whiteboard.
