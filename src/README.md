# Frontend And Backend Architecture

The app separates React-owned UI state from plain JavaScript whiteboard mechanics. The same `src/` tree also contains the local FastAPI gateway used by browser recognition, semantic scoring, grading, audit logging, and math feedback.

## Folder Map

- `App.jsx` and `main.jsx`: React app entry points.
- `components/`: visual UI components for the board, toolbar, problem layer, model shell, and handwritten-problem dialog.
- `hooks/`: viewport, toolbar, shortcut, E2E bridge, and problem-flow controllers.
- `state/`: problem-flow state machine and equation/problem catalog loading.
- `whiteboard/`: framework-free canvas engine, renderer, stroke store, smoothing, geometry, and viewport helpers.
- `recognition/`: browser OCR pipeline, line segmentation, rasterization, model clients, audit client, and latency telemetry.
- `grading/`: browser grading client plus Python graders used by the backend.
- `feedback/`: browser client for post-submit tutoring feedback.
- `server/`: FastAPI app, route modules, configuration, and services.
- `styles/`: feature CSS files imported by `styles.css`.

## React Shell

`App.jsx` composes the application and wires together focused hooks:

- `hooks/useViewportController.js`: viewport state, home viewport state, smooth viewport animation, and reset-window visibility.
- `hooks/useToolbarCollapse.js`: active tool selection and the pencil-drawing toolbar collapse rule.
- `hooks/useWhiteboardShortcuts.js`: keyboard shortcuts, including the debug-box toggle.
- `hooks/useProblemFlowController.js`: problem sequence state, model-shell response text, submit behavior, live recognition, grading, feedback, and audit handoff.
- `hooks/useE2ETestBridge.js`: deterministic hooks used by Playwright tests.

## Components

- `components/WhiteboardStage.jsx` owns the canvas elements and connects React props/callbacks to `CanvasStrokeEngine`.
- `components/ProblemLayer.jsx` renders KaTeX problem text in board space and, when enabled, debug boxes for problem/answer regions.
- `components/Toolbar.jsx` renders drawing controls and delegates all state changes upward.
- `components/ModelShell.jsx` renders recognition/grading status, feedback text, and submit actions.
- `components/HandwrittenProblemDialog.jsx` supports custom handwritten problem entry.

## Problem Flow

Problem state lives in `state/problemFlow.js`. Temporary rendered problems are loaded from `state/equationProblemSource.js` when the dev/test problem-source flag is enabled.

The active problem has an invisible catchment box. Whenever the stroke store changes, the active answer box is recomputed from current strokes. Submitted answer boxes are frozen so advancing to the next problem does not rewrite previous layout decisions.

Submitting a problem snapshots the current strokes, freezes the dynamic answer box, starts recognition for that submitted attempt, grades the settled recognition result, and stores LaTeX/segmentation/grading metadata on the problem session. If OCR is offline, the submission still advances and preserves an error state for the recognition attempt.

After grading settles, `useProblemFlowController.js` starts math feedback for non-correct work by calling `feedback/requestMathFeedback()`. Correct work gets deterministic feedback immediately. Feedback is keyed by `attemptId` and `inputSignature`, hidden when stale, and attached to any queued VLM audit for the same attempt.

## Recognition, Grading, Audit, And Feedback

The browser uses `recognition/config.js` to choose the API base URL. `VITE_API_URL` wins, `VITE_OCR_API_URL` is a compatibility fallback, and otherwise the browser targets the current hostname on port `8010`.

`recognition/studentWritingPipeline.js` performs the browser-side OCR pipeline: answer-box rasterization, optional detector bands, geometric line candidates, crop OCR, semantic rescoring, exact-cover selection, selected-line retries, long-line chunk fallback, and final grading handoff.

`recognition/auditClient.js` compacts settled recognition attempts for VLM auditing. It includes compact strokes, problem metadata, fast recognition results, trigger reasons, and same-attempt feedback when available. Audit enqueueing is automatic for suspicious failures and deterministic samples of normal correct work.

`feedback/feedbackClient.js` posts a compact feedback request to `/feedback/math-work`, enforces a client timeout, trims returned text to at most two sentences, and falls back to deterministic text when the server is unavailable.

## FastAPI Gateway

`server/app.py` creates the gateway and includes these route groups:

- `server/routes/recognition.py`: proxies `/recognize`, `/segment-lines`, `/segment-lines/*`, and `/health` to the upstream CoMER/DBNet server.
- `server/routes/grading.py`: serves `/score-latex-candidates`, `/grade-equation-work`, and `/grade-math-work` with request-level timeouts.
- `server/routes/feedback.py`: serves `/feedback/math-work` through `MathFeedbackService`.
- `server/routes/audit.py`: serves `/audit-recognition`, `/audit-recognition-feedback`, `/audit-recognition-note`, and audit status polling.
- `server/routes/health.py`: serves `/gateway/health` with current gateway, audit, feedback, and upstream configuration.

`server/config.py` reads `WHITEBOARD_API_*`, `UPSTREAM_OCR_*`, `SEMANTIC_TIMEOUT`, `WHITEBOARD_AUDIT_*`, `VLM_AUDIT_*`, and `WHITEBOARD_FEEDBACK_*` environment variables.

## Math Feedback Contract

The frontend request to `/feedback/math-work` is built by `buildMathFeedbackRequest()`:

- `problemId`, `problemLatex`, and `problemMetadata`
- `inputSignature` and `attemptId`
- full `grading` result from the settled recognition attempt
- compact `fastResult` from `recognition/auditClient.js`

`MathFeedbackService` builds a `promptContext`, chooses a concrete `targetLine`, and either returns deterministic correct feedback, LLM text from local Ollama, or deterministic fallback feedback if the LLM errors, outputs JSON/list text, leaks prompt wording, or omits the exact target line.

Returned fields include `attemptId`, `inputSignature`, `status`, `source`, `text`, `model`, `promptVersion`, `createdAt`, `targetLine`, `targetLineSource`, `targetLineReason`, `promptContext`, and optional `skippedReason`, `error`, or `rejectionReason`.

The internal request sent from `MathFeedbackService` to Ollama is a chat payload, not the browser's recognition JSON:

```json
{
  "model": "qwen3:1.7b",
  "messages": [
    {
      "role": "system",
      "content": "You provide concise math feedback..."
    },
    {
      "role": "user",
      "content": "Write one short student-facing feedback message.\nStatus: incorrect\nStudent line to respond to: x = 5\nThe correct target line is exactly: x = 4\nReason to mention: Divide both sides by the coefficient of the variable.\nRules: include the exact target line, do not output JSON, do not mention these rules, and do not say 'give one valid correction or next step'."
    }
  ],
  "stream": false,
  "think": false,
  "options": {
    "temperature": 0,
    "num_predict": 120
  }
}
```

## Styles

`styles.css` is an import-only aggregator:

- `styles/base.css`
- `styles/whiteboard.css`
- `styles/toolbar.css`
- `styles/model-shell.css`

Keep new component styles near their feature area unless a value is truly global.

## Start All Local Servers

From the repo root:

```sh
npm run dev:stop
npm run dev:all
```
