# Frontend Architecture

The frontend separates React-owned UI state from plain JavaScript whiteboard mechanics.

## React Shell

`App.jsx` composes the application and wires together focused hooks:

- `hooks/useViewportController.js`: viewport state, home viewport state, smooth viewport animation, and reset-window visibility.
- `hooks/useToolbarCollapse.js`: active tool selection and the pencil-drawing toolbar collapse rule.
- `hooks/useWhiteboardShortcuts.js`: keyboard shortcuts, including the debug-box toggle.
- `hooks/useProblemFlowController.js`: problem sequence state, model-shell response text, submit/recognition behavior, and stroke reconciliation.

## Components

- `components/WhiteboardStage.jsx` owns the canvas elements and connects React props/callbacks to the plain JS engine.
- `components/ProblemLayer.jsx` renders KaTeX problem text in board space and, when enabled, debug boxes for problem/answer regions.
- `components/Toolbar.jsx` renders drawing controls and delegates all state changes upward.
- `components/ModelShell.jsx` renders the model-response shell, submit action, and recent recognition output.

## Problem Flow

Problem state lives in `state/problemFlow.js`; placeholder problem data lives in `state/problemFixtures.js`.

The active problem has an invisible catchment box. Whenever the stroke store changes, the active answer box is recomputed from the current strokes. Submitted answer boxes are frozen so advancing to the next problem does not rewrite previous layout decisions.

Submitting a problem snapshots the current strokes, freezes the dynamic answer box, starts the recognition pipeline for that submitted problem, and stores the resulting LaTeX/segmentation metadata on the problem session. If the OCR endpoint is offline, the submission still advances and preserves an error state for the recognition attempt.

## Styles

`styles.css` is an import-only aggregator:

- `styles/base.css`
- `styles/whiteboard.css`
- `styles/toolbar.css`
- `styles/model-shell.css`

Keep new component styles near their feature area unless a value is truly global.
