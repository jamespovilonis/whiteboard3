# Whiteboard Engine Notes

The `whiteboard` folder contains framework-free JavaScript modules for drawing, hit-testing, history, rendering, stroke storage, and board geometry. React talks to this layer through `CanvasStrokeEngine`; recognition and problem-flow code consume the finalized stroke records emitted by the engine.

## Modules

- `CanvasStrokeEngine.js`: pointer/tool state, pen and eraser behavior, undo/redo snapshots, stroke-store mutations, viewport updates, and callbacks to React.
- `CanvasRenderer.js`: canvas sizing, board fill, viewport transforms, full redraws, live stroke rendering, finalized stroke rendering, and eraser preview rendering.
- `StrokeStore.js`: finalized stroke persistence, lookup, replacement, deletion, and snapshot serialization.
- `StrokeSmoother.js`: whiteboard-2-style streamline smoothing, Catmull-Rom interpolation, pressure-aware thickness, and polygon outline generation.
- `geometry.js`: rectangle, polygon, point, distance, and intersection helpers shared by erasing and board math.
- `viewport.js`: board/screen coordinate conversion, clamping, home viewport, and zoom/pan helpers.
- `constants.js`: engine defaults shared across drawing and rendering.

## Stroke Shape

Finalized strokes are stored by `StrokeStore` with the whiteboard-2-compatible shape:

- `id`
- normalized `points`
- board-space `rawPoints`
- board-space `outlinePoints`
- `color`
- normalized `bbox`
- board-space `canvasBbox`
- timing fields such as `startTime` and `endTime`
- `relationsToPrev`

Problem-flow and recognition code use `canvasBbox` because problem boxes, answer boxes, debug overlays, and audit crops are board-space rectangles. Audit payloads downsample point arrays, but the in-app store keeps the full finalized stroke geometry.

## Rendering Pipeline

Pointer input arrives in screen space, converts to board space through the active viewport, and is accumulated as live raw points. During drawing, `StrokeSmoother` builds a pressure-aware outline and `CanvasRenderer` renders the live stroke. On pointer up, the finalized stroke is written to `StrokeStore`, history is snapshotted, and React is notified through `onStrokesChanged`.

`CanvasRenderer` applies the inverse viewport transform during redraws so strokes, problems, debug boxes, and eraser preview geometry move together while panning and zooming.

## Erasing And History

The eraser removes whole strokes by checking intersection between the smoothed eraser polygon and each stroke outline polygon. Undo and redo snapshot the stroke store and notify React through `onStrokesChanged` so active answer boxes, recognition scheduling, and submitted/frozen boxes stay reconciled.

## Viewport Coordinates

The board is larger than the screen. Canvas input converts from screen space to board space before strokes are stored. Rendering converts board space back through the current viewport. This keeps recognition, problem geometry, debug overlays, and audit crops in one consistent coordinate system even as the user pans.
