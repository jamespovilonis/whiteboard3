# Whiteboard Engine Notes

The `whiteboard` folder contains plain JavaScript modules for drawing and board geometry. React talks to this layer through `CanvasStrokeEngine`.

## Stroke Shape

Finalized strokes are stored by `StrokeStore` with the whiteboard 2-compatible shape:

- `id`
- normalized `points`
- board-space `rawPoints`
- board-space `outlinePoints`
- `color`
- normalized `bbox`
- board-space `canvasBbox`
- timing fields
- `relationsToPrev`

Problem-flow code uses `canvasBbox` because problem boxes and answer boxes are board-space rectangles.

## Rendering Pipeline

`CanvasStrokeEngine` owns pointer/tool state, undo/redo history, stroke-store mutations, and callbacks to React.

`CanvasRenderer` owns canvas sizing, board fill, viewport transforms, full redraws, live stroke rendering, and eraser preview rendering.

`StrokeSmoother` preserves the whiteboard 2 drawing feel: streamline, Catmull-Rom interpolation, pressure-aware thickness, and polygon fill rendering.

## Erasing And History

The eraser removes whole strokes by checking intersection between the smoothed eraser polygon and each stroke outline polygon. Undo and redo snapshot the stroke store and notify React through `onStrokesChanged` so active answer boxes stay reconciled.

## Viewport Coordinates

The board is larger than the screen. Pointer input converts from screen space to board space before strokes are stored. Canvas rendering applies the inverse viewport transform so strokes, problems, and debug boxes all move together while panning.
