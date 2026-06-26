import { DEFAULT_PEN_COLOR, DEFAULT_PEN_WIDTH } from './constants.js';
import { computePolygonBbox, bboxOverlap, polygonsIntersect } from './geometry.js';
import { CanvasRenderer } from './CanvasRenderer.js';
import { StrokeSmoother } from './StrokeSmoother.js';
import { screenToBoard } from './viewport.js';

const UNDO_MAX = 100;
const ERASER_SIZE = 20;
const MIN_ERASE_DISTANCE = 8;

export class CanvasStrokeEngine {
  constructor({
    bgCanvas,
    fgCanvas,
    strokeStore,
    boardSize,
    viewport,
    tool = 'pen',
    penColor = DEFAULT_PEN_COLOR,
    penWidth = DEFAULT_PEN_WIDTH,
    onViewportChange,
    onPanStateChange,
    onStrokeFinalized,
    onStrokesChanged,
    onPenStrokeStart
  }) {
    this.bgCanvas = bgCanvas;
    this.fgCanvas = fgCanvas;
    this.strokeStore = strokeStore;
    this.boardSize = boardSize;
    this.viewport = { ...viewport };
    this.renderer = new CanvasRenderer({ bgCanvas, fgCanvas, viewport });
    this.tool = tool;
    this.penColor = penColor;
    this.penWidth = penWidth;
    this.callbacks = {};
    this.setCallbacks({
      onViewportChange,
      onPanStateChange,
      onStrokeFinalized,
      onStrokesChanged,
      onPenStrokeStart
    });

    this.strokeSmoother = new StrokeSmoother({ size: penWidth });
    this.eraserSmoother = new StrokeSmoother({
      size: ERASER_SIZE,
      thinning: 0,
      smoothing: 0.5,
      streamline: 0.7,
      startTaper: 0,
      endTaper: 0,
      cap: true
    });

    this.undoStack = [];
    this.redoStack = [];
    this.isDrawing = false;
    this.isErasing = false;
    this.isPanning = false;
    this.rawPoints = [];
    this.eraserPoints = [];
    this.lastCheckedPosition = null;
    this.panStart = null;
    this.rafPending = false;
    this.redrawRaf = null;
    this.pendingRawPoints = null;
    this.pendingColor = null;

    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
    this.handlePointerLeave = this.handlePointerLeave.bind(this);
  }

  attach() {
    this.fgCanvas.addEventListener('pointerdown', this.handlePointerDown);
    this.fgCanvas.addEventListener('pointermove', this.handlePointerMove);
    this.fgCanvas.addEventListener('pointerup', this.handlePointerUp);
    this.fgCanvas.addEventListener('pointercancel', this.handlePointerUp);
    this.fgCanvas.addEventListener('pointerleave', this.handlePointerLeave);
    this.resize();
  }

  detach() {
    this.fgCanvas.removeEventListener('pointerdown', this.handlePointerDown);
    this.fgCanvas.removeEventListener('pointermove', this.handlePointerMove);
    this.fgCanvas.removeEventListener('pointerup', this.handlePointerUp);
    this.fgCanvas.removeEventListener('pointercancel', this.handlePointerUp);
    this.fgCanvas.removeEventListener('pointerleave', this.handlePointerLeave);
    if (this.redrawRaf) {
      cancelAnimationFrame(this.redrawRaf);
      this.redrawRaf = null;
    }
  }

  resize() {
    this.renderer.resize();
    this.redraw();
  }

  redraw() {
    this.renderer.redraw(this.strokeStore.getStrokes(), this.strokeSmoother);
  }

  requestRedraw() {
    if (this.redrawRaf) return;

    this.redrawRaf = requestAnimationFrame(() => {
      this.redrawRaf = null;
      this.redraw();
    });
  }

  setCallbacks(callbacks = {}) {
    this.callbacks = {
      ...this.callbacks,
      ...callbacks
    };
  }

  undo() {
    if (this.undoStack.length === 0) return;

    this.redoStack.push(this.strokeStore.snapshot());
    if (this.redoStack.length > UNDO_MAX) this.redoStack.shift();

    this.strokeStore.restore(this.undoStack.pop());
    this.redraw();
    this.notifyStrokesChanged('undo');
  }

  redo() {
    if (this.redoStack.length === 0) return;

    this.undoStack.push(this.strokeStore.snapshot());
    if (this.undoStack.length > UNDO_MAX) this.undoStack.shift();

    this.strokeStore.restore(this.redoStack.pop());
    this.redraw();
    this.notifyStrokesChanged('redo');
  }

  clear() {
    if (this.strokeStore.getStrokes().length === 0) return;
    this.pushUndoState();
    this.strokeStore.clear();
    this.redraw();
    this.notifyStrokesChanged('clear');
  }

  setTool(tool) {
    this.tool = tool;
    this.finishActiveGesture();
  }

  setPenColor(color) {
    this.penColor = color;
  }

  setPenWidth(width) {
    this.penWidth = width;
    this.strokeSmoother.opts.size = width;
  }

  setViewport(viewport) {
    this.viewport = { ...viewport };
    this.renderer.setViewport(this.viewport);
    this.requestRedraw();
  }

  handlePointerDown(event) {
    event.preventDefault();
    this.fgCanvas.setPointerCapture?.(event.pointerId);

    if (this.tool === 'mouse') {
      const screenPoint = this.getScreenPoint(event);
      this.isPanning = true;
      this.panStart = {
        screenPoint,
        viewport: { ...this.viewport }
      };
      this.callbacks.onPanStateChange?.(true);
      return;
    }

    if (this.tool === 'pen') {
      const point = this.getBoardPoint(event);
      this.isDrawing = true;
      this.rawPoints = [{ ...point, pressure: event.pressure || 0.5 }];
      this.callbacks.onPenStrokeStart?.();
      this.strokeStore.startStroke(
        point.x,
        point.y,
        event.pressure || 0.5,
        this.boardSize.width,
        this.boardSize.height
      );
      return;
    }

    if (this.tool === 'eraser') {
      const point = this.getBoardPoint(event);
      this.isErasing = true;
      this.eraserPoints = [{ ...point, pressure: event.pressure || 0.5, time: Date.now() }];
      this.lastCheckedPosition = null;
      this.renderEraserPreview(point);
    }
  }

  handlePointerMove(event) {
    if (this.isPanning && this.tool === 'mouse') {
      const screenPoint = this.getScreenPoint(event);
      const dx = screenPoint.x - this.panStart.screenPoint.x;
      const dy = screenPoint.y - this.panStart.screenPoint.y;
      const nextViewport = {
        ...this.viewport,
        x: this.panStart.viewport.x - dx / this.viewport.scale,
        y: this.panStart.viewport.y - dy / this.viewport.scale
      };
      this.viewport = nextViewport;
      this.renderer.setViewport(nextViewport);
      this.requestRedraw();
      this.callbacks.onViewportChange?.(nextViewport);
      return;
    }

    if (this.tool === 'pen' && this.isDrawing) {
      const point = this.getBoardPoint(event);
      const pointWithPressure = { ...point, pressure: event.pressure || 0.5 };
      this.rawPoints.push(pointWithPressure);
      this.strokeStore.addPoint(
        point.x,
        point.y,
        event.pressure || 0.5,
        this.boardSize.width,
        this.boardSize.height
      );
      this.requestLiveStroke(this.rawPoints, this.penColor);
      return;
    }

    if (this.tool === 'eraser') {
      const point = this.getBoardPoint(event);

      if (!this.isErasing) {
        this.renderEraserPreview(point);
        return;
      }

      this.eraserPoints.push({ ...point, pressure: event.pressure || 0.5, time: Date.now() });

      if (this.shouldCheckErase(point)) {
        this.performErase();
        this.lastCheckedPosition = point;
      }

      this.renderEraserPreview(point);
    }
  }

  handlePointerUp(event) {
    if (this.fgCanvas.hasPointerCapture?.(event.pointerId)) {
      this.fgCanvas.releasePointerCapture(event.pointerId);
    }

    if (this.isPanning) {
      this.isPanning = false;
      this.panStart = null;
      this.callbacks.onPanStateChange?.(false);
      return;
    }

    if (this.isDrawing) {
      this.finalizeCurrentStroke();
    }

    if (this.isErasing) {
      this.performErase();
      this.stopErasing();
    }
  }

  handlePointerLeave() {
    if (this.isPanning) return;

    if (this.isDrawing) {
      this.finalizeCurrentStroke();
    }

    if (this.isErasing) {
      this.performErase();
      this.stopErasing();
    } else if (this.tool === 'eraser') {
      this.clearForeground();
    }
  }

  finalizeCurrentStroke() {
    if (!this.isDrawing) return;

    const color = this.penColor || DEFAULT_PEN_COLOR;
    this.pushUndoState();

    let finalizedStroke = null;

    if (this.rawPoints.length === 1) {
      const point = this.rawPoints[0];
      const outline = createDotOutline(point.x, point.y, this.strokeSmoother.opts.size || DEFAULT_PEN_WIDTH);
      finalizedStroke = this.strokeStore.endStroke(outline, color);
    } else if (this.rawPoints.length >= 2) {
      const outline = this.strokeSmoother.smooth(this.rawPoints);
      finalizedStroke = this.strokeStore.endStroke(outline, color);
    }

    this.isDrawing = false;
    this.rawPoints = [];
    this.clearForeground();
    this.redraw();
    if (finalizedStroke) {
      this.callbacks.onStrokeFinalized?.(finalizedStroke);
      this.notifyStrokesChanged('draw');
    }
  }

  performErase() {
    if (this.eraserPoints.length < 2) return;

    const eraserOutline = this.eraserSmoother.smooth(
      this.eraserPoints.map((point) => ({ x: point.x, y: point.y, pressure: 1 }))
    );
    if (eraserOutline.length < 3) return;

    const eraserBbox = computePolygonBbox(eraserOutline);
    const toRemove = [];

    this.strokeStore.getStrokes().forEach((stroke, index) => {
      if (!stroke.outlinePoints || stroke.outlinePoints.length < 3) return;
      if (stroke.canvasBbox && !bboxOverlap(eraserBbox, stroke.canvasBbox)) return;
      if (polygonsIntersect(eraserOutline, stroke.outlinePoints)) toRemove.push(index);
    });

    if (toRemove.length > 0) {
      this.pushUndoState();
      this.strokeStore.removeStrokes(toRemove);
      this.redraw();
      this.notifyStrokesChanged('erase');
    }
  }

  stopErasing() {
    this.isErasing = false;
    this.eraserPoints = [];
    this.lastCheckedPosition = null;
    this.clearForeground();
  }

  finishActiveGesture() {
    if (this.isDrawing) this.finalizeCurrentStroke();
    if (this.isErasing) this.stopErasing();
    if (this.isPanning) {
      this.isPanning = false;
      this.panStart = null;
      this.callbacks.onPanStateChange?.(false);
    }
    this.clearForeground();
  }

  requestLiveStroke(rawPoints, color) {
    this.pendingRawPoints = rawPoints;
    this.pendingColor = color;

    if (this.rafPending) return;

    this.rafPending = true;
    requestAnimationFrame(() => {
      this.rafPending = false;
      if (this.pendingRawPoints) {
        this.renderLiveStroke(this.pendingRawPoints, this.pendingColor);
      }
    });
  }

  renderLiveStroke(rawPoints, color) {
    if (!rawPoints || rawPoints.length < 2) return;

    const outline = this.strokeSmoother.smooth(rawPoints);
    this.renderer.renderLiveStroke(outline, color, this.strokeSmoother);
  }

  renderEraserPreview(point) {
    this.renderer.renderEraserPreview(point, ERASER_SIZE);
  }

  shouldCheckErase(point) {
    if (this.lastCheckedPosition === null) return true;

    const dx = point.x - this.lastCheckedPosition.x;
    const dy = point.y - this.lastCheckedPosition.y;
    return Math.sqrt(dx * dx + dy * dy) >= MIN_ERASE_DISTANCE;
  }

  pushUndoState() {
    this.undoStack.push(this.strokeStore.snapshot());
    if (this.undoStack.length > UNDO_MAX) this.undoStack.shift();
    this.redoStack = [];
  }

  notifyStrokesChanged(reason) {
    this.callbacks.onStrokesChanged?.(this.strokeStore.getStrokes(), reason);
  }

  getScreenPoint(event) {
    const rect = this.fgCanvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  getBoardPoint(event) {
    return screenToBoard(this.getScreenPoint(event), this.viewport);
  }

  clearForeground() {
    this.renderer.clearForeground();
  }
}

function createDotOutline(x, y, size) {
  const radius = Math.max(1, size / 2);
  const segments = 16;
  const outline = [];

  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    outline.push({
      x: x + Math.cos(angle) * radius,
      y: y + Math.sin(angle) * radius
    });
  }

  return outline;
}
