import { useCallback, useEffect, useMemo, useRef } from 'react';
import ProblemLayer from './ProblemLayer.jsx';
import { CanvasStrokeEngine } from '../whiteboard/CanvasStrokeEngine.js';
import { StrokeStore } from '../whiteboard/StrokeStore.js';

export default function WhiteboardStage({
  boardSize,
  viewport,
  activeTool,
  penColor,
  penWidth,
  problems,
  debugBoxesEnabled,
  isPanning,
  onEngineReady,
  onViewportChange,
  onPanStateChange,
  onStrokeFinalized,
  onStrokesChanged,
  onPenStrokeStart,
  className = '',
  canvasTestId = 'whiteboard-canvas'
}) {
  const bgCanvasRef = useRef(null);
  const fgCanvasRef = useRef(null);
  const engineRef = useRef(null);
  const strokeStore = useMemo(() => new StrokeStore(), []);

  useEffect(() => {
    const bgCanvas = bgCanvasRef.current;
    const fgCanvas = fgCanvasRef.current;
    if (!bgCanvas || !fgCanvas) return undefined;

    const engine = new CanvasStrokeEngine({
      bgCanvas,
      fgCanvas,
      strokeStore,
      boardSize,
      viewport,
      tool: activeTool,
      penColor,
      penWidth,
      onViewportChange,
      onPanStateChange,
      onStrokeFinalized,
      onStrokesChanged,
      onPenStrokeStart
    });

    engineRef.current = engine;
    engine.attach();
    onEngineReady(engine);

    const handleResize = () => engine.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      engine.detach();
      engineRef.current = null;
      onEngineReady(null);
    };
  }, []);

  useEffect(() => {
    engineRef.current?.setViewport(viewport);
  }, [viewport]);

  useEffect(() => {
    engineRef.current?.setCallbacks({
      onViewportChange,
      onPanStateChange,
      onStrokeFinalized,
      onStrokesChanged,
      onPenStrokeStart
    });
  }, [
    onPanStateChange,
    onPenStrokeStart,
    onStrokeFinalized,
    onStrokesChanged,
    onViewportChange
  ]);

  useEffect(() => {
    engineRef.current?.setTool(activeTool);
  }, [activeTool]);

  useEffect(() => {
    engineRef.current?.setPenColor(penColor);
  }, [penColor]);

  useEffect(() => {
    engineRef.current?.setPenWidth(penWidth);
  }, [penWidth]);

  const handleWheel = useCallback((event) => {
    event.preventDefault();
    const deltaModeScale = event.deltaMode === 1
      ? 16
      : (event.deltaMode === 2 ? window.innerHeight : 1);
    const dx = (Number(event.deltaX) || 0) * deltaModeScale / viewport.scale;
    const dy = (Number(event.deltaY) || 0) * deltaModeScale / viewport.scale;
    if (dx === 0 && dy === 0) return;

    onViewportChange?.({
      ...viewport,
      x: viewport.x + dx,
      y: viewport.y + dy
    });
  }, [onViewportChange, viewport]);

  return (
    <div
      className={`whiteboard-stage tool-${activeTool} ${isPanning ? 'is-panning' : ''} ${className}`}
      onWheel={handleWheel}
    >
      <canvas ref={bgCanvasRef} className="whiteboard-canvas whiteboard-bg" aria-hidden="true" />
      <canvas
        ref={fgCanvasRef}
        className="whiteboard-canvas whiteboard-fg"
        aria-label="Whiteboard drawing surface"
        data-testid={canvasTestId}
      />
      <ProblemLayer
        problems={problems}
        viewport={viewport}
        debugBoxesEnabled={debugBoxesEnabled}
      />
    </div>
  );
}
