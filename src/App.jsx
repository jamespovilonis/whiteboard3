import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Toolbar from './components/Toolbar.jsx';
import WhiteboardStage from './components/WhiteboardStage.jsx';
import { createInitialProblemState } from './state/problemState.js';
import {
  BOARD_SIZE,
  DEFAULT_PEN_COLOR,
  DEFAULT_PEN_WIDTH,
  DEFAULT_TOOL,
  INITIAL_VIEWPORT,
  RESET_DRIFT_THRESHOLD,
  sliderToWidth
} from './whiteboard/constants.js';
import { isAwayFromViewport } from './whiteboard/viewport.js';

export default function App() {
  const [activeTool, setActiveTool] = useState(DEFAULT_TOOL);
  const [penColor, setPenColor] = useState(DEFAULT_PEN_COLOR);
  const [sliderValue, setSliderValue] = useState(4);
  const [viewport, setViewport] = useState(INITIAL_VIEWPORT);
  const [isPanning, setIsPanning] = useState(false);
  const engineRef = useRef(null);
  const resetAnimationRef = useRef(null);
  const problem = useMemo(
    () => createInitialProblemState(window.innerWidth || 1024),
    []
  );

  const penWidth = sliderToWidth(sliderValue);
  const showReset = isAwayFromViewport(viewport, INITIAL_VIEWPORT, RESET_DRIFT_THRESHOLD);

  const handleEngineReady = useCallback((engine) => {
    engineRef.current = engine;
  }, []);

  const handleSliderChange = useCallback((value) => {
    setSliderValue(Number(value));
  }, []);

  const resetViewport = useCallback(() => {
    if (resetAnimationRef.current) {
      cancelAnimationFrame(resetAnimationRef.current);
    }

    const start = { ...viewport };
    const duration = 260;
    const startedAt = performance.now();

    const tick = (now) => {
      const t = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setViewport({
        x: start.x + (INITIAL_VIEWPORT.x - start.x) * eased,
        y: start.y + (INITIAL_VIEWPORT.y - start.y) * eased,
        scale: start.scale + (INITIAL_VIEWPORT.scale - start.scale) * eased
      });

      if (t < 1) {
        resetAnimationRef.current = requestAnimationFrame(tick);
      }
    };

    resetAnimationRef.current = requestAnimationFrame(tick);
  }, [viewport]);

  useEffect(() => {
    const ignoredTag = () => {
      const tag = document.activeElement?.tagName || '';
      return tag === 'INPUT' || tag === 'TEXTAREA';
    };

    const handleKeyDown = (event) => {
      if (ignoredTag()) return;

      if ((event.key === 'p' || event.key === 'P') && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setActiveTool('pen');
      }

      if ((event.key === 'm' || event.key === 'M') && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setActiveTool('mouse');
      }

      if ((event.key === 'e' || event.key === 'E') && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        setActiveTool('eraser');
      }

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !event.shiftKey) {
        event.preventDefault();
        engineRef.current?.undo();
      }

      if (
        ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && event.shiftKey) ||
        ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y')
      ) {
        event.preventDefault();
        engineRef.current?.redo();
      }

      if ((event.key === 'c' || event.key === 'C') && event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        engineRef.current?.clear();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (resetAnimationRef.current) {
        cancelAnimationFrame(resetAnimationRef.current);
      }
    };
  }, []);

  return (
    <main className="app-shell">
      <WhiteboardStage
        boardSize={BOARD_SIZE}
        viewport={viewport}
        activeTool={activeTool}
        penColor={penColor}
        penWidth={penWidth}
        problem={problem}
        isPanning={isPanning}
        onEngineReady={handleEngineReady}
        onViewportChange={setViewport}
        onPanStateChange={setIsPanning}
      />

      <Toolbar
        activeTool={activeTool}
        penColor={penColor}
        sliderValue={sliderValue}
        penWidth={penWidth}
        onToolChange={setActiveTool}
        onColorChange={setPenColor}
        onSliderChange={handleSliderChange}
        onUndo={() => engineRef.current?.undo()}
        onRedo={() => engineRef.current?.redo()}
        onClear={() => engineRef.current?.clear()}
      />

      <button
        className={`reset-window-btn ${showReset ? 'visible' : ''}`}
        type="button"
        onClick={resetViewport}
        aria-hidden={!showReset}
        tabIndex={showReset ? 0 : -1}
      >
        Reset window
      </button>
    </main>
  );
}
