import { useCallback, useRef, useState } from 'react';
import ModelShell from './components/ModelShell.jsx';
import Toolbar from './components/Toolbar.jsx';
import WhiteboardStage from './components/WhiteboardStage.jsx';
import { useProblemFlowController } from './hooks/useProblemFlowController.js';
import { useToolbarCollapse } from './hooks/useToolbarCollapse.js';
import { useViewportController } from './hooks/useViewportController.js';
import { useWhiteboardShortcuts } from './hooks/useWhiteboardShortcuts.js';
import {
  BOARD_SIZE,
  DEFAULT_PEN_COLOR,
  sliderToWidth
} from './whiteboard/constants.js';

export default function App() {
  const [penColor, setPenColor] = useState(DEFAULT_PEN_COLOR);
  const [sliderValue, setSliderValue] = useState(4);
  const [isPanning, setIsPanning] = useState(false);
  const [debugBoxesEnabled, setDebugBoxesEnabled] = useState(false);
  const engineRef = useRef(null);

  const {
    activeTool,
    toolbarForceCollapsed,
    selectTool,
    collapseToolbarForDrawing,
    requestToolbarOpen
  } = useToolbarCollapse();

  const {
    viewport,
    setViewport,
    showReset,
    resetViewport,
    moveHomeViewport
  } = useViewportController();

  const {
    problemFlow,
    modelResponse,
    reconcileStrokes,
    submitAnswer
  } = useProblemFlowController({ moveHomeViewport });

  const penWidth = sliderToWidth(sliderValue);

  const handleEngineReady = useCallback((engine) => {
    engineRef.current = engine;
  }, []);

  const handleSliderChange = useCallback((value) => {
    setSliderValue(Number(value));
  }, []);

  const toggleDebugBoxes = useCallback(() => {
    setDebugBoxesEnabled((isEnabled) => !isEnabled);
  }, []);

  useWhiteboardShortcuts({
    engineRef,
    onSelectTool: selectTool,
    onToggleDebugBoxes: toggleDebugBoxes
  });

  return (
    <main className="app-shell">
      <WhiteboardStage
        boardSize={BOARD_SIZE}
        viewport={viewport}
        activeTool={activeTool}
        penColor={penColor}
        penWidth={penWidth}
        problems={problemFlow.problems}
        debugBoxesEnabled={debugBoxesEnabled}
        isPanning={isPanning}
        onEngineReady={handleEngineReady}
        onViewportChange={setViewport}
        onPanStateChange={setIsPanning}
        onStrokesChanged={reconcileStrokes}
        onPenStrokeStart={collapseToolbarForDrawing}
      />

      <Toolbar
        activeTool={activeTool}
        penColor={penColor}
        sliderValue={sliderValue}
        onToolChange={selectTool}
        onColorChange={setPenColor}
        onSliderChange={handleSliderChange}
        onUndo={() => engineRef.current?.undo()}
        onRedo={() => engineRef.current?.redo()}
        onClear={() => engineRef.current?.clear()}
        forceCollapsed={toolbarForceCollapsed}
        onRequestOpen={requestToolbarOpen}
      />

      <ModelShell
        response={modelResponse}
        onSubmitAnswer={submitAnswer}
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
