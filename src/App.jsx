import { useCallback, useRef, useState } from 'react';
import ModelShell from './components/ModelShell.jsx';
import Toolbar from './components/Toolbar.jsx';
import WhiteboardStage from './components/WhiteboardStage.jsx';
import {
  E2E_TEST_ENABLED,
  recordE2EEvent,
  useE2ETestBridge
} from './hooks/useE2ETestBridge.js';
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
  const e2eEventsRef = useRef([]);

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

  const handleRecognitionEvent = useCallback((type, detail) => {
    recordE2EEvent(e2eEventsRef, type, detail);
  }, []);

  const {
    problemFlow,
    modelResponse,
    recognitionResults,
    reconcileStrokes,
    submitAnswer
  } = useProblemFlowController({
    moveHomeViewport,
    engineRef,
    onRecognitionEvent: handleRecognitionEvent
  });

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

  const handlePenStrokeStart = useCallback(() => {
    recordE2EEvent(e2eEventsRef, 'pen-stroke-start');
    collapseToolbarForDrawing();
  }, [collapseToolbarForDrawing]);

  const handleStrokeFinalized = useCallback((stroke) => {
    recordE2EEvent(e2eEventsRef, 'stroke-finalized', {
      strokeId: stroke?.id,
      canvasBbox: stroke?.canvasBbox || null,
      startTime: stroke?.startTime,
      endTime: stroke?.endTime
    });
  }, []);

  const handleStrokesChanged = useCallback((strokes, reason) => {
    recordE2EEvent(e2eEventsRef, 'strokes-changed', {
      reason,
      strokeCount: strokes?.length || 0
    });
    reconcileStrokes(strokes);
  }, [reconcileStrokes]);

  const handleSubmitAnswer = useCallback(() => {
    recordE2EEvent(e2eEventsRef, 'submit-answer');
    submitAnswer();
  }, [submitAnswer]);

  useWhiteboardShortcuts({
    engineRef,
    onSelectTool: selectTool,
    onToggleDebugBoxes: toggleDebugBoxes
  });

  useE2ETestBridge({
    engineRef,
    viewport,
    problemFlow,
    recognitionResults,
    eventsRef: e2eEventsRef
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
        onStrokeFinalized={handleStrokeFinalized}
        onStrokesChanged={handleStrokesChanged}
        onPenStrokeStart={handlePenStrokeStart}
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
        recognitionResults={recognitionResults}
        debugMode={debugBoxesEnabled || E2E_TEST_ENABLED}
        onSubmitAnswer={handleSubmitAnswer}
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
