import katex from 'katex';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useToolbarCollapse } from '../hooks/useToolbarCollapse.js';
import { getRecognitionApiUrl } from '../recognition/config.js';
import { recognizeStudentWriting } from '../recognition/studentWritingPipeline.js';
import {
  DEFAULT_PEN_COLOR,
  sliderToWidth
} from '../whiteboard/constants.js';
import Toolbar from './Toolbar.jsx';
import WhiteboardStage from './WhiteboardStage.jsx';

const HANDWRITING_BOARD_SIZE = Object.freeze({
  width: 760,
  height: 320
});

const HANDWRITING_VIEWPORT = Object.freeze({
  x: 0,
  y: 0,
  scale: 1
});

const PROBLEM_MODES = {
  solve: {
    problemType: 'equation-solving',
    title: 'Write an equation',
    copy: 'Draw the problem you want to solve.',
    actionLabel: 'Solve',
    previewLabel: 'Recognized equation preview'
  },
  evaluate: {
    problemType: 'evaluate-expression',
    title: 'Write an expression',
    copy: 'Draw the numeric expression you want to evaluate.',
    actionLabel: 'Evaluate',
    previewLabel: 'Recognized expression preview'
  }
};

export default function HandwrittenProblemDialog({ onSubmit }) {
  const [mode, setMode] = useState('solve');
  const [penColor, setPenColor] = useState(DEFAULT_PEN_COLOR);
  const [sliderValue, setSliderValue] = useState(4);
  const [isPanning, setIsPanning] = useState(false);
  const [strokes, setStrokes] = useState([]);
  const [status, setStatus] = useState('drawing');
  const [recognizedLatex, setRecognizedLatex] = useState('');
  const [error, setError] = useState('');
  const engineRef = useRef(null);
  const abortControllerRef = useRef(null);
  const {
    activeTool,
    toolbarForceCollapsed,
    selectTool,
    collapseToolbarForDrawing,
    requestToolbarOpen
  } = useToolbarCollapse('pen');
  const modeConfig = PROBLEM_MODES[mode] || PROBLEM_MODES.solve;
  const penWidth = sliderToWidth(sliderValue);
  const trimmedLatex = recognizedLatex.trim();
  const isRecognizing = status === 'recognizing';

  useEffect(() => () => {
    abortControllerRef.current?.abort();
  }, []);

  const previewHtml = useMemo(() => {
    if (!trimmedLatex) return '';
    try {
      return katex.renderToString(trimmedLatex, {
        throwOnError: false,
        displayMode: true
      });
    } catch {
      return trimmedLatex;
    }
  }, [trimmedLatex]);

  const handleEngineReady = useCallback((engine) => {
    engineRef.current = engine;
  }, []);

  const handleStrokesChanged = useCallback((nextStrokes) => {
    setStrokes([...(nextStrokes || [])]);
    if (status !== 'drawing') {
      setStatus('drawing');
      setRecognizedLatex('');
      setError('');
    }
  }, [status]);

  const clearDrawing = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    engineRef.current?.clear();
    setStrokes([]);
    setRecognizedLatex('');
    setError('');
    setStatus('drawing');
  }, []);

  const submitHandwriting = useCallback(async () => {
    const currentStrokes = engineRef.current?.getStrokes?.() || strokes;
    if (!currentStrokes.length || isRecognizing) return;

    abortControllerRef.current?.abort();
    const controller = typeof AbortController !== 'undefined'
      ? new AbortController()
      : null;
    abortControllerRef.current = controller;
    setStatus('recognizing');
    setError('');
    setRecognizedLatex('');

    try {
      const result = await recognizeStudentWriting({
        strokes: currentStrokes,
        answerBox: {
          xMin: 0,
          yMin: 0,
          xMax: HANDWRITING_BOARD_SIZE.width,
          yMax: HANDWRITING_BOARD_SIZE.height
        },
        problemMetadata: {
          problemType: modeConfig.problemType,
          source: 'user-handwriting'
        },
        apiUrl: getRecognitionApiUrl(),
        model: 'comer',
        semanticScoring: false,
        gradeWork: null,
        signal: controller?.signal
      });
      const latex = String(result?.latex || '').trim();
      if (!latex) {
        setStatus('error');
        setError('Unable to read the problem. Adjust the drawing and try again.');
        return;
      }
      setRecognizedLatex(latex);
      setStatus('preview');
    } catch (recognitionError) {
      if (controller?.signal?.aborted) return;
      setStatus('error');
      setError(
        recognitionError instanceof Error
          ? recognitionError.message
          : 'Unable to read the problem. Adjust the drawing and try again.'
      );
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, [isRecognizing, modeConfig.problemType, strokes]);

  const confirmProblem = useCallback(() => {
    if (!trimmedLatex) return;
    onSubmit?.({
      latex: trimmedLatex,
      problemType: modeConfig.problemType,
      source: 'user-handwriting'
    });
    clearDrawing();
  }, [clearDrawing, modeConfig.problemType, onSubmit, trimmedLatex]);

  const adjustDrawing = useCallback(() => {
    setStatus('drawing');
    setError('');
  }, []);

  return (
    <div className="latex-dialog-backdrop" role="presentation">
      <section
        className="latex-dialog handwritten-problem-dialog"
        aria-labelledby="handwritten-problem-title"
      >
        <div className="latex-dialog-copy">
          <h1 id="handwritten-problem-title">{modeConfig.title}</h1>
          <p>{modeConfig.copy}</p>
        </div>

        <div className="latex-dialog-mode" role="group" aria-label="Problem type">
          <button
            type="button"
            className={mode === 'solve' ? 'is-selected' : ''}
            aria-pressed={mode === 'solve'}
            data-testid="handwritten-problem-type-solve"
            onClick={() => setMode('solve')}
          >
            Solve
          </button>
          <button
            type="button"
            className={mode === 'evaluate' ? 'is-selected' : ''}
            aria-pressed={mode === 'evaluate'}
            data-testid="handwritten-problem-type-evaluate"
            onClick={() => setMode('evaluate')}
          >
            Evaluate
          </button>
        </div>

        <div className="handwritten-problem-pad" data-testid="handwritten-problem-pad">
          <WhiteboardStage
            boardSize={HANDWRITING_BOARD_SIZE}
            viewport={HANDWRITING_VIEWPORT}
            activeTool={activeTool}
            penColor={penColor}
            penWidth={penWidth}
            problems={[]}
            debugBoxesEnabled={false}
            isPanning={isPanning}
            onEngineReady={handleEngineReady}
            onViewportChange={() => {}}
            onPanStateChange={setIsPanning}
            onStrokeFinalized={() => {}}
            onStrokesChanged={handleStrokesChanged}
            onPenStrokeStart={collapseToolbarForDrawing}
            className="handwritten-problem-stage"
            canvasTestId="handwritten-problem-canvas"
          />
          <Toolbar
            activeTool={activeTool}
            penColor={penColor}
            sliderValue={sliderValue}
            onToolChange={selectTool}
            onColorChange={setPenColor}
            onSliderChange={(value) => setSliderValue(Number(value))}
            onUndo={() => engineRef.current?.undo()}
            onRedo={() => engineRef.current?.redo()}
            onClear={clearDrawing}
            forceCollapsed={toolbarForceCollapsed}
            onRequestOpen={requestToolbarOpen}
          />
        </div>

        {status === 'preview' && (
          <div
            className="handwritten-problem-preview"
            aria-label={modeConfig.previewLabel}
            data-testid="handwritten-problem-preview"
          >
            <span data-testid="handwritten-problem-action">{modeConfig.actionLabel}</span>
            <div
              className="handwritten-problem-preview-math"
              data-testid="handwritten-problem-latex"
              data-latex={trimmedLatex}
              dangerouslySetInnerHTML={{ __html: previewHtml }}
            />
          </div>
        )}

        {error && (
          <div
            className="handwritten-problem-error"
            role="status"
            data-testid="handwritten-problem-error"
          >
            {error}
          </div>
        )}

        <div className="latex-dialog-actions handwritten-problem-actions">
          {status === 'preview' ? (
            <>
              <button
                type="button"
                className="secondary"
                data-testid="handwritten-problem-adjust"
                onClick={adjustDrawing}
              >
                Adjust drawing
              </button>
              <button
                type="button"
                className="secondary"
                data-testid="handwritten-problem-clear"
                onClick={clearDrawing}
              >
                Clear
              </button>
              <button
                type="button"
                data-testid="handwritten-problem-confirm"
                disabled={!trimmedLatex}
                onClick={confirmProblem}
              >
                Confirm problem
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="secondary"
                data-testid="handwritten-problem-clear"
                disabled={!strokes.length || isRecognizing}
                onClick={clearDrawing}
              >
                Clear
              </button>
              <button
                type="button"
                data-testid="handwritten-problem-submit"
                disabled={!strokes.length || isRecognizing}
                onClick={submitHandwriting}
              >
                {isRecognizing ? 'Reading problem' : 'Submit problem'}
              </button>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
