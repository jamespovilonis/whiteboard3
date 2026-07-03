import katex from 'katex';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useToolbarCollapse } from '../hooks/useToolbarCollapse.js';
import {
  buildProblemInputAuditPayload,
  enqueueRecognitionAudit,
  getRecognitionAuditStatus,
  normalizeProblemInputRecognitionResult
} from '../recognition/auditClient.js';
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
  },
  simplify: {
    problemType: 'simplify-expression',
    title: 'Write an expression',
    copy: 'Draw the algebraic expression you want to simplify.',
    actionLabel: 'Simplify',
    previewLabel: 'Recognized expression preview'
  }
};

export default function HandwrittenProblemDialog({ onSubmit, onAuditEvent }) {
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
  const lastRecognitionRef = useRef(null);
  const auditInputSignaturesRef = useRef(new Set());
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
      lastRecognitionRef.current = null;
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
    lastRecognitionRef.current = null;
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
      const normalizedResult = normalizeProblemInputRecognitionResult(result);
      const latex = String(normalizedResult?.latex || '').trim();
      if (!latex) {
        setStatus('error');
        setError('Unable to read the problem. Adjust the drawing and try again.');
        return;
      }
      lastRecognitionRef.current = {
        result: normalizedResult,
        strokes: currentStrokes,
        latex,
        mode,
        problemType: modeConfig.problemType,
        inputSignature: buildProblemInputSignature({
          mode,
          problemType: modeConfig.problemType,
          latex,
          result,
          strokes: currentStrokes
        })
      };
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
  }, [isRecognizing, mode, modeConfig.problemType, strokes]);

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
    auditProblemInputAdjustment({
      entry: lastRecognitionRef.current,
      auditInputSignaturesRef,
      onAuditEvent
    });
    setStatus('drawing');
    setError('');
  }, [onAuditEvent]);

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
          <button
            type="button"
            className={mode === 'simplify' ? 'is-selected' : ''}
            aria-pressed={mode === 'simplify'}
            data-testid="handwritten-problem-type-simplify"
            onClick={() => setMode('simplify')}
          >
            Simplify
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

function auditProblemInputAdjustment({ entry, auditInputSignaturesRef, onAuditEvent }) {
  if (!entry?.result || !entry?.strokes?.length) return;
  const inputSignature = entry.inputSignature || buildProblemInputSignature(entry);
  const auditKey = `problem-input::${inputSignature}`;
  if (auditInputSignaturesRef.current.has(auditKey)) return;
  auditInputSignaturesRef.current.add(auditKey);

  const answerBox = {
    xMin: 0,
    yMin: 0,
    xMax: HANDWRITING_BOARD_SIZE.width,
    yMax: HANDWRITING_BOARD_SIZE.height
  };
  const triggerReasons = ['user_adjusted_problem_input'];
  const payload = buildProblemInputAuditPayload({
    mode: entry.mode,
    problemType: entry.problemType,
    recognizedLatex: entry.latex,
    result: entry.result,
    strokes: entry.strokes,
    answerBox,
    inputSignature,
    triggerReasons
  });

  enqueueRecognitionAudit(payload, { apiUrl: getRecognitionApiUrl() })
    .then((response) => {
      onAuditEvent?.('recognition-audit-queued', {
        problemId: payload.problemId,
        inputSignature,
        auditId: response.auditId || null,
        triggerReasons,
        sampled: false,
        queued: response.queued !== false,
        auditSubject: 'problem-input'
      });
      if (response.queued === false) {
        onAuditEvent?.('recognition-audit-disabled', {
          problemId: payload.problemId,
          inputSignature,
          auditId: response.auditId || null,
          triggerReasons,
          auditSubject: 'problem-input'
        });
        return null;
      }
      if (response.auditId) {
        return pollProblemInputAuditStatus({
          auditId: response.auditId,
          problemId: payload.problemId,
          inputSignature,
          triggerReasons,
          onAuditEvent
        });
      }
      return null;
    })
    .catch((error) => {
      onAuditEvent?.('recognition-audit-error', {
        problemId: payload.problemId,
        inputSignature,
        triggerReasons,
        auditSubject: 'problem-input',
        error: error instanceof Error ? error.message : String(error)
      });
    });
}

function pollProblemInputAuditStatus({
  auditId,
  problemId,
  inputSignature,
  triggerReasons,
  onAuditEvent
}) {
  const maxPolls = 90;
  const pollIntervalMs = 2000;
  let pollCount = 0;
  let lastStatus = 'queued';

  const poll = () => {
    pollCount += 1;
    return getRecognitionAuditStatus(auditId, { apiUrl: getRecognitionApiUrl() })
      .then((statusPayload) => {
        const status = statusPayload.status || 'unknown';
        if (status !== lastStatus || statusPayload.done) {
          lastStatus = status;
          onAuditEvent?.('recognition-audit-status', {
            problemId,
            inputSignature,
            auditId,
            triggerReasons,
            auditSubject: 'problem-input',
            ...statusPayload
          });
        }
        if (statusPayload.done || pollCount >= maxPolls) return statusPayload;
        window.setTimeout(poll, pollIntervalMs);
        return statusPayload;
      })
      .catch((error) => {
        onAuditEvent?.('recognition-audit-error', {
          problemId,
          inputSignature,
          auditId,
          triggerReasons,
          auditSubject: 'problem-input',
          error: error instanceof Error ? error.message : String(error)
        });
        return null;
      });
  };

  window.setTimeout(poll, 400);
  return null;
}

function buildProblemInputSignature({ mode = '', problemType = '', latex = '', result = {}, strokes = [] } = {}) {
  const strokeSignature = (strokes || []).map((stroke) => [
    stroke?.id || '',
    Math.round(Number(stroke?.startTime) || 0),
    Math.round(Number(stroke?.endTime) || 0),
    Math.round(Number(stroke?.canvasBbox?.xMin) || 0),
    Math.round(Number(stroke?.canvasBbox?.yMin) || 0),
    Math.round(Number(stroke?.canvasBbox?.xMax) || 0),
    Math.round(Number(stroke?.canvasBbox?.yMax) || 0)
  ].join(':')).join('|');
  return [
    'problem-input-adjust',
    mode,
    problemType,
    latex || result?.latex || '',
    strokeSignature
  ].join('::');
}
