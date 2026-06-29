import { useEffect } from 'react';
import { getActiveProblem } from '../state/problemFlow.js';

export const E2E_TEST_ENABLED = import.meta.env?.VITE_E2E_TEST === '1';

export function useE2ETestBridge({
  engineRef,
  viewport,
  problemFlow,
  recognitionResults,
  eventsRef,
  onRecognitionPausedChange
}) {
  useEffect(() => {
    if (!E2E_TEST_ENABLED || typeof window === 'undefined') return undefined;

    const bridge = {
      snapshot() {
        const activeProblem = getActiveProblem(problemFlow);
        const strokes = engineRef.current?.getStrokes?.() || [];
        const lightProblemFlow = stripRecognitionImageData(problemFlow);
        const lightActiveProblem = activeProblem
          ? lightProblemFlow.problems.find((problem) => problem.id === activeProblem.id) || activeProblem
          : null;

        return deepClone({
          strokes,
          activeProblem: lightActiveProblem,
          answerBox: lightActiveProblem?.answerBox || null,
          problemFlow: lightProblemFlow,
          recognitionResults: stripRecognitionImageData(recognitionResults),
          events: eventsRef.current || [],
          viewport,
          canvas: canvasRect()
        });
      },
      boardToScreen(point) {
        const rect = canvasRect();
        return {
          x: rect.left + (Number(point.x) - viewport.x) * viewport.scale,
          y: rect.top + (Number(point.y) - viewport.y) * viewport.scale
        };
      },
      screenToBoard(point) {
        const rect = canvasRect();
        return {
          x: viewport.x + (Number(point.x) - rect.left) / viewport.scale,
          y: viewport.y + (Number(point.y) - rect.top) / viewport.scale
        };
      },
      setRealtimeRecognitionPaused(paused) {
        onRecognitionPausedChange?.(Boolean(paused));
      },
      replaceStrokes(strokes, reason = 'e2e-inject') {
        engineRef.current?.replaceStrokesForE2E?.(strokes, reason);
      }
    };

    window.__whiteboardE2E = bridge;

    return () => {
      if (window.__whiteboardE2E === bridge) {
        delete window.__whiteboardE2E;
      }
    };
  }, [engineRef, eventsRef, onRecognitionPausedChange, problemFlow, recognitionResults, viewport]);
}

export function recordE2EEvent(eventsRef, type, detail = {}) {
  if (!E2E_TEST_ENABLED) return;

  eventsRef.current.push({
    type,
    time: Date.now(),
    performanceTime: performance.now(),
    detail: deepClone(detail)
  });
}

function canvasRect() {
  const canvas = document.querySelector('[data-testid="whiteboard-canvas"]');
  if (!canvas) {
    return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 };
  }

  const rect = canvas.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
    right: rect.right,
    bottom: rect.bottom
  };
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value ?? null));
}

function stripRecognitionImageData(value) {
  if (Array.isArray(value)) return value.map(stripRecognitionImageData);
  if (!value || typeof value !== 'object') return value;

  if (value.dataUrl && typeof value.dataUrl === 'string') {
    return {
      ...value,
      dataUrl: value.dataUrl ? '[stripped]' : ''
    };
  }

  const output = {};
  for (const [key, child] of Object.entries(value)) {
    output[key] = stripRecognitionImageData(child);
  }
  return output;
}
