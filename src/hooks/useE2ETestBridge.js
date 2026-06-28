import { useEffect } from 'react';
import { getActiveProblem } from '../state/problemFlow.js';

export const E2E_TEST_ENABLED = import.meta.env?.VITE_E2E_TEST === '1';

export function useE2ETestBridge({
  engineRef,
  viewport,
  problemFlow,
  recognitionResults,
  eventsRef
}) {
  useEffect(() => {
    if (!E2E_TEST_ENABLED || typeof window === 'undefined') return undefined;

    const bridge = {
      snapshot() {
        const activeProblem = getActiveProblem(problemFlow);
        const strokes = engineRef.current?.getStrokes?.() || [];

        return deepClone({
          strokes,
          activeProblem,
          answerBox: activeProblem?.answerBox || null,
          problemFlow,
          recognitionResults,
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
      }
    };

    window.__whiteboardE2E = bridge;

    return () => {
      if (window.__whiteboardE2E === bridge) {
        delete window.__whiteboardE2E;
      }
    };
  }, [engineRef, eventsRef, problemFlow, recognitionResults, viewport]);
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
