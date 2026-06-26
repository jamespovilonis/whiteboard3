import { useCallback, useMemo, useState } from 'react';
import {
  createInitialProblemFlow,
  getActiveModelResponse,
  reconcileProblemFlowWithStrokes,
  submitActiveProblem
} from '../state/problemFlow.js';

export function useProblemFlowController({ moveHomeViewport }) {
  const [problemFlow, setProblemFlow] = useState(() => (
    createInitialProblemFlow(getViewportWidth())
  ));

  const modelResponse = useMemo(() => (
    getActiveModelResponse(problemFlow)
  ), [problemFlow]);

  const reconcileStrokes = useCallback((strokes) => {
    setProblemFlow((currentFlow) => reconcileProblemFlowWithStrokes(currentFlow, strokes));
  }, []);

  const submitAnswer = useCallback(() => {
    const result = submitActiveProblem(problemFlow, getViewportWidth());
    setProblemFlow(result.flow);

    if (result.targetViewport) {
      moveHomeViewport(result.targetViewport, 420);
    }
  }, [moveHomeViewport, problemFlow]);

  return {
    problemFlow,
    modelResponse,
    reconcileStrokes,
    submitAnswer
  };
}

function getViewportWidth() {
  if (typeof window === 'undefined') return 1024;
  return window.innerWidth || 1024;
}
