import { useCallback, useEffect, useMemo, useState } from 'react';
import { getRecognitionApiUrl } from '../recognition/config.js';
import { recognizeStudentWriting } from '../recognition/studentWritingPipeline.js';
import {
  E2E_PROBLEM_SOURCE_ENABLED,
  loadE2EEquationSolvingProblems
} from '../state/equationProblemSource.js';
import {
  applyProblemRecognitionError,
  applyProblemRecognitionResult,
  createInitialProblemFlow,
  getActiveProblem,
  getCompletedRecognitionResults,
  getActiveModelResponse,
  reconcileProblemFlowWithStrokes,
  requestNextProblem,
  startCustomProblem,
  submitActiveProblem
} from '../state/problemFlow.js';

export function useProblemFlowController({ moveHomeViewport, engineRef, onRecognitionEvent }) {
  const [problemFlow, setProblemFlow] = useState(() => (
    createInitialProblemFlow(getViewportWidth())
  ));

  useEffect(() => {
    if (!E2E_PROBLEM_SOURCE_ENABLED) return undefined;

    let didCancel = false;
    loadE2EEquationSolvingProblems().then((problemDefinitions) => {
      if (didCancel) return;
      setProblemFlow(createInitialProblemFlow(getViewportWidth(), problemDefinitions));
      onRecognitionEvent?.('problem-source-loaded', {
        problemCount: problemDefinitions.length,
        firstProblemLatex: problemDefinitions[0]?.latex || null
      });
    });

    return () => {
      didCancel = true;
    };
  }, [onRecognitionEvent]);

  const modelResponse = useMemo(() => (
    getActiveModelResponse(problemFlow)
  ), [problemFlow]);

  const recognitionResults = useMemo(() => (
    getCompletedRecognitionResults(problemFlow)
  ), [problemFlow]);

  const reconcileStrokes = useCallback((strokes) => {
    setProblemFlow((currentFlow) => reconcileProblemFlowWithStrokes(currentFlow, strokes));
  }, []);

  const createCustomProblem = useCallback((latex) => {
    const started = startCustomProblem(problemFlow, latex, getViewportWidth());
    setProblemFlow(started.flow);

    if (started?.targetViewport) {
      moveHomeViewport(started.targetViewport, 420);
    }

    if (started?.problem) {
      onRecognitionEvent?.('custom-problem-created', {
        problemId: started.problem.id,
        latex: started.problem.latex
      });
    }
  }, [moveHomeViewport, onRecognitionEvent, problemFlow]);

  const submitAnswer = useCallback(() => {
    const activeProblem = getActiveProblem(problemFlow);
    if (!activeProblem) {
      onRecognitionEvent?.('submit-active-problem', {
        problemId: null,
        strokeCount: engineRef?.current?.getStrokes?.()?.length || 0,
        answerStrokeCount: 0
      });
      onRecognitionEvent?.('recognition-skipped', {
        problemId: null,
        reason: 'no-active-problem'
      });
      return;
    }

    const strokes = engineRef?.current?.getStrokes?.() || [];
    const result = submitActiveProblem(problemFlow, getViewportWidth());
    onRecognitionEvent?.('submit-active-problem', {
      problemId: activeProblem?.id || null,
      strokeCount: strokes.length,
      answerStrokeCount: activeProblem?.answerStrokeIds?.length || 0
    });
    setProblemFlow(result.flow);

    if (result.targetViewport) {
      moveHomeViewport(result.targetViewport, 420);
    }

    if (!activeProblem || activeProblem.answerStrokeIds.length === 0) {
      onRecognitionEvent?.('recognition-skipped', {
        problemId: activeProblem?.id || null,
        reason: activeProblem ? 'empty-answer' : 'no-active-problem'
      });
      return;
    }

    onRecognitionEvent?.('recognition-start', {
      problemId: activeProblem.id,
      strokeCount: strokes.length,
      answerStrokeCount: activeProblem.answerStrokeIds.length,
      answerBox: activeProblem.answerBox
    });
    recognizeStudentWriting({
      strokes,
      answerBox: activeProblem.answerBox,
      problemLatex: activeProblem.latex,
      problemMetadata: activeProblem.metadata || {},
      previousLatex: previousLatexForSubmission(problemFlow, activeProblem.id),
      apiUrl: getRecognitionApiUrl(),
      detectLineBands: true,
      semanticScoring: true
    }).then((recognitionResult) => {
      const summary = summarizeRecognitionResult(recognitionResult);
      setProblemFlow((currentFlow) => (
        applyProblemRecognitionResult(currentFlow, activeProblem.id, summary)
      ));
      onRecognitionEvent?.('recognition-complete', {
        problemId: activeProblem.id,
        lineCount: summary.lines.length,
        candidateCount: summary.candidatePredictions.length,
        latex: summary.latex
      });
    }).catch((error) => {
      setProblemFlow((currentFlow) => (
        applyProblemRecognitionError(currentFlow, activeProblem.id, error)
      ));
      onRecognitionEvent?.('recognition-error', {
        problemId: activeProblem.id,
        error: error instanceof Error ? error.message : String(error)
      });
    });
  }, [engineRef, moveHomeViewport, onRecognitionEvent, problemFlow]);

  const goToNextProblem = useCallback(() => {
    const result = requestNextProblem(problemFlow, getViewportWidth());
    setProblemFlow(result.flow);

    if (result.targetViewport) {
      moveHomeViewport(result.targetViewport, 420);
    }

    onRecognitionEvent?.('next-problem', {
      activeProblemId: result.flow.activeProblemId || null,
      awaitingEquation: Boolean(result.flow.awaitingEquation)
    });
  }, [moveHomeViewport, onRecognitionEvent, problemFlow]);

  return {
    problemFlow,
    modelResponse,
    recognitionResults,
    reconcileStrokes,
    createCustomProblem,
    goToNextProblem,
    submitAnswer
  };
}

function getViewportWidth() {
  if (typeof window === 'undefined') return 1024;
  return window.innerWidth || 1024;
}

export function previousLatexForSubmission(_flow, _problemId) {
  return [];
}

export function summarizeRecognitionResult(result) {
  return {
    latex: result.latex,
    latexLines: result.latexLines,
    lines: result.lines.map((line) => ({
      lineIndex: line.lineIndex,
      candidateId: line.candidateId,
      debugLabel: line.debugLabel,
      selected: Boolean(line.selected),
      profiles: line.profiles,
      strokeIds: line.strokeIds,
      tightBbox: line.tightBbox,
      image: summarizeLineImage(line.image),
      latex: line.latex,
      acceptedLatex: line.acceptedLatex,
      ocrLatex: line.ocrLatex,
      candidates: line.candidates,
      prediction: line.prediction,
      contextualSemantic: line.contextualSemantic,
      sequentialSemantic: line.sequentialSemantic,
      semanticRetryPredictions: line.semanticRetryPredictions || [],
      retryPredictions: line.retryPredictions || [],
      ocrRepair: line.ocrRepair || null,
      evidenceScore: line.evidenceScore,
      timing: line.timing || null
    })),
    detection: result.detection,
    semantic: result.semantic,
    timing: result.timing || null,
    candidatePredictions: result.candidatePredictions.map((entry) => ({
      candidateId: entry.candidateId,
      debugLabel: entry.debugLabel,
      selected: Boolean(entry.selected),
      discarded: Boolean(entry.discarded),
      selectedLineIndex: entry.selectedLineIndex ?? null,
      profiles: entry.profiles,
      strokeIds: entry.strokeIds,
      tightBbox: entry.tightBbox,
      image: summarizeLineImage(entry.image),
      latex: entry.latex,
      acceptedLatex: entry.acceptedLatex || null,
      ocrLatex: entry.ocrLatex,
      candidates: entry.candidates,
      prediction: entry.prediction,
      semantic: entry.semantic,
      contextualSemantic: entry.contextualSemantic,
      sequentialSemantic: entry.sequentialSemantic,
      retryPredictions: entry.retryPredictions || [],
      semanticRetryPredictions: entry.semanticRetryPredictions || [],
      ocrRepair: entry.ocrRepair || null,
      evidenceScore: entry.evidenceScore,
      timing: entry.timing || null
    })),
    segmentation: {
      selected: result.segmentation.selected.map((candidate) => ({
        candidateId: candidate.candidateId,
        profiles: candidate.profiles,
        strokeIds: candidate.strokeIds,
        tightBbox: candidate.tightBbox
      })),
      candidates: result.segmentation.candidates.map((candidate) => ({
        candidateId: candidate.candidateId,
        profiles: candidate.profiles,
        strokeIds: candidate.strokeIds,
        tightBbox: candidate.tightBbox,
        conflicts: candidate.conflicts
      })),
      partitions: result.segmentation.partitions,
      parentCandidateId: result.segmentation.parentCandidateId,
      ocrSelectedCandidateIds: result.segmentation.ocrSelectedCandidateIds || []
    }
  };
}

function summarizeLineImage(image) {
  if (!image) return null;
  return {
    dataUrl: image.dataUrl || '',
    width: image.width ?? null,
    height: image.height ?? null,
    cssWidth: image.cssWidth ?? null,
    cssHeight: image.cssHeight ?? null,
    targetPixelHeight: image.targetPixelHeight ?? null,
    padding: image.padding ?? null,
    originX: image.originX ?? null,
    originY: image.originY ?? null,
    devicePixelRatio: image.devicePixelRatio ?? null
  };
}
