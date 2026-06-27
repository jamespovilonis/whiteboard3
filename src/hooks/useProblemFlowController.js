import { useCallback, useMemo, useState } from 'react';
import { getRecognitionApiUrl } from '../recognition/config.js';
import { recognizeStudentWriting } from '../recognition/studentWritingPipeline.js';
import {
  applyProblemRecognitionError,
  applyProblemRecognitionResult,
  createInitialProblemFlow,
  getActiveProblem,
  getCompletedRecognitionResults,
  getActiveModelResponse,
  reconcileProblemFlowWithStrokes,
  submitActiveProblem
} from '../state/problemFlow.js';

export function useProblemFlowController({ moveHomeViewport, engineRef }) {
  const [problemFlow, setProblemFlow] = useState(() => (
    createInitialProblemFlow(getViewportWidth())
  ));

  const modelResponse = useMemo(() => (
    getActiveModelResponse(problemFlow)
  ), [problemFlow]);

  const recognitionResults = useMemo(() => (
    getCompletedRecognitionResults(problemFlow)
  ), [problemFlow]);

  const reconcileStrokes = useCallback((strokes) => {
    setProblemFlow((currentFlow) => reconcileProblemFlowWithStrokes(currentFlow, strokes));
  }, []);

  const submitAnswer = useCallback(() => {
    const activeProblem = getActiveProblem(problemFlow);
    const strokes = engineRef?.current?.getStrokes?.() || [];
    const result = submitActiveProblem(problemFlow, getViewportWidth());
    setProblemFlow(result.flow);

    if (result.targetViewport) {
      moveHomeViewport(result.targetViewport, 420);
    }

    if (!activeProblem || activeProblem.answerStrokeIds.length === 0) return;

    recognizeStudentWriting({
      strokes,
      answerBox: activeProblem.answerBox,
      problemLatex: activeProblem.latex,
      previousLatex: previousLatexForSubmission(problemFlow, activeProblem.id),
      apiUrl: getRecognitionApiUrl(),
      detectLineBands: true,
      semanticScoring: true
    }).then((recognitionResult) => {
      setProblemFlow((currentFlow) => (
        applyProblemRecognitionResult(currentFlow, activeProblem.id, summarizeRecognitionResult(recognitionResult))
      ));
    }).catch((error) => {
      setProblemFlow((currentFlow) => (
        applyProblemRecognitionError(currentFlow, activeProblem.id, error)
      ));
    });
  }, [engineRef, moveHomeViewport, problemFlow]);

  return {
    problemFlow,
    modelResponse,
    recognitionResults,
    reconcileStrokes,
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
