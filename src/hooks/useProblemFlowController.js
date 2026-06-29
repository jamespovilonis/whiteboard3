import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getRecognitionApiUrl } from '../recognition/config.js';
import { IncrementalRecognitionScheduler } from '../recognition/incrementalRecognitionScheduler.js';
import {
  E2E_PROBLEM_SOURCE_ENABLED,
  loadE2EEquationSolvingProblems
} from '../state/equationProblemSource.js';
import {
  applyProblemRecognitionProgress,
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
  const latestStrokesRef = useRef([]);
  const schedulerRef = useRef(null);
  const startedInputSignaturesRef = useRef(new Set());
  const completedInputSignaturesRef = useRef(new Set());

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

  useEffect(() => {
    const scheduler = new IncrementalRecognitionScheduler({
      debounceMs: 500,
      apiUrl: getRecognitionApiUrl(),
      semanticScoring: true,
      onEvent: (type, detail) => {
        onRecognitionEvent?.(type, detail);
      },
      onStateChange: (snapshot) => {
        const inputSignature = snapshot.realtime?.inputSignature || '';
        if (snapshot.status === 'pending' && inputSignature && !startedInputSignaturesRef.current.has(inputSignature)) {
          startedInputSignaturesRef.current.add(inputSignature);
          onRecognitionEvent?.('recognition-start', {
            problemId: snapshot.problemId,
            answerStrokeCount: snapshot.realtime?.answerStrokeCount || 0,
            realtime: true
          });
        }

        setProblemFlow((currentFlow) => {
          const targetProblem = currentFlow.problems.find((problem) => problem.id === snapshot.problemId);
          if (!targetProblem || targetProblem.status !== 'solving') return currentFlow;
          return applyProblemRecognitionProgress(currentFlow, snapshot.problemId, {
            status: snapshot.status,
            result: snapshot.result,
            realtime: snapshot.realtime
          });
        });

        if (snapshot.status === 'complete' && inputSignature && !completedInputSignaturesRef.current.has(inputSignature)) {
          completedInputSignaturesRef.current.add(inputSignature);
          const result = snapshot.result || {};
          onRecognitionEvent?.('recognition-complete', {
            problemId: snapshot.problemId,
            lineCount: result.lines?.length || 0,
            candidateCount: result.candidatePredictions?.length || 0,
            latex: result.latex || '',
            realtime: true
          });
        }
      }
    });
    schedulerRef.current = scheduler;

    return () => {
      scheduler.dispose();
      if (schedulerRef.current === scheduler) schedulerRef.current = null;
    };
  }, [onRecognitionEvent]);

  useEffect(() => {
    const activeProblem = getActiveProblem(problemFlow);
    schedulerRef.current?.update({
      problemId: activeProblem?.status === 'solving' ? activeProblem.id : null,
      strokes: latestStrokesRef.current,
      answerBox: activeProblem?.answerBox || null,
      problemLatex: activeProblem?.latex || '',
      problemMetadata: activeProblem?.metadata || {},
      previousLatex: activeProblem ? previousLatexForSubmission(problemFlow, activeProblem.id) : [],
      apiUrl: getRecognitionApiUrl()
    });
  }, [problemFlow]);

  const reconcileStrokes = useCallback((strokes) => {
    latestStrokesRef.current = strokes || [];
    setProblemFlow((currentFlow) => reconcileProblemFlowWithStrokes(currentFlow, strokes));
  }, []);

  const beginStroke = useCallback(() => {
    schedulerRef.current?.beginStroke();
  }, []);

  const setRecognitionPaused = useCallback((paused) => {
    schedulerRef.current?.setPaused(paused);
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
    const strokes = engineRef?.current?.getStrokes?.() || [];
    const result = submitActiveProblem(problemFlow, getViewportWidth());
    setProblemFlow(result.flow);
    schedulerRef.current?.update({
      problemId: null,
      strokes,
      answerBox: null,
      problemLatex: '',
      problemMetadata: {},
      previousLatex: [],
      apiUrl: getRecognitionApiUrl()
    });
    onRecognitionEvent?.('submit-active-problem', {
      problemId: activeProblem?.id || null,
      strokeCount: strokes.length,
      answerStrokeCount: activeProblem?.answerStrokeIds?.length || 0,
      frozen: activeProblem?.status === 'solving'
    });

  }, [engineRef, onRecognitionEvent, problemFlow]);

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
    beginStroke,
    setRecognitionPaused,
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
      grading: line.grading || null,
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
      grading: entry.grading || null,
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
    },
    grading: result.grading || null
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
