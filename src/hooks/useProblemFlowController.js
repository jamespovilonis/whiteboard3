import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getRecognitionApiUrl } from '../recognition/config.js';
import {
  attachRecognitionAuditFeedback,
  buildAttemptId,
  buildRecognitionAuditPayload,
  compactRecognitionResult,
  enqueueRecognitionAudit,
  getRecognitionAuditStatus,
  getRecognitionAuditDecision
} from '../recognition/auditClient.js';
import {
  createCorrectFeedback,
  requestMathFeedback,
  trimFeedbackText
} from '../feedback/feedbackClient.js';
import { IncrementalRecognitionScheduler } from '../recognition/incrementalRecognitionScheduler.js';
import {
  E2E_PROBLEM_SOURCE_ENABLED,
  loadE2EEquationSolvingProblems
} from '../state/equationProblemSource.js';
import {
  applyProblemGradingProgress,
  applyProblemFeedbackProgress,
  applyProblemRecognitionProgress,
  createInitialProblemFlow,
  getActiveProblem,
  getCompletedRecognitionResults,
  getActiveModelResponse,
  isProblemSubmittable,
  reconcileProblemFlowWithStrokes,
  requestNextProblem,
  startCustomProblem,
  submitActiveProblem
} from '../state/problemFlow.js';
import { gradeMathWork } from '../grading/gradingClient.js';

export function useProblemFlowController({ moveHomeViewport, engineRef, onRecognitionEvent }) {
  const [problemFlow, setProblemFlow] = useState(() => (
    createInitialProblemFlow(getViewportWidth())
  ));
  const latestStrokesRef = useRef([]);
  const schedulerRef = useRef(null);
  const startedInputSignaturesRef = useRef(new Set());
  const completedInputSignaturesRef = useRef(new Set());
  const auditInputSignaturesRef = useRef(new Set());
  const initializedGradingProblemIdsRef = useRef(new Set());
  const problemFlowRef = useRef(problemFlow);
  const feedbackControllersRef = useRef(new Map());
  const feedbackGenerationsRef = useRef(new Map());
  const feedbackByAttemptRef = useRef(new Map());
  const auditByAttemptRef = useRef(new Map());

  useEffect(() => {
    problemFlowRef.current = problemFlow;
  }, [problemFlow]);

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
          if (!shouldAcceptRecognitionSnapshot(targetProblem)) return currentFlow;
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
          maybeEnqueueRecognitionAudit({
            snapshot,
            inputSignature,
            problemFlow: problemFlowRef.current,
            strokes: latestStrokesRef.current,
            auditInputSignaturesRef,
            feedbackByAttemptRef,
            auditByAttemptRef,
            onRecognitionEvent
          });
          maybeStartMathFeedback({
            snapshot,
            inputSignature,
            problemFlow: problemFlowRef.current,
            feedbackControllersRef,
            feedbackGenerationsRef,
            feedbackByAttemptRef,
            auditByAttemptRef,
            setProblemFlow,
            onRecognitionEvent
          });
        }
      }
    });
    schedulerRef.current = scheduler;

    return () => {
      for (const controller of feedbackControllersRef.current.values()) {
        controller.abort();
      }
      feedbackControllersRef.current.clear();
      scheduler.dispose();
      if (schedulerRef.current === scheduler) schedulerRef.current = null;
    };
  }, [onRecognitionEvent]);

  useEffect(() => {
    const activeProblem = getActiveProblem(problemFlow);
    const shouldRecognize = shouldKeepRecognitionAttached(activeProblem);
    schedulerRef.current?.update({
      problemId: shouldRecognize ? activeProblem.id : null,
      strokes: latestStrokesRef.current,
      answerBox: shouldRecognize ? activeProblem?.answerBox || null : null,
      problemLatex: shouldRecognize ? activeProblem?.latex || '' : '',
      problemMetadata: shouldRecognize ? activeProblem?.metadata || {} : {},
      previousLatex: activeProblem ? previousLatexForSubmission(problemFlow, activeProblem.id) : [],
      apiUrl: getRecognitionApiUrl()
    });
  }, [problemFlow]);

  useEffect(() => {
    const activeProblem = getActiveProblem(problemFlow);
    if (!activeProblem?.id || !activeProblem.latex) return undefined;
    if (initializedGradingProblemIdsRef.current.has(activeProblem.id)) return undefined;
    initializedGradingProblemIdsRef.current.add(activeProblem.id);

    let didCancel = false;
    const problemId = activeProblem.id;
    const problemLatex = activeProblem.latex;
    const problemMetadata = activeProblem.metadata || {};
    gradeMathWork({
      problemLatex,
      problemMetadata,
      lines: []
    }, {
      apiUrl: getRecognitionApiUrl(),
      timeoutMs: 5000
    }).then((grading) => {
      if (didCancel || !grading || grading.failed) return;
      setProblemFlow((currentFlow) => (
        applyProblemGradingProgress(currentFlow, problemId, {
          ...grading,
          source: 'python-grader',
          failed: false
        })
      ));
    }).catch(() => {
      // Initial solution generation is helpful but non-blocking.
    });

    return () => {
      didCancel = true;
    };
  }, [problemFlow]);

  const reconcileStrokes = useCallback((strokes) => {
    invalidateFeedbackForActiveProblem(problemFlowRef.current, feedbackControllersRef, feedbackGenerationsRef);
    latestStrokesRef.current = strokes || [];
    setProblemFlow((currentFlow) => reconcileProblemFlowWithStrokes(currentFlow, strokes));
  }, []);

  const beginStroke = useCallback(() => {
    invalidateFeedbackForActiveProblem(problemFlowRef.current, feedbackControllersRef, feedbackGenerationsRef);
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
        latex: started.problem.latex,
        problemType: started.problem.metadata?.problemType || started.problem.kind || 'equation-solving'
      });
    }
  }, [moveHomeViewport, onRecognitionEvent, problemFlow]);

  const submitAnswer = useCallback(() => {
    const activeProblem = getActiveProblem(problemFlow);
    const strokes = engineRef?.current?.getStrokes?.() || [];
    const result = submitActiveProblem(problemFlow, getViewportWidth());
    const submittedProblem = getActiveProblem(result.flow);
    setProblemFlow(result.flow);
    schedulerRef.current?.flushNow?.();
    onRecognitionEvent?.('submit-active-problem', {
      problemId: activeProblem?.id || null,
      strokeCount: strokes.length,
      answerStrokeCount: activeProblem?.answerStrokeIds?.length || 0,
      frozen: Boolean(submittedProblem?.answerBoxFrozen)
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

function shouldAcceptRecognitionSnapshot(problem) {
  if (!problem) return false;
  if (problem.status === 'solving') return true;
  if (problem.status !== 'submitted') return false;
  if (problem.revisionAllowed) return true;
  return !recognitionIsSettled(problem.recognition);
}

function shouldKeepRecognitionAttached(problem) {
  if (!problem?.answerStrokeIds?.length) return false;
  if (problem.status === 'solving') return true;
  if (problem.status !== 'submitted') return false;
  if (problem.revisionAllowed) return true;
  return !recognitionIsSettled(problem.recognition);
}

export function shouldRunRecognitionForProblem(problem) {
  return shouldKeepRecognitionAttached(problem);
}

function recognitionIsSettled(recognition) {
  if (['empty', 'error'].includes(recognition?.status)) return true;
  if (recognition?.status !== 'complete') return false;
  const realtime = recognition.result?.realtime || recognition.realtime || null;
  if (realtime?.allFinal === false) return false;
  const components = Array.isArray(realtime?.components) ? realtime.components : [];
  if (components.some((component) => component?.status !== 'final' || component?.contested)) return false;
  const grading = recognition.result?.grading || null;
  return !grading || (grading.status !== 'pending' && !grading.running);
}

function maybeEnqueueRecognitionAudit({
  snapshot,
  inputSignature,
  problemFlow,
  strokes,
  auditInputSignaturesRef,
  feedbackByAttemptRef,
  auditByAttemptRef,
  onRecognitionEvent
}) {
  const result = snapshot?.result || null;
  if (!result || result.realtime?.allFinal === false || snapshot?.realtime?.allFinal === false) return;
  const problem = (problemFlow?.problems || []).find((item) => item.id === snapshot.problemId);
  if (!problem) return;

  const auditKey = `${snapshot.problemId || ''}::${inputSignature || ''}`;
  if (!auditKey.trim() || auditInputSignaturesRef.current.has(auditKey)) return;

  const decision = getRecognitionAuditDecision(result, {
    inputSignature,
    sampleKey: auditKey
  });
  if (!decision.shouldAudit) {
    onRecognitionEvent?.('recognition-audit-skipped', {
      problemId: snapshot.problemId,
      inputSignature,
      triggerReasons: [],
      sampled: false
    });
    return;
  }

  auditInputSignaturesRef.current.add(auditKey);
  const payload = buildRecognitionAuditPayload({
    problem,
    result,
    strokes,
    inputSignature,
    triggerReasons: decision.triggerReasons,
    feedback: feedbackByAttemptRef.current.get(buildAttemptId(problem.id, inputSignature)) || null
  });

  enqueueRecognitionAudit(payload, { apiUrl: getRecognitionApiUrl() })
    .then((response) => {
      const attemptId = payload.attemptId;
      onRecognitionEvent?.('recognition-audit-queued', {
        problemId: snapshot.problemId,
        inputSignature,
        attemptId,
        auditId: response.auditId || null,
        triggerReasons: decision.triggerReasons,
        sampled: decision.sampled,
        queued: response.queued !== false
      });
      if (response.queued === false) {
        onRecognitionEvent?.('recognition-audit-disabled', {
          problemId: snapshot.problemId,
          inputSignature,
          auditId: response.auditId || null,
          triggerReasons: decision.triggerReasons
        });
        return null;
      }
      if (response.auditId) {
        auditByAttemptRef.current.set(attemptId, {
          auditId: response.auditId,
          problemId: snapshot.problemId,
          inputSignature
        });
        const feedback = feedbackByAttemptRef.current.get(attemptId);
        if (feedback) {
          attachFeedbackToAudit({
            auditId: response.auditId,
            problemId: snapshot.problemId,
            inputSignature,
            attemptId,
            feedback,
            onRecognitionEvent
          });
        }
        return pollRecognitionAuditStatus({
          auditId: response.auditId,
          problemId: snapshot.problemId,
          inputSignature,
          triggerReasons: decision.triggerReasons,
          onRecognitionEvent
        });
      }
      return null;
    })
    .catch((error) => {
      onRecognitionEvent?.('recognition-audit-error', {
        problemId: snapshot.problemId,
        inputSignature,
        triggerReasons: decision.triggerReasons,
        error: error instanceof Error ? error.message : String(error)
      });
    });
}

function maybeStartMathFeedback({
  snapshot,
  inputSignature,
  problemFlow,
  feedbackControllersRef,
  feedbackGenerationsRef,
  feedbackByAttemptRef,
  auditByAttemptRef,
  setProblemFlow,
  onRecognitionEvent
}) {
  const result = snapshot?.result || null;
  if (!result || result.realtime?.allFinal === false || snapshot?.realtime?.allFinal === false) return;
  const grading = result.grading || null;
  if (!grading || grading.status === 'pending' || grading.failed) return;
  const problem = (problemFlow?.problems || []).find((item) => item.id === snapshot.problemId);
  if (!problem) return;

  const attemptId = buildAttemptId(problem.id, inputSignature);
  if (feedbackByAttemptRef.current.has(attemptId)) return;

  const problemId = problem.id;
  const generation = feedbackGenerationsRef.current.get(problemId) || 0;
  const problemStatus = grading?.result?.problemStatus || '';

  if (problemStatus === 'correct') {
    const feedback = createCorrectFeedback({ attemptId, inputSignature });
    feedbackByAttemptRef.current.set(attemptId, feedback);
    setProblemFlow((currentFlow) => (
      shouldAcceptFeedback(currentFlow, problemId, inputSignature, generation, feedbackGenerationsRef)
        ? applyProblemFeedbackProgress(currentFlow, problemId, feedback)
        : currentFlow
    ));
    attachFeedbackToQueuedAudit({
      auditByAttemptRef,
      attemptId,
      feedback,
      onRecognitionEvent
    });
    return;
  }

  const controller = new AbortController();
  feedbackControllersRef.current.set(attemptId, controller);
  setProblemFlow((currentFlow) => (
    shouldAcceptFeedback(currentFlow, problemId, inputSignature, generation, feedbackGenerationsRef)
      ? applyProblemFeedbackProgress(currentFlow, problemId, {
          status: 'pending',
          attemptId,
          inputSignature,
          source: 'ollama'
        })
      : currentFlow
  ));

  requestMathFeedback(buildMathFeedbackRequest({ problem, result, grading, inputSignature, attemptId }), {
    apiUrl: getRecognitionApiUrl(),
    signal: controller.signal
  }).then((feedback) => {
    feedbackControllersRef.current.delete(attemptId);
    if (!feedback || feedback.status === 'aborted') return;
    const normalizedFeedback = {
      ...feedback,
      attemptId,
      inputSignature,
      status: 'complete',
      text: trimFeedbackText(feedback.text)
    };
    if (!normalizedFeedback.text) return;
    feedbackByAttemptRef.current.set(attemptId, normalizedFeedback);
    setProblemFlow((currentFlow) => (
      shouldAcceptFeedback(currentFlow, problemId, inputSignature, generation, feedbackGenerationsRef)
        ? applyProblemFeedbackProgress(currentFlow, problemId, normalizedFeedback)
        : currentFlow
    ));
    attachFeedbackToQueuedAudit({
      auditByAttemptRef,
      attemptId,
      feedback: normalizedFeedback,
      onRecognitionEvent
    });
  }).catch((error) => {
    feedbackControllersRef.current.delete(attemptId);
    onRecognitionEvent?.('feedback-error', {
      problemId,
      inputSignature,
      attemptId,
      error: error instanceof Error ? error.message : String(error)
    });
  });
}

function buildMathFeedbackRequest({ problem, result, grading, inputSignature, attemptId }) {
  return {
    problemId: problem.id,
    problemLatex: problem.latex || '',
    problemMetadata: problem.metadata || {},
    inputSignature,
    attemptId,
    grading,
    fastResult: compactRecognitionResult(result)
  };
}

function shouldAcceptFeedback(flow, problemId, inputSignature, generation, feedbackGenerationsRef) {
  if ((feedbackGenerationsRef.current.get(problemId) || 0) !== generation) return false;
  const problem = (flow?.problems || []).find((item) => item.id === problemId);
  if (!problem) return false;
  const currentSignature = problem.recognition?.result?.realtime?.inputSignature ||
    problem.recognition?.realtime?.inputSignature ||
    '';
  return !currentSignature || currentSignature === inputSignature;
}

function invalidateFeedbackForActiveProblem(flow, feedbackControllersRef, feedbackGenerationsRef) {
  const problem = getActiveProblem(flow);
  if (!problem || !isProblemSubmittable(problem)) return;
  feedbackGenerationsRef.current.set(
    problem.id,
    (feedbackGenerationsRef.current.get(problem.id) || 0) + 1
  );
  for (const [attemptId, controller] of feedbackControllersRef.current.entries()) {
    if (String(attemptId || '').startsWith('attempt_')) {
      controller.abort();
      feedbackControllersRef.current.delete(attemptId);
    }
  }
}

function attachFeedbackToQueuedAudit({ auditByAttemptRef, attemptId, feedback, onRecognitionEvent }) {
  const audit = auditByAttemptRef.current.get(attemptId);
  if (!audit?.auditId) return;
  attachFeedbackToAudit({
    ...audit,
    attemptId,
    feedback,
    onRecognitionEvent
  });
}

function attachFeedbackToAudit({
  auditId,
  problemId,
  inputSignature,
  attemptId,
  feedback,
  onRecognitionEvent
}) {
  attachRecognitionAuditFeedback({
    auditId,
    problemId,
    inputSignature,
    attemptId,
    feedback
  }, { apiUrl: getRecognitionApiUrl() })
    .then((response) => {
      onRecognitionEvent?.('recognition-audit-feedback-attached', {
        problemId,
        inputSignature,
        attemptId,
        auditId,
        ...response
      });
    })
    .catch((error) => {
      onRecognitionEvent?.('recognition-audit-error', {
        problemId,
        inputSignature,
        attemptId,
        auditId,
        error: error instanceof Error ? error.message : String(error)
      });
    });
}

function pollRecognitionAuditStatus({
  auditId,
  problemId,
  inputSignature,
  triggerReasons,
  onRecognitionEvent
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
          onRecognitionEvent?.('recognition-audit-status', {
            problemId,
            inputSignature,
            auditId,
            triggerReasons,
            ...statusPayload
          });
        }
        if (statusPayload.done || pollCount >= maxPolls) return statusPayload;
        window.setTimeout(poll, pollIntervalMs);
        return statusPayload;
      })
      .catch((error) => {
        onRecognitionEvent?.('recognition-audit-error', {
          problemId,
          inputSignature,
          auditId,
          triggerReasons,
          error: error instanceof Error ? error.message : String(error)
        });
        return null;
      });
  };

  window.setTimeout(poll, 400);
  return null;
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
