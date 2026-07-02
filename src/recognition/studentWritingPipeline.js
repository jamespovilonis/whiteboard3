import {
  computeTightBbox,
  horizontalOverlapRatio,
  segmentMathLines,
  selectCandidateCover,
  scoreRecognitionEvidence,
  strokeBelongsToAnswerBox
} from './lineSegmentation.js';
import { rasterizeLineCandidate } from './lineRasterizer.js';
import { recognizeLineImage } from './ocrClient.js';
import { requestLineDetections } from './segmentationClient.js';
import { scoreLatexCandidates } from './semanticClient.js';
import { gradeMathWork } from '../grading/gradingClient.js';

const FRACTION_CHUNK_RASTER_HEIGHTS = [72, 88, 104];

export async function recognizeStudentWriting(options = {}) {
  const pipelineStartedAt = performanceNow();
  const {
    strokes = [],
    answerBox = null,
    detections = [],
    detectLineBands = false,
    detectLines = requestLineDetections,
    problemLatex = '',
    problemMetadata = {},
    previousLatex = [],
    apiUrl = '',
    model = 'comer',
    timeoutMs = 20000,
    detectionTimeoutMs = 10000,
    semanticScoring = false,
    semanticTimeoutMs = 5000,
    semanticCandidateLimit = 5,
    scoreSemantics = scoreLatexCandidates,
    rasterPadding = 24,
    initialRasterHeight = 104,
    initialRasterMinCssHeight = 1,
    retryRasterHeights = [88, 104, 72],
    structuralRetryTimeoutMs = 12000,
    semanticRetryRasterHeights = [48, 64, 72, 88, 104],
    chunkFallback = true,
    chunkFallbackMinCssWidth = 460,
    chunkFallbackMaxCssWidth = 340,
    chunkFallbackMinGap = 18,
    recognizeAlternatives = true,
    skipSingleStrokeAlternatives = true,
    stagedAlternativeRecognition = true,
    deferCoveredParentRecognition = true,
    initialRecognitionConcurrency = 1,
    signal = null,
    recognizeLine = recognizeLineImage,
    gradeWork = gradeMathWork,
    gradingTimeoutMs = 5000,
    finalizationBudgetMs = 15000
  } = options;
  const finalizationBudget = createFinalizationBudget(pipelineStartedAt, finalizationBudgetMs);
  const ignoredStrokeIds = (options.ignoredStrokeIds || [])
    .concat((strokes || []).filter((stroke) => stroke?.visualOnly).map((stroke) => stroke.id))
    .map(String);

  throwIfAborted(signal);
  const detection = await resolveDetections({
    strokes,
    answerBox,
    detections,
    detectLineBands,
    detectLines,
    apiUrl,
    timeoutMs: detectionTimeoutMs,
    signal
  });
  throwIfAborted(signal);

  const segmentation = segmentMathLines(strokes, {
    answerBox,
    detections: detection.detections,
    ignoredStrokeIds,
    problemLatex,
    previousLatex
  });
  const deterministicSegmentation = detection.detections?.length
    ? segmentMathLines(strokes, {
        answerBox,
        detections: [],
        ignoredStrokeIds,
        problemLatex,
        previousLatex
      })
    : null;
  const baselineCover = recognitionBaselineCover(
    segmentation.selected,
    deterministicSegmentation?.selected
  );

  const candidatesToRecognize = recognizeAlternatives
    ? candidatesForRecognition(segmentation)
    : segmentation.selected;
  const initialRecognitionSkipContext = {
    enabled: Boolean(recognizeAlternatives),
    skipSingleStrokeAlternatives: Boolean(skipSingleStrokeAlternatives),
    stagedAlternativeRecognition: Boolean(stagedAlternativeRecognition),
    candidates: candidatesToRecognize,
    baselineCandidateIds: new Set((baselineCover || []).map((candidate) => candidate.candidateId)),
    coveredByValidDeterministicLineCandidateIds: new Set()
  };
  const candidatePredictions = await recognizeInitialCandidates(candidatesToRecognize, {
    baselineCover,
    apiUrl,
    model,
    timeoutMs,
    problemLatex,
    previousLatex,
    rasterPadding,
    initialRasterHeight,
    initialRasterMinCssHeight,
    pipelineStartedAt,
    initialRecognitionSkipContext,
    deferCoveredParentRecognition: Boolean(deferCoveredParentRecognition),
    initialRecognitionConcurrency,
    signal,
    recognizeLine
  });
  throwIfAborted(signal);

  const evidenceByCandidateId = new Map(
    candidatePredictions.map((entry) => [entry.candidateId, entry.evidenceScore])
  );
  if (recognizeAlternatives && !finalizationBudget.check()) {
    const preSemanticSelected = selectCandidateCover(candidatesToRecognize, {
      scoreByCandidateId: evidenceByCandidateId,
      baselineCandidates: baselineCover,
    });
    await recognizeDeferredAlternativesInsideWeakSelection(preSemanticSelected, {
      candidates: candidatesToRecognize,
      candidatePredictions,
      evidenceByCandidateId,
      apiUrl,
      model,
      timeoutMs,
      problemLatex,
      previousLatex,
      pipelineStartedAt,
      rasterPadding,
      recognizeLine,
      signal
    });
    throwIfAborted(signal);
  }
  let semantic = await resolveSemanticScores({
    candidatePredictions,
    problemLatex,
    problemMetadata,
    previousLatex,
    semanticScoring,
    scoreSemantics,
    apiUrl,
    timeoutMs: semanticTimeoutMs,
    semanticCandidateLimit,
    signal
  });
  throwIfAborted(signal);
  const semanticByCandidateId = new Map(
    (semantic.candidateScores || []).map((entry) => [entry.candidateId, entry])
  );

  for (const entry of candidatePredictions) {
    const semanticEntry = semanticByCandidateId.get(entry.candidateId);
    if (!semanticEntry) continue;
    entry.semantic = semanticEntry;
    entry.grading = semanticEntry.grading || null;
    entry.timing.semanticElapsedSeconds = finiteSeconds(semantic.elapsedSeconds);
    entry.timing.submitToFinalPredictionSeconds = secondsSince(pipelineStartedAt);
    entry.evidenceScore += Number(semanticEntry.semanticScore) || 0;
    entry.evidenceScore += gradingEvidenceBoost(entry, semanticEntry.grading);
    evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
    const gradingLatex = gradingSelectedLatex(entry, semanticEntry);
    if (gradingLatex) {
      entry.latex = gradingLatex;
    } else if (semanticLatexSafeForReplacement(entry, semanticEntry) && shouldUseSemanticLatex(entry.latex, semanticEntry)) {
      entry.latex = semanticEntry.bestLatex;
    }
  }

  let selected = recognizeAlternatives
    ? selectCandidateCover(candidatesToRecognize, {
        scoreByCandidateId: evidenceByCandidateId,
        baselineCandidates: baselineCover,
      })
    : segmentation.selected;
  const contextualCandidateSemantic = await resolveContextualCandidateSemanticScores({
    candidatePredictions,
    problemLatex,
    problemMetadata,
    previousLatex,
    semanticScoring: semanticScoring && !semantic.failed,
    scoreSemantics,
    apiUrl,
    timeoutMs: semanticTimeoutMs,
    semanticCandidateLimit,
    signal
  });
  throwIfAborted(signal);
  const contextualSemanticByCandidateId = new Map(
    (contextualCandidateSemantic.candidateScores || []).map((entry) => [entry.candidateId, entry])
  );

  if (contextualSemanticByCandidateId.size > 0) {
    for (const entry of candidatePredictions) {
      const contextualEntry = contextualSemanticByCandidateId.get(entry.candidateId);
      if (!contextualEntry) continue;
      const baseSemanticScore = Number(entry.semantic?.semanticScore) || 0;
      const contextualScore = Number(contextualEntry.semanticScore) || 0;
      const evidenceDelta = clamp(contextualScore - baseSemanticScore, -3, 6);
      entry.contextualSemantic = {
        ...contextualEntry,
        evidenceDelta
      };
      entry.grading = contextualEntry.grading || entry.grading || null;
      entry.timing.contextualSemanticElapsedSeconds = finiteSeconds(contextualCandidateSemantic.elapsedSeconds);
      entry.timing.submitToFinalPredictionSeconds = secondsSince(pipelineStartedAt);
      entry.evidenceScore += evidenceDelta + gradingEvidenceBoost(entry, contextualEntry.grading);
      evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
      const gradingLatex = gradingSelectedLatex(entry, contextualEntry);
      if (gradingLatex) {
        entry.latex = gradingLatex;
      } else if (semanticLatexSafeForReplacement(entry, contextualEntry) && shouldUseSemanticLatex(entry.latex, contextualEntry)) {
        entry.latex = contextualEntry.bestLatex;
      }
    }
    if (recognizeAlternatives) {
      selected = selectCandidateCover(candidatesToRecognize, {
        scoreByCandidateId: evidenceByCandidateId,
        baselineCandidates: baselineCover,
      });
    }
  }
  selected = preferFullSolutionCandidates(selected, candidatePredictions, candidatesToRecognize);

  for (const candidate of selected) {
    throwIfAborted(signal);
    const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
    if (!entry?.skippedRecognition) continue;
    await recognizeSkippedSelectedEntry(entry, candidate, {
      apiUrl,
      model,
      timeoutMs,
      problemLatex,
      previousLatex,
      pipelineStartedAt,
      rasterPadding,
      recognizeLine,
      signal,
      evidenceByCandidateId
    });
  }

  const selectedIds = new Set(selected.map((candidate) => candidate.candidateId));
  if (chunkFallback && !finalizationBudget.check()) {
    for (const candidate of selected) {
      throwIfAborted(signal);
      if (finalizationBudget.check()) break;
      const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
      if (!entry || entry.skippedRecognition || !predictionNeedsRetry(entry.prediction) || !candidateCanUseChunking(candidate, {
        rasterPadding,
        minCssWidth: chunkFallbackMinCssWidth
      })) {
        continue;
      }
      await applyChunkFallbackToEntry(entry, candidate, {
        apiUrl,
        model,
        timeoutMs,
        rasterPadding,
        initialRasterHeight,
        minCssWidth: chunkFallbackMinCssWidth,
        maxChunkCssWidth: chunkFallbackMaxCssWidth,
        minGap: chunkFallbackMinGap,
        recognizeLine,
        problemLatex,
        previousLatex,
        signal,
        evidenceByCandidateId
      });
    }
  }
  if (retryRasterHeights?.length && !finalizationBudget.check()) {
    for (const candidate of selected) {
      throwIfAborted(signal);
      if (finalizationBudget.check()) break;
      const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
      if (!entry || entry.skippedRecognition || !predictionNeedsRetry(entry.prediction)) continue;
      const retry = await retrySelectedLineRecognition(candidate, {
        apiUrl,
        model,
        timeoutMs,
        problemLatex,
        rasterPadding,
        retryRasterHeights,
        finalizationBudget,
        skipRasterHeights: [entry.initialTargetPixelHeight],
        extendedTimeoutMs: structuralRetryTimeoutMs,
        signal,
        recognizeLine
      });
      if (!retry.attempts.length) continue;
      entry.retryPredictions = retry.attempts;
      const merged = mergePredictionAttempts([entry.prediction, ...retry.attempts]);
      entry.prediction = merged;
      entry.candidates = merged.candidates || [];
      entry.ocrLatex = predictionLatex(merged);
      entry.latex = merged.latex || merged.top?.latex || entry.latex || '';
      entry.evidenceScore = scoreRecognitionEvidence(candidate, merged, { problemLatex, previousLatex });
      entry.timing.retryElapsedSeconds = sumPredictionElapsedSeconds(retry.attempts);
      entry.timing.submitToFinalPredictionSeconds = secondsSince(pipelineStartedAt);
      entry.timing.ocrElapsedSeconds = finiteSeconds(merged.elapsedSeconds);
      evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
    }
  }
  if (chunkFallback && !finalizationBudget.check()) {
    for (const candidate of selected) {
      throwIfAborted(signal);
      if (finalizationBudget.check()) break;
      const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
      if (!entry || entry.skippedRecognition || !predictionNeedsRetry(entry.prediction)) continue;
      await applyChunkFallbackToEntry(entry, candidate, {
        apiUrl,
        model,
        timeoutMs,
        rasterPadding,
        initialRasterHeight,
        minCssWidth: chunkFallbackMinCssWidth,
        maxChunkCssWidth: chunkFallbackMaxCssWidth,
        minGap: chunkFallbackMinGap,
        recognizeLine,
        problemLatex,
        previousLatex,
        signal,
        evidenceByCandidateId
      });
    }
  }

  let recognizedLines = selected.map((candidate, index) => {
    const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
    return {
      ...entry,
      lineIndex: index,
      selected: true
    };
  });
  recognizedLines = postOcrMergePass(recognizedLines, {
    candidatePredictions,
    candidatesToRecognize,
    evidenceByCandidateId,
    problemLatex,
    previousLatex
  });
  applyGeometryOperationAnnotationRepairs(recognizedLines, problemLatex);

  let selectedLineSemantic = await resolveSelectedLineSemanticScores({
    recognizedLines,
    problemLatex,
    problemMetadata,
    previousLatex,
    semanticScoring: semanticScoring && !semantic.failed,
    scoreSemantics,
    apiUrl,
    timeoutMs: semanticTimeoutMs,
    semanticCandidateLimit,
    signal
  });
  throwIfAborted(signal);
  const selectedLineSemanticBeforeRetry = selectedLineSemantic;
  const selectedLineSemanticById = new Map(
    (selectedLineSemantic.lineScores || []).map((entry) => [entry.candidateId, entry])
  );

  let semanticRetryUsed = false;
  if (semanticRetryRasterHeights?.length && semanticScoring && !selectedLineSemantic.failed && !finalizationBudget.check()) {
    for (const line of recognizedLines) {
      throwIfAborted(signal);
      if (finalizationBudget.check()) break;
      const lineSemantic = selectedLineSemanticById.get(line.candidateId);
      if (!lineSemantic || !semanticNeedsRetry(line, lineSemantic)) continue;

      const candidate = selected.find((item) => item.candidateId === line.candidateId);
      if (!candidate) continue;

      const retry = await retrySelectedLineRecognition(candidate, {
        apiUrl,
        model,
        timeoutMs,
        rasterPadding,
        initialRasterHeight,
        retryRasterHeights: semanticRetryRasterHeights,
        finalizationBudget,
        skipRasterHeights: [
          line.initialTargetPixelHeight,
          ...(line.retryPredictions || []).map((attempt) => attempt.retryTargetPixelHeight)
        ],
        extendedTimeoutMs: structuralRetryTimeoutMs,
        chunkFallback,
        chunkFallbackMinCssWidth,
        chunkFallbackMaxCssWidth,
        chunkFallbackMinGap,
        signal,
        recognizeLine
      });
      if (!retry.attempts.length) continue;

      semanticRetryUsed = true;
      line.semanticRetryPredictions = retry.attempts;
      const hasSemanticChunkFallback = retry.attempts.some((attempt) => attempt.semanticRetryChunkFallback);
      const merged = mergePredictionAttempts(hasSemanticChunkFallback
        ? [
            ...retry.attempts,
            line.prediction,
            ...(line.retryPredictions || [])
          ]
        : [
            line.prediction,
            ...(line.retryPredictions || []),
            ...retry.attempts
          ]);
      line.prediction = merged;
      line.candidates = merged.candidates || [];
      line.ocrLatex = predictionLatex(merged);
      line.latex = merged.latex || merged.top?.latex || line.latex || '';
      line.evidenceScore = scoreRecognitionEvidence(candidate, merged, { problemLatex, previousLatex });
      line.timing = {
        ...(line.timing || {}),
        retryElapsedSeconds: sumPredictionElapsedSeconds(retry.attempts),
        submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt),
        ocrElapsedSeconds: finiteSeconds(merged.elapsedSeconds)
      };
    }
  }

  if (semanticRetryUsed) {
    selectedLineSemantic = await resolveSelectedLineSemanticScores({
      recognizedLines,
      problemLatex,
      problemMetadata,
      previousLatex,
      semanticScoring: semanticScoring && !semantic.failed,
      scoreSemantics,
      apiUrl,
      timeoutMs: semanticTimeoutMs,
      semanticCandidateLimit,
      signal
    });
    throwIfAborted(signal);
    selectedLineSemanticById.clear();
    for (const entry of selectedLineSemantic.lineScores || []) {
      selectedLineSemanticById.set(entry.candidateId, entry);
    }
  }

  const acceptedContextLatex = [...(previousLatex || [])].filter(Boolean);
  for (const line of recognizedLines) {
    if (line.skippedRecognition) continue;
    const lineSemantic = selectedLineSemanticById.get(line.candidateId);
    if (!lineSemantic) continue;
    line.sequentialSemantic = lineSemantic;
    line.grading = lineSemantic.grading || line.grading || null;
    line.timing = {
      ...(line.timing || {}),
      sequentialSemanticElapsedSeconds: finiteSeconds(selectedLineSemantic.elapsedSeconds),
      submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt)
    };
    if (line.ocrRepair?.source === 'geometry-operation-annotation') {
      line.excludedFromGrading = true;
      acceptedContextLatex.push(line.latex);
      continue;
    }
    const operationRepair = repairStandaloneOperationLatex(line.latex, lineSemantic, { problemLatex });
    if (operationRepair) {
      const operationRepairSource = detachedOperationOperand(line.latex) && /\\frac\b/.test(String(problemLatex || ''))
        ? 'geometry-operation-annotation'
        : 'standalone-operation';
      line.ocrRepair = {
        source: operationRepairSource,
        originalLatex: line.latex,
        repairedLatex: operationRepair,
        annotationBbox: operationRepairSource === 'geometry-operation-annotation' ? line.tightBbox || null : undefined
      };
      if (operationRepairSource === 'geometry-operation-annotation') {
        line.excludedFromGrading = true;
      }
      line.latex = operationRepair;
      acceptedContextLatex.push(line.latex);
      continue;
    }
    const contextualOperationRepair = repairOperationAnnotationFromPrevious(
      line.latex,
      acceptedContextLatex
    );
    if (contextualOperationRepair) {
      line.ocrRepair = {
        source: 'contextual-operation',
        originalLatex: line.latex,
        repairedLatex: contextualOperationRepair
      };
      line.latex = contextualOperationRepair;
      acceptedContextLatex.push(line.latex);
      continue;
    }
    const gradingLatex = gradingSelectedLatex(line, lineSemantic);
    if (gradingLatex) {
      line.latex = gradingLatex;
    } else if (semanticLatexSafeForReplacement(line, lineSemantic) && shouldUseSemanticLatex(line.latex, lineSemantic)) {
      line.latex = lineSemantic.bestLatex;
    }
    const quadraticFormulaRepair = repairQuadraticFormulaFromProblem(line.latex, problemLatex);
    if (quadraticFormulaRepair && quadraticFormulaRepair !== line.latex) {
      line.ocrRepair = {
        source: 'contextual-quadratic-formula',
        originalLatex: line.latex,
        repairedLatex: quadraticFormulaRepair
      };
      line.latex = quadraticFormulaRepair;
    }
    if (line.latex) acceptedContextLatex.push(line.latex);
  }
  for (const line of recognizedLines) {
    if (line.skippedRecognition) {
      line.acceptedLatex = '';
      line.timing = {
        ...(line.timing || {}),
        submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt)
      };
      continue;
    }
    const quadraticFormulaRepair = repairQuadraticFormulaFromProblem(line.latex, problemLatex);
    if (quadraticFormulaRepair && quadraticFormulaRepair !== line.latex) {
      line.ocrRepair = {
        source: 'contextual-quadratic-formula',
        originalLatex: line.latex,
        repairedLatex: quadraticFormulaRepair
      };
      line.latex = quadraticFormulaRepair;
    }
    const rationalProblemRepair = repairInitialRationalProblemLine(line.latex, problemLatex, line.lineIndex);
    if (rationalProblemRepair && rationalProblemRepair !== line.latex) {
      line.ocrRepair = {
        source: 'contextual-rational-problem',
        originalLatex: line.latex,
        repairedLatex: rationalProblemRepair
      };
      line.latex = rationalProblemRepair;
    }
    line.latex = normalizeContextualVariableCase(line.latex, [
      problemLatex,
      ...previousLatex,
      ...acceptedContextLatex
    ]);
    line.acceptedLatex = line.latex;
    line.timing = {
      ...(line.timing || {}),
      submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt)
    };
  }
  applyFinalCandidateDebugState(candidatePredictions, recognizedLines, pipelineStartedAt);
  semantic = {
    ...semantic,
    contextual: contextualCandidateSemantic,
    sequential: selectedLineSemantic,
    sequentialBeforeRetry: semanticRetryUsed ? selectedLineSemanticBeforeRetry : null
  };
  const answerManifest = semantic.answerManifest ||
    contextualCandidateSemantic.answerManifest ||
    selectedLineSemantic.answerManifest ||
    null;
  const gradableLines = gradableRecognitionLines(recognizedLines);
  let grading = buildLiveGradingResult({
    problemLatex,
    answerManifest,
    lines: gradableLines
  });

  // Defer to the Python grader for the authoritative problem-level verdict
  // when the gateway is available. The JS aggregation is a fast fallback.
  if (apiUrl && typeof gradeWork === 'function') {
    try {
      const pythonGrading = await gradeWork({
        problemLatex,
        problemMetadata,
        lines: gradableLines.map((line) => ({
          lineIndex: line.lineIndex,
          latex: line.acceptedLatex || line.latex || '',
          candidates: gradingPayloadCandidates(line)
        }))
      }, { apiUrl, timeoutMs: gradingTimeoutMs, signal });
      if (!pythonGrading.failed) {
        grading = {
          ...grading,
          ...pythonGrading,
          source: 'python-grader',
          failed: false
        };
      }
    } catch (_error) {
      // Keep the JS-aggregated grading result on failure.
    }
  }

  return {
    segmentation: {
      ...segmentation,
      selected,
      ocrSelectedCandidateIds: [...selectedIds]
    },
    detection,
    semantic,
    candidatePredictions,
    lines: recognizedLines,
    latexLines: gradableLines.map((line) => line.acceptedLatex),
    latex: gradableLines.map((line) => line.acceptedLatex).filter(Boolean).join(' \\\\ '),
    grading,
    timing: {
      totalElapsedSeconds: secondsSince(pipelineStartedAt),
      ...(finalizationBudget.check() ? { finalizationBudgetExceeded: true } : {})
    }
  };
}

async function recognizeInitialCandidates(candidates, {
  baselineCover,
  apiUrl,
  model,
  timeoutMs,
  problemLatex,
  previousLatex,
  rasterPadding,
  initialRasterHeight,
  initialRasterMinCssHeight,
  pipelineStartedAt,
  initialRecognitionSkipContext,
  deferCoveredParentRecognition,
  initialRecognitionConcurrency,
  signal,
  recognizeLine
}) {
  const candidateList = Array.isArray(candidates) ? candidates : [];
  const recognizeCandidate = (candidate) => recognizeInitialCandidate(candidate, {
    apiUrl,
    model,
    timeoutMs,
    problemLatex,
    previousLatex,
    rasterPadding,
    initialRasterHeight,
    initialRasterMinCssHeight,
    pipelineStartedAt,
    initialRecognitionSkipContext,
    signal,
    recognizeLine
  });

  if (!deferCoveredParentRecognition || candidateList.length === 0) {
    return mapWithConcurrency(
      candidateList,
      initialRecognitionConcurrency,
      recognizeCandidate
    );
  }

  const predictionByCandidateId = new Map();
  const baselineIds = new Set((baselineCover || []).map((candidate) => candidate.candidateId));
  const baselineCandidates = orderInitialRecognitionCandidates(
    candidateList.filter((candidate) => baselineIds.has(candidate.candidateId)),
    baselineIds
  );

  const baselinePredictions = await mapWithConcurrency(
    baselineCandidates,
    initialRecognitionConcurrency,
    recognizeCandidate
  );
  for (const prediction of baselinePredictions) {
    predictionByCandidateId.set(prediction.candidateId, prediction);
  }

  const validBaselineCandidates = baselineCandidates.filter((candidate) => (
    deterministicPredictionIsValid(
      predictionByCandidateId.get(candidate.candidateId)
    )
  ));
  initialRecognitionSkipContext.coveredByValidDeterministicLineCandidateIds =
    coveredLargerCandidateIds(candidateList, validBaselineCandidates);

  const remainingCandidates = orderInitialRecognitionCandidates(
    candidateList.filter((candidate) => !predictionByCandidateId.has(candidate.candidateId)),
    baselineIds
  );
  const remainingPredictions = await mapWithConcurrency(
    remainingCandidates,
    initialRecognitionConcurrency,
    recognizeCandidate
  );
  for (const prediction of remainingPredictions) {
    predictionByCandidateId.set(prediction.candidateId, prediction);
  }

  return candidateList.map((candidate) => predictionByCandidateId.get(candidate.candidateId));
}

async function recognizeInitialCandidate(candidate, {
  apiUrl,
  model,
  timeoutMs,
  problemLatex,
  previousLatex,
  rasterPadding,
  initialRasterHeight,
  initialRasterMinCssHeight,
  pipelineStartedAt,
  initialRecognitionSkipContext,
  signal,
  recognizeLine
}) {
  const initialTargetPixelHeight = initialTargetPixelHeightForCandidate(candidate, {
    rasterPadding,
    initialRasterHeight,
    initialRasterMinCssHeight
  });
  const initialSkipReason = initialRecognitionSkipReason(candidate, initialRecognitionSkipContext);
  if (initialSkipReason) {
    const prediction = skippedInitialRecognitionPrediction(model, initialSkipReason);
    const initialPredictionElapsedSeconds = secondsSince(pipelineStartedAt);
    return {
      candidateId: candidate.candidateId,
      profiles: (candidate.profiles || []).slice(),
      strokeIds: (candidate.strokeIds || []).slice(),
      tightBbox: candidate.tightBbox ? { ...candidate.tightBbox } : null,
      image: null,
      prediction,
      ocrLatex: '',
      latex: '',
      candidates: [],
      skippedRecognition: true,
      skipReason: initialSkipReason,
      initialTargetPixelHeight,
      evidenceScore: scoreRecognitionEvidence(candidate, prediction, { problemLatex, previousLatex }),
      timing: {
        submitToInitialPredictionSeconds: initialPredictionElapsedSeconds,
        submitToFinalPredictionSeconds: initialPredictionElapsedSeconds,
        ocrElapsedSeconds: 0,
        semanticElapsedSeconds: null,
        contextualSemanticElapsedSeconds: null,
        sequentialSemanticElapsedSeconds: null,
        retryElapsedSeconds: null
      }
    };
  }

  const image = rasterizeLineCandidate(candidate, {
    padding: rasterPadding,
    targetPixelHeight: initialTargetPixelHeight
  });
  const prediction = await Promise.resolve()
      .then(() => recognizeLine(image, { apiUrl, model, timeoutMs, signal }))
      .catch((error) => recognitionFailureFromError(error, { model }));
  throwIfAborted(signal);
  const initialPredictionElapsedSeconds = secondsSince(pipelineStartedAt);
  const evidenceScore = scoreRecognitionEvidence(
    candidate,
    prediction,
    { problemLatex, previousLatex }
  );

  return {
    candidateId: candidate.candidateId,
    profiles: image.profiles,
    strokeIds: image.strokeIds,
    tightBbox: image.tightBbox,
    image,
    prediction,
    ocrLatex: predictionLatex(prediction),
    latex: prediction?.latex || prediction?.top?.latex || '',
    candidates: prediction?.candidates || [],
    skippedRecognition: Boolean(prediction?.skippedRecognition),
    skipReason: prediction?.skipReason || null,
    initialTargetPixelHeight,
    evidenceScore,
    timing: {
      submitToInitialPredictionSeconds: initialPredictionElapsedSeconds,
      submitToFinalPredictionSeconds: initialPredictionElapsedSeconds,
      ocrElapsedSeconds: finiteSeconds(prediction?.elapsedSeconds),
      semanticElapsedSeconds: null,
      contextualSemanticElapsedSeconds: null,
      sequentialSemanticElapsedSeconds: null,
      retryElapsedSeconds: null
    }
  };
}

async function resolveContextualCandidateSemanticScores({
  candidatePredictions,
  problemLatex,
  problemMetadata,
  previousLatex,
  semanticScoring,
  scoreSemantics,
  apiUrl,
  timeoutMs,
  semanticCandidateLimit,
  signal
}) {
  const scorablePredictions = (candidatePredictions || []).filter(isSemanticallyScorablePrediction);
  if (!semanticScoring || scorablePredictions.length === 0) {
    return {
      source: semanticScoring ? 'empty' : 'disabled',
      failed: false,
      candidateScores: []
    };
  }

  const candidateScores = [];

  try {
    const sameAnswerContextByCandidateId = new Map();
    const contextualGroups = [];
    for (const entry of scorablePredictions) {
      const sameAnswerContext = priorLineContextLatex(entry, scorablePredictions);
      if (sameAnswerContext.length === 0) continue;
      sameAnswerContextByCandidateId.set(entry.candidateId, sameAnswerContext);
      contextualGroups.push({
        candidateId: entry.candidateId,
        latex: entry.latex,
        previousLatex: [
          ...(previousLatex || []),
          ...sameAnswerContext
        ].filter(Boolean),
        candidates: semanticCandidateAlternatives(entry.candidates, semanticCandidateLimit),
        elapsedSeconds: entry.prediction?.elapsedSeconds
      });
    }

    if (contextualGroups.length === 0) {
      return {
        source: 'empty',
        failed: false,
        elapsedSeconds: 0,
        candidateScores: []
      };
    }

    const payload = await scoreSemantics({
      problemLatex,
      problemMetadata,
      previousLatex,
      candidateGroups: contextualGroups
    }, { apiUrl, timeoutMs, signal });
    throwIfAborted(signal);

    for (const score of payload?.candidateScores || []) {
      if (score) {
        candidateScores.push({
          ...score,
          answerManifest: payload?.answerManifest || score.answerManifest || null,
          sameAnswerContext: sameAnswerContextByCandidateId.get(score.candidateId) || []
        });
      }
    }

    return {
      source: 'semantic-service',
      failed: false,
      elapsedSeconds: Number(payload?.elapsedSeconds) || 0,
      answerManifest: candidateScores.find((entry) => entry.answerManifest)?.answerManifest || null,
      candidateScores
    };
  } catch (error) {
    throwIfAborted(signal);
    return {
      source: 'semantic-service',
      failed: true,
      error: error instanceof Error ? error.message : String(error),
      candidateScores
    };
  }
}

async function resolveSelectedLineSemanticScores({
  recognizedLines,
  problemLatex,
  problemMetadata,
  previousLatex,
  semanticScoring,
  scoreSemantics,
  apiUrl,
  timeoutMs,
  semanticCandidateLimit,
  signal
}) {
  const scorableLines = (recognizedLines || []).filter(isSemanticallyScorablePrediction);
  if (!semanticScoring || scorableLines.length === 0) {
    return {
      source: semanticScoring ? 'empty' : 'disabled',
      failed: false,
      lineScores: []
    };
  }

  const contextLatex = [...(previousLatex || [])].filter(Boolean);
  const lineScores = [];
  let elapsedSeconds = 0;
  let cachedManifest = null;

  try {
    for (const line of scorableLines) {
      const payload = await scoreSemantics({
        problemLatex,
        problemMetadata,
        previousLatex: contextLatex.slice(),
        answerManifest: cachedManifest,
        candidateGroups: [{
          candidateId: line.candidateId,
          lineIndex: line.lineIndex,
          latex: line.latex,
          candidates: semanticCandidateAlternatives(line.candidates, semanticCandidateLimit),
          elapsedSeconds: line.prediction?.elapsedSeconds
        }]
      }, { apiUrl, timeoutMs, signal });
      throwIfAborted(signal);

      elapsedSeconds += Number(payload?.elapsedSeconds) || 0;
      if (!cachedManifest && payload?.answerManifest) {
        cachedManifest = payload.answerManifest;
      }
      const score = (payload?.candidateScores || [])[0];
      if (score) {
        const lineScore = {
          ...score,
          answerManifest: payload?.answerManifest || score.answerManifest || cachedManifest,
          lineIndex: line.lineIndex,
          candidateId: line.candidateId
        };
        lineScores.push(lineScore);
        const trustedLatex = gradingSelectedLatex(line, lineScore) ||
          (semanticLatexSafeForReplacement(line, lineScore) && shouldUseSemanticLatex(line.latex, lineScore) ? lineScore.bestLatex : line.latex);
        if (trustedLatex) {
          contextLatex.push(trustedLatex);
          continue;
        }
      }
      if (line.latex) contextLatex.push(line.latex);
    }

    return {
      source: 'semantic-service',
      failed: false,
      elapsedSeconds,
      answerManifest: lineScores.find((entry) => entry.answerManifest)?.answerManifest || null,
      lineScores
    };
  } catch (error) {
    throwIfAborted(signal);
    return {
      source: 'semantic-service',
      failed: true,
      error: error instanceof Error ? error.message : String(error),
      lineScores
    };
  }
}

async function resolveSemanticScores({
  candidatePredictions,
  problemLatex,
  problemMetadata,
  previousLatex,
  semanticScoring,
  scoreSemantics,
  apiUrl,
  timeoutMs,
  semanticCandidateLimit,
  signal
}) {
  const scorablePredictions = (candidatePredictions || []).filter(isSemanticallyScorablePrediction);
  if (!semanticScoring || scorablePredictions.length === 0) {
    return {
      source: semanticScoring ? 'empty' : 'disabled',
      failed: false,
      candidateScores: []
    };
  }

  try {
    const payload = await scoreSemantics({
      problemLatex,
      problemMetadata,
      previousLatex,
      candidateGroups: scorablePredictions.map((entry) => ({
        candidateId: entry.candidateId,
        latex: entry.latex,
        candidates: semanticCandidateAlternatives(entry.candidates, semanticCandidateLimit),
        elapsedSeconds: entry.prediction?.elapsedSeconds
      }))
    }, { apiUrl, timeoutMs, signal });
    throwIfAborted(signal);
    return {
      source: 'semantic-service',
      failed: Boolean(payload?.failed),
      error: payload?.error || null,
      elapsedSeconds: payload?.elapsedSeconds ?? null,
      answerManifest: payload?.answerManifest || null,
      candidateScores: payload?.candidateScores || []
    };
  } catch (error) {
    throwIfAborted(signal);
    return {
      source: 'semantic-service',
      failed: true,
      error: error instanceof Error ? error.message : String(error),
      candidateScores: []
    };
  }
}

async function resolveDetections({
  strokes,
  answerBox,
  detections,
  detectLineBands,
  detectLines,
  apiUrl,
  timeoutMs,
  signal
}) {
  if (detections && detections.length > 0) {
    return {
      detections,
      source: 'provided',
      failed: false
    };
  }

  if (!detectLineBands) {
    return {
      detections: [],
      source: 'disabled',
      failed: false
    };
  }

  const candidate = answerCandidateFromStrokes(strokes, answerBox);
  if (!candidate) {
    return {
      detections: [],
      source: 'empty',
      failed: false
    };
  }

  try {
    const image = rasterizeLineCandidate(candidate, {
      padding: answerBox ? 0 : 12,
      devicePixelRatio: 1
    });
    const result = await detectLines(image, { apiUrl, timeoutMs, signal });
    throwIfAborted(signal);
    return {
      detections: result?.detections || [],
      source: 'detector',
      failed: Boolean(result?.failed),
      error: result?.error || null,
      elapsedSeconds: result?.elapsedSeconds ?? null,
      rawDetections: result?.rawDetections || []
    };
  } catch (error) {
    throwIfAborted(signal);
    return {
      detections: [],
      source: 'detector',
      failed: true,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

function answerCandidateFromStrokes(strokes, answerBox) {
  const eligible = (strokes || []).filter((stroke) => {
    if (!stroke?.canvasBbox) return false;
    return strokeBelongsToAnswerBox(stroke, answerBox);
  });
  if (eligible.length === 0) return null;

  const tightBbox = answerBox || computeTightBbox(eligible);
  return {
    candidateId: 'answer-detector',
    profiles: ['answer-detector'],
    strokeIds: eligible.map((stroke) => String(stroke.id)),
    strokes: eligible,
    tightBbox
  };
}

async function retrySelectedLineRecognition(candidate, {
  apiUrl,
  model,
  timeoutMs,
  problemLatex = '',
  rasterPadding,
  initialRasterHeight,
  retryRasterHeights,
  skipRasterHeights = [],
  extendedTimeoutMs = 0,
  chunkFallback = true,
  chunkFallbackMinCssWidth = 460,
  chunkFallbackMaxCssWidth = 340,
  chunkFallbackMinGap = 18,
  finalizationBudget = null,
  signal,
  recognizeLine
}) {
  const attempts = [];
  let sawTimeout = false;
  const seenHeights = new Set((skipRasterHeights || [])
    .map((height) => Number(height))
    .filter((height) => Number.isFinite(height) && height > 0));
  for (const height of retryRasterHeights || []) {
    throwIfAborted(signal);
    if (finalizationBudget?.check?.()) break;
    if (sawTimeout) break;
    const targetPixelHeight = Number(height);
    if (!Number.isFinite(targetPixelHeight) || targetPixelHeight <= 0 || seenHeights.has(targetPixelHeight)) {
      continue;
    }
    seenHeights.add(targetPixelHeight);
    const image = rasterizeLineCandidate(candidate, {
      padding: rasterPadding,
      targetPixelHeight
    });
    const prediction = await Promise.resolve()
      .then(() => recognizeLine(image, { apiUrl, model, timeoutMs, signal }))
      .catch((error) => recognitionFailureFromError(error, {
        model,
        retryTargetPixelHeight: targetPixelHeight
      }));
    throwIfAborted(signal);
    attempts.push({
      ...prediction,
      retryTargetPixelHeight: targetPixelHeight
    });
    if (prediction?.timedOut) sawTimeout = true;
  }
  if (
    attempts.length &&
    (!sawTimeout || candidateNeedsExtendedTimeout(candidate)) &&
    attempts.every((attempt) => predictionNeedsRetry(attempt)) &&
    candidateNeedsExtendedTimeout(candidate) &&
    Number(extendedTimeoutMs) > Number(timeoutMs) &&
    !finalizationBudget?.check?.()
  ) {
    throwIfAborted(signal);
    const targetPixelHeight = [...seenHeights][0] || Number(retryRasterHeights?.[0]) || undefined;
    const image = rasterizeLineCandidate(candidate, {
      padding: rasterPadding,
      targetPixelHeight
    });
    const prediction = await Promise.resolve()
      .then(() => recognizeLine(image, { apiUrl, model, timeoutMs: Number(extendedTimeoutMs), signal }))
      .catch((error) => recognitionFailureFromError(error, {
        model,
        retryTargetPixelHeight: targetPixelHeight,
        extendedTimeoutMs: Number(extendedTimeoutMs)
      }));
    throwIfAborted(signal);
    attempts.push({
      ...prediction,
      retryTargetPixelHeight: targetPixelHeight,
      extendedTimeoutMs: Number(extendedTimeoutMs)
    });
  }
  if (
    chunkFallback &&
    !sawTimeout &&
    attempts.length &&
    attempts.every((attempt) => predictionNeedsRetry(attempt)) &&
    !finalizationBudget?.check?.()
  ) {
    const chunked = await recognizeChunkedLine(candidate, {
      apiUrl,
      model,
      timeoutMs,
      problemLatex,
      rasterPadding,
      initialRasterHeight,
      minCssWidth: chunkFallbackMinCssWidth,
      maxChunkCssWidth: chunkFallbackMaxCssWidth,
      minGap: chunkFallbackMinGap,
      signal,
      recognizeLine
    });
    if (chunked && !predictionNeedsRetry(chunked)) {
      attempts.push({
        ...chunked,
        semanticRetryChunkFallback: true
      });
    }
  }
  return { attempts };
}

function initialTargetPixelHeightForCandidate(candidate, {
  rasterPadding,
  initialRasterHeight,
  initialRasterMinCssHeight
}) {
  const targetPixelHeight = Number(initialRasterHeight);
  if (!Number.isFinite(targetPixelHeight) || targetPixelHeight <= 0) return undefined;
  const minCssHeight = Number(initialRasterMinCssHeight);
  if (!Number.isFinite(minCssHeight) || minCssHeight <= 0) return undefined;
  const box = candidate?.tightBbox;
  if (!box) return undefined;
  const cssHeight = Math.max(1, Math.ceil(box.yMax) - Math.floor(box.yMin) + rasterPadding * 2);
  return cssHeight >= minCssHeight ? targetPixelHeight : undefined;
}

function predictionNeedsRetry(prediction = {}) {
  if (!prediction || prediction.failed || prediction.timedOut) return true;
  const latex = predictionLatex(prediction);
  return !latex || isSuspiciousOperationLatex(latex);
}

function candidateNeedsExtendedTimeout(candidate) {
  if (!candidate) return false;
  const profiles = new Set(candidate.profiles || []);
  if (profiles.has('fraction-stack-line')) return true;
  const strokes = (candidate.strokes || []).filter((stroke) => stroke?.canvasBbox);
  const box = candidate.bbox || candidate.tightBbox || (strokes.length ? computeTightBbox(strokes) : null);
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  if (strokes.length >= 10 && height >= 80) return true;
  if (strokes.length >= 8 && width >= 240 && height >= 90) return true;
  return false;
}

function candidateHasLocalFractionStructure(candidate) {
  const strokes = (candidate?.strokes || []).filter((stroke) => stroke?.canvasBbox);
  return findLocalFractionGroups(strokes).length > 0;
}

function predictionLatex(prediction = {}) {
  const candidates = prediction?.candidates || [];
  return String(prediction?.latex || prediction?.top?.latex || candidates[0]?.latex || '').trim();
}

function skippedInitialRecognitionPrediction(model, skipReason) {
  return {
    model,
    latex: '',
    top: null,
    candidates: [],
    confidence: 0,
    failed: false,
    timedOut: false,
    skippedRecognition: true,
    skipReason,
    elapsedSeconds: 0
  };
}

async function recognizeSkippedSelectedEntry(entry, candidate, {
  apiUrl,
  model,
  timeoutMs,
  problemLatex,
  previousLatex,
  pipelineStartedAt,
  rasterPadding,
  recognizeLine,
  signal,
  evidenceByCandidateId
}) {
  if (!entry.image) {
    entry.image = rasterizeLineCandidate(candidate, {
      padding: rasterPadding,
      targetPixelHeight: entry.initialTargetPixelHeight
    });
  }
  const prediction = await Promise.resolve()
    .then(() => recognizeLine(entry.image, { apiUrl, model, timeoutMs, signal }))
    .catch((error) => recognitionFailureFromError(error, { model }));
  throwIfAborted(signal);

  entry.prediction = prediction;
  entry.candidates = prediction?.candidates || [];
  entry.ocrLatex = predictionLatex(prediction);
  entry.latex = prediction?.latex || prediction?.top?.latex || '';
  entry.skippedRecognition = false;
  entry.skipReason = null;
  entry.deferredRecognition = true;
  entry.evidenceScore = scoreRecognitionEvidence(candidate, prediction, { problemLatex, previousLatex });
  entry.timing = {
    ...(entry.timing || {}),
    submitToInitialPredictionSeconds: secondsSince(pipelineStartedAt),
    submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt),
    ocrElapsedSeconds: finiteSeconds(prediction?.elapsedSeconds)
  };
  evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
}

async function recognizeDeferredAlternativesInsideWeakSelection(selected, {
  candidates,
  candidatePredictions,
  evidenceByCandidateId,
  apiUrl,
  model,
  timeoutMs,
  problemLatex,
  previousLatex,
  pipelineStartedAt,
  rasterPadding,
  recognizeLine,
  signal
}) {
  const selectedEntries = (selected || [])
    .map((candidate) => ({
      candidate,
      entry: candidatePredictions.find((item) => item.candidateId === candidate.candidateId)
    }))
    .filter(({ entry }) => selectedEntryNeedsDeferredAlternatives(entry));
  if (selectedEntries.length === 0) return;

  for (const deferredEntry of candidatePredictions) {
    throwIfAborted(signal);
    if (!deferredEntry?.skippedRecognition) continue;
    if (deferredEntry.skipReason !== 'contained-nonstructural-alternative') continue;
    const deferredCandidate = (candidates || []).find((candidate) => (
      candidate.candidateId === deferredEntry.candidateId
    ));
    if (!deferredCandidate || !deferredCandidate.strokeIds?.length) continue;
    const containedByWeakSelection = selectedEntries.some(({ candidate }) => (
      candidate.candidateId !== deferredCandidate.candidateId &&
      strokeSetStrictlyContainsIds(candidate.strokeIds, deferredCandidate.strokeIds)
    ));
    if (!containedByWeakSelection) continue;

    await recognizeSkippedSelectedEntry(deferredEntry, deferredCandidate, {
      apiUrl,
      model,
      timeoutMs,
      problemLatex,
      previousLatex,
      pipelineStartedAt,
      rasterPadding,
      recognizeLine,
      signal,
      evidenceByCandidateId
    });
  }
}

function selectedEntryNeedsDeferredAlternatives(entry) {
  if (!entry || entry.skippedRecognition) return false;
  if (predictionNeedsRetry(entry.prediction)) return true;
  return Number(entry.evidenceScore) <= -35;
}

function initialRecognitionSkipReason(candidate, context = {}) {
  if (!context.enabled || !candidate) return '';
  if (context.coveredByValidDeterministicLineCandidateIds?.has(candidate.candidateId)) {
    return 'covered-by-valid-deterministic-line';
  }
  if (shouldSkipSingleStrokeAlternative(candidate, context)) return 'single-stroke-alternative';
  if (shouldDeferContainedNonstructuralAlternative(candidate, context)) {
    return 'contained-nonstructural-alternative';
  }
  return '';
}

function deterministicPredictionIsValid(entry) {
  if (!entry || entry.skippedRecognition || entry.prediction?.skippedRecognition) return false;
  if (predictionNeedsRetry(entry.prediction)) return false;
  if (!predictionLatex(entry.prediction)) return false;
  return Number(entry.evidenceScore) > -20;
}

function coveredLargerCandidateIds(candidates, validLines) {
  const covered = new Set();
  const lines = (validLines || []).filter((candidate) => candidate?.strokeIds?.length);
  if (lines.length === 0) return covered;

  for (const candidate of candidates || []) {
    const candidateIds = uniqueStrings(candidate?.strokeIds || []);
    if (candidateIds.length <= 1) continue;
    if (!isCoveredRecognitionDeferrableCandidate(candidate)) continue;

    const coveringLines = lines.filter((line) => (
      line.candidateId !== candidate.candidateId &&
      strokeSetStrictlyContainsIds(candidateIds, line.strokeIds)
    ));
    if (coveringLines.length === 0) continue;

    const coveredIds = new Set(coveringLines.flatMap((line) => line.strokeIds || []).map(String));
    if (candidateIds.every((id) => coveredIds.has(id))) {
      covered.add(candidate.candidateId);
    }
  }

  return covered;
}

function orderInitialRecognitionCandidates(candidates, baselineIds = new Set()) {
  return (candidates || []).slice().sort((a, b) => (
    initialRecognitionPriority(a, baselineIds) - initialRecognitionPriority(b, baselineIds) ||
    (a.tightBbox?.yMin ?? 0) - (b.tightBbox?.yMin ?? 0) ||
    (a.tightBbox?.xMin ?? 0) - (b.tightBbox?.xMin ?? 0) ||
    String(a.candidateId).localeCompare(String(b.candidateId))
  ));
}

function initialRecognitionPriority(candidate, baselineIds = new Set()) {
  if (baselineIds.has(candidate?.candidateId)) return 0;
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fraction-stack-line')) return 1;
  if (profiles.includes('superscript-line')) return 1;
  if (
    profiles.includes('row-line') ||
    profiles.includes('raw-row-line') ||
    profiles.includes('dbnet-line') ||
    profiles.includes('strict')
  ) {
    return 2;
  }
  if (profiles.includes('temporal') || profiles.includes('loose') || profiles.includes('projection-line')) {
    return 3;
  }
  if (isParentLikeRecognitionCandidate(candidate)) return 4;
  return 5;
}

function isParentLikeRecognitionCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  return profiles.includes('parent') ||
    profiles.includes('row-parent') ||
    profiles.includes('dbnet-parent') ||
    profiles.includes('projection-line') ||
    profiles.includes('temporal') ||
    profiles.includes('loose');
}

function isCoveredRecognitionDeferrableCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  if (profiles.includes('fraction-stack-line')) return false;
  if (profiles.includes('superscript-line')) return false;
  return profiles.some((profile) => (
    profile === 'parent' ||
    profile === 'row-parent' ||
    profile === 'dbnet-parent' ||
    profile === 'projection-line' ||
    profile === 'temporal' ||
    profile === 'loose' ||
    profile === 'strict' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'dbnet-line'
  ));
}

function shouldSkipSingleStrokeAlternative(candidate, context = {}) {
  if (!context.skipSingleStrokeAlternatives || !candidate) return false;
  if (context.baselineCandidateIds?.has(candidate.candidateId)) return false;
  const strokeIds = uniqueStrings(candidate.strokeIds || []);
  if (strokeIds.length !== 1) return false;
  const strokeId = strokeIds[0];
  return (context.candidates || []).some((other) => {
    if (!other || other === candidate) return false;
    const otherIds = uniqueStrings(other.strokeIds || []);
    return otherIds.length > 1 && otherIds.includes(strokeId);
  });
}

function shouldDeferContainedNonstructuralAlternative(candidate, context = {}) {
  if (!context.stagedAlternativeRecognition || !candidate) return false;
  if (context.baselineCandidateIds?.has(candidate.candidateId)) return false;
  const strokeIds = uniqueStrings(candidate.strokeIds || []);
  if (strokeIds.length <= 1) return false;
  const profiles = candidate.profiles || [];
  if (!profiles.some((profile) => profile === 'strict' || profile === 'loose')) return false;
  if (isImmediateRecognitionStructuralCandidate(candidate)) return false;

  return (context.candidates || []).some((other) => {
    if (!other || other === candidate) return false;
    if (!isImmediateRecognitionStructuralCandidate(other)) return false;
    return strokeSetStrictlyContainsIds(other.strokeIds, strokeIds);
  });
}

function isImmediateRecognitionStructuralCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  return profiles.some((profile) => (
    profile === 'parent' ||
    profile === 'temporal' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'projection-line' ||
    profile === 'dbnet-parent' ||
    profile === 'dbnet-line'
  ));
}

function strokeSetStrictlyContainsIds(containerIds, childIds) {
  const container = new Set((containerIds || []).map(String));
  const child = uniqueStrings(childIds || []);
  return container.size > child.length && child.every((id) => container.has(id));
}

function isSemanticallyScorablePrediction(entry) {
  if (!entry || entry.skippedRecognition || entry.prediction?.skippedRecognition) return false;
  if (entry.prediction?.failed || entry.prediction?.timedOut) return false;
  if (entry.latex) return true;
  return (entry.candidates || []).some((candidate) => String(candidate?.latex || '').trim());
}

function semanticCandidateAlternatives(candidates = [], limit = 5) {
  const items = Array.isArray(candidates) ? candidates : [];
  const max = Number(limit);
  if (!Number.isFinite(max) || max <= 0) return items;
  return items.slice(0, Math.floor(max));
}

function applyFinalCandidateDebugState(candidatePredictions, recognizedLines, pipelineStartedAt) {
  const selectedById = new Map((recognizedLines || []).map((line) => [line.candidateId, line]));
  for (const [index, entry] of (candidatePredictions || []).entries()) {
    const selectedLine = selectedById.get(entry.candidateId);
    entry.debugLabel = `C${index + 1}`;
    entry.selected = Boolean(selectedLine);
    entry.discarded = !selectedLine;
    entry.selectedLineIndex = selectedLine?.lineIndex ?? null;
    entry.acceptedLatex = selectedLine?.acceptedLatex || null;
    entry.timing = {
      ...(entry.timing || {}),
      submitToFinalPredictionSeconds: entry.timing?.submitToFinalPredictionSeconds ?? secondsSince(pipelineStartedAt)
    };
    if (!selectedLine) continue;

    entry.prediction = selectedLine.prediction;
    entry.candidates = selectedLine.candidates || entry.candidates || [];
    entry.ocrLatex = selectedLine.ocrLatex || entry.ocrLatex || '';
    entry.latex = selectedLine.latex || entry.latex || '';
    entry.evidenceScore = selectedLine.evidenceScore;
    entry.retryPredictions = selectedLine.retryPredictions || entry.retryPredictions || [];
    entry.semanticRetryPredictions = selectedLine.semanticRetryPredictions || entry.semanticRetryPredictions || [];
    entry.sequentialSemantic = selectedLine.sequentialSemantic || null;
    entry.grading = selectedLine.grading || entry.grading || null;
    entry.ocrRepair = selectedLine.ocrRepair || null;
    entry.timing = {
      ...(entry.timing || {}),
      ...(selectedLine.timing || {}),
      submitToFinalPredictionSeconds: selectedLine.timing?.submitToFinalPredictionSeconds ?? secondsSince(pipelineStartedAt)
    };
    selectedLine.debugLabel = entry.debugLabel;
  }
}

function preferFullSolutionCandidates(selected = [], candidatePredictions = [], candidates = []) {
  const selectedIds = new Set((selected || []).map((candidate) => candidate.candidateId));
  const candidateById = new Map((candidates || []).map((candidate) => [candidate.candidateId, candidate]));
  return (selected || []).map((candidate) => {
    const selectedEntry = candidatePredictions.find((entry) => entry.candidateId === candidate.candidateId);
    if (solutionCoverageRank(selectedEntry) >= 2) return candidate;

    const selectedBox = candidate.tightBbox || selectedEntry?.tightBbox;
    const replacement = (candidatePredictions || [])
      .filter((entry) => entry?.candidateId && !selectedIds.has(entry.candidateId))
      .filter((entry) => solutionCoverageRank(entry) >= 2)
      .filter((entry) => bboxOverlapRatio(selectedBox, entry.tightBbox) >= 0.55)
      .sort((a, b) => (
        solutionCoverageRank(b) - solutionCoverageRank(a) ||
        (Number(b.evidenceScore) || 0) - (Number(a.evidenceScore) || 0)
      ))[0];

    if (!replacement) return candidate;
    selectedIds.delete(candidate.candidateId);
    selectedIds.add(replacement.candidateId);
    return candidateById.get(replacement.candidateId) || {
      ...candidate,
      candidateId: replacement.candidateId,
      strokeIds: replacement.strokeIds || candidate.strokeIds,
      tightBbox: replacement.tightBbox || candidate.tightBbox,
    };
  });
}

function solutionCoverageRank(entry = {}) {
  const grading = entry?.grading || entry?.semantic?.grading || entry?.contextualSemantic?.grading || entry?.sequentialSemantic?.grading || null;
  if (gradingPreferred(grading) && !candidateSelectionSafeForGrading(entry, grading)) return 0;
  if (grading?.solutionCoverage === 'full') return 2;
  if (grading?.solutionCoverage === 'partial' || (grading?.matchedSolutions || []).length > 0) return 1;
  return 0;
}

function applyGeometryOperationAnnotationRepairs(lines = [], problemLatex = '') {
  const equationLines = lines.filter((line) => (
    line?.tightBbox &&
    /[=]/.test(String(line.latex || '')) &&
    /\\frac\b/.test(String(line.latex || problemLatex || ''))
  ));
  if (!equationLines.length) return;

  for (const line of lines) {
    if (!line?.tightBbox || line.ocrRepair) continue;
    const operand = detachedOperationOperand(line.latex || line.ocrLatex || line.acceptedLatex || '');
    if (!operand) continue;
    const anchor = nearestOperationAnchor(line, equationLines);
    if (!anchor) continue;
    const repairedLatex = `\\times ${operand} \\times ${operand}`;
    line.ocrRepair = {
      source: 'geometry-operation-annotation',
      originalLatex: line.latex,
      repairedLatex,
      operand,
      anchorCandidateId: anchor.candidateId || null,
      anchorBbox: anchor.tightBbox || null,
      annotationBbox: line.tightBbox || null,
    };
    line.latex = repairedLatex;
    line.excludedFromGrading = true;
  }
}

function nearestOperationAnchor(line, equationLines) {
  const lineBox = line.tightBbox;
  const lineCenterY = bboxYCenter(lineBox);
  return equationLines
    .map((anchor) => {
      const anchorBox = anchor.tightBbox;
      const verticalGap = Math.max(0, Math.max(anchorBox.yMin - lineBox.yMax, lineBox.yMin - anchorBox.yMax));
      const verticalDistance = Math.abs(lineCenterY - bboxYCenter(anchorBox));
      const horizontallyRelevant = lineBox.xMax >= anchorBox.xMin - 120 && lineBox.xMin <= anchorBox.xMax + 120;
      const closeEnough = verticalGap <= Math.max(90, bboxHeight(anchorBox) * 0.9) ||
        verticalDistance <= Math.max(120, bboxHeight(anchorBox) * 1.2);
      return {
        anchor,
        score: verticalGap + verticalDistance * 0.25,
        valid: horizontallyRelevant && closeEnough,
      };
    })
    .filter((item) => item.valid)
    .sort((a, b) => a.score - b.score)[0]?.anchor || null;
}

/**
 * Post-OCR merge pass: coalesce adjacent selected lines when both have low
 * individual OCR confidence and no structural fraction boundary between them,
 * or when same-row fragments form a plausible equation in left-to-right order.
 * This prevents over-segmentation where a single equation is split into
 * multiple weak lines or symbol fragments.
 */
function postOcrMergePass(lines, {
  candidatePredictions = [],
  candidatesToRecognize = [],
  evidenceByCandidateId = new Map(),
  problemLatex = '',
  previousLatex = []
} = {}) {
  if (!lines || lines.length < 2) return lines;

  const merged = [];
  let i = 0;
  while (i < lines.length) {
    const current = lines[i];
    const next = lines[i + 1];
    if (!next) {
      merged.push(current);
      i += 1;
      continue;
    }

    const mergeMode = adjacentLineMergeMode(current, next, { problemLatex, previousLatex });
    if (mergeMode) {
      const combined = mergeLineEntries(current, next, {
        candidatePredictions,
        candidatesToRecognize,
        evidenceByCandidateId,
        mergeMode,
        problemLatex,
        previousLatex
      });
      merged.push(combined);
      i += 2;
    } else {
      merged.push(current);
      i += 1;
    }
  }

  return merged.map((line, index) => ({ ...line, lineIndex: index }));
}

function adjacentLineMergeMode(upper, lower, { problemLatex = '', previousLatex = [] } = {}) {
  if (!upper || !lower) return false;
  if (upper.skippedRecognition || lower.skippedRecognition) return false;
  if (shouldMergeInlineEquationFragments(upper, lower)) return 'inline';

  const upperEvidence = Number(upper.evidenceScore);
  const lowerEvidence = Number(lower.evidenceScore);
  if (Number.isFinite(upperEvidence) && upperEvidence > -50) return false;
  if (Number.isFinite(lowerEvidence) && lowerEvidence > -50) return false;

  const upperBox = upper.tightBbox;
  const lowerBox = lower.tightBbox;
  if (!upperBox || !lowerBox) return false;

  const gap = Math.max(0, lowerBox.yMin - upperBox.yMax);
  const upperHeight = Math.max(1, upperBox.yMax - upperBox.yMin);
  if (gap > upperHeight * 0.8) return false;

  const overlap = horizontalOverlapRatio(upperBox, lowerBox);
  if (overlap < 0.2) return false;

  if (hasFractionBoundaryBetween(upper, lower)) return false;

  const combinedLatex = `${upper.latex || ''} ${lower.latex || ''}`.trim();
  if (!combinedLatex) return false;
  if (!/[=]/.test(combinedLatex) && !/\\frac/.test(combinedLatex)) return false;

  return 'vertical';
}

function shouldMergeInlineEquationFragments(first, second) {
  const firstBox = first?.tightBbox;
  const secondBox = second?.tightBbox;
  if (!firstBox || !secondBox) return false;

  const verticalOverlap = bboxVerticalOverlapRatio(firstBox, secondBox);
  const centerDistance = Math.abs(bboxYCenter(firstBox) - bboxYCenter(secondBox));
  const maxHeight = Math.max(1, bboxHeight(firstBox), bboxHeight(secondBox));
  if (verticalOverlap < 0.25 && centerDistance > maxHeight * 0.45) return false;

  const gap = bboxHorizontalGap(firstBox, secondBox);
  if (gap > Math.max(80, maxHeight * 1.1)) return false;

  const ordered = orderLinesForInlineMerge(first, second);
  const parts = ordered.map((line) => normalizedLineLatex(line));
  if (!parts[0] || !parts[1]) return false;
  if (!parts.some(looksLikeEquationFragment)) return false;

  return looksLikeInlineEquation(parts.join(' '));
}

function hasFractionBoundaryBetween(upper, lower) {
  const upperStrokes = upper.strokes || [];
  const lowerStrokes = lower.strokes || [];
  const allStrokes = [...upperStrokes, ...lowerStrokes];
  const bars = allStrokes.filter((stroke) => {
    const box = stroke?.canvasBbox;
    if (!box) return false;
    const width = Math.max(0, box.xMax - box.xMin);
    const height = Math.max(1, box.yMax - box.yMin);
    return width >= 18 && height <= 12 && width >= height * 3.5;
  });
  if (bars.length === 0) return false;

  const upperBox = upper.tightBbox;
  const lowerBox = lower.tightBbox;
  for (const bar of bars) {
    const barBox = bar.canvasBbox;
    const barY = (barBox.yMin + barBox.yMax) / 2;
    if (barY >= upperBox.yMin && barY <= lowerBox.yMax) return true;
  }
  return false;
}

function mergeLineEntries(upper, lower, {
  candidatePredictions = [],
  candidatesToRecognize = [],
  evidenceByCandidateId = new Map(),
  mergeMode = 'vertical',
  problemLatex = '',
  previousLatex = []
}) {
  const orderedLines = mergeMode === 'inline' ? orderLinesForInlineMerge(upper, lower) : [upper, lower];
  const combinedStrokes = orderedLines.flatMap((line) => line.strokes || []);
  const combinedStrokeIds = orderedLines.flatMap((line) => line.strokeIds || []);
  const combinedTightBbox = bboxForStrokes(combinedStrokes);
  const combinedCandidates = orderedLines.flatMap((line) => line.candidates || []);
  const mergedPrediction = mergePredictionAttempts(orderedLines.map((line) => line.prediction));
  const combinedLatex = orderedLines.map((line) => line.latex).filter(Boolean).join(' ').trim();

  return {
    ...upper,
    candidateId: `${upper.candidateId}+${lower.candidateId}`,
    strokeIds: combinedStrokeIds,
    strokes: combinedStrokes,
    tightBbox: combinedTightBbox,
    prediction: mergedPrediction,
    candidates: combinedCandidates,
    ocrLatex: orderedLines.map((line) => line.ocrLatex).filter(Boolean).join(' ').trim(),
    latex: combinedLatex,
    evidenceScore: Number(upper.evidenceScore) + Number(lower.evidenceScore),
    mergedFrom: [upper.candidateId, lower.candidateId],
    timing: {
      ...(upper.timing || {}),
      ocrElapsedSeconds: finiteSeconds(
        (Number(upper.timing?.ocrElapsedSeconds) || 0) +
        (Number(lower.timing?.ocrElapsedSeconds) || 0)
      ),
      submitToFinalPredictionSeconds: upper.timing?.submitToFinalPredictionSeconds || 0
    }
  };
}

function orderLinesForInlineMerge(first, second) {
  return [first, second].slice().sort((a, b) => (
    (a?.tightBbox?.xMin ?? 0) - (b?.tightBbox?.xMin ?? 0) ||
    (a?.tightBbox?.yMin ?? 0) - (b?.tightBbox?.yMin ?? 0)
  ));
}

function normalizedLineLatex(line = {}) {
  return String(line.latex || line.ocrLatex || line.acceptedLatex || '')
    .replace(/\s+/g, ' ')
    .trim();
}

function looksLikeEquationFragment(latex = '') {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return false;
  if (/^(?:=|[+\-*/]|\\(?:times|cdot|div)\b)/.test(normalized)) return true;
  if (/(?:=|[+\-*/]|\\times|\\cdot|\\div)\s*$/.test(normalized)) return true;
  return /^[a-zA-Z](?:\s*(?:_\s*\{?\w+\}?|\^\s*\{?\w+\}?))?$/.test(normalized);
}

function looksLikeInlineEquation(latex = '') {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || (normalized.match(/=/g) || []).length !== 1) return false;
  const [left, right] = normalized.split('=').map((part) => part.trim());
  if (!left || !right) return false;
  if (!/[a-zA-Z0-9\\)]/.test(left)) return false;
  if (!/[a-zA-Z0-9\\(]/.test(right)) return false;
  return true;
}

function finiteSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? Number(seconds.toFixed(3)) : null;
}

function sumPredictionElapsedSeconds(predictions = []) {
  const total = (predictions || []).reduce((sum, prediction) => {
    const elapsed = Number(prediction?.elapsedSeconds);
    return Number.isFinite(elapsed) ? sum + elapsed : sum;
  }, 0);
  return total > 0 ? finiteSeconds(total) : null;
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const values = Array.isArray(items) ? items : [];
  if (values.length === 0) return [];

  const limit = Math.max(1, Math.min(
    values.length,
    Number.isFinite(Number(concurrency)) ? Math.floor(Number(concurrency)) : 1
  ));
  const results = new Array(values.length);
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(values[index], index);
    }
  }

  await Promise.all(Array.from({ length: limit }, () => worker()));
  return results;
}

function secondsSince(startedAt) {
  return finiteSeconds((performanceNow() - startedAt) / 1000) || 0;
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function createFinalizationBudget(startedAt, budgetMs) {
  const limitMs = Number(budgetMs);
  let exceeded = false;
  return {
    check() {
      if (!Number.isFinite(limitMs) || limitMs < 0) return false;
      if (performanceNow() - startedAt >= limitMs) {
        exceeded = true;
        return true;
      }
      return false;
    },
    get exceeded() {
      return exceeded;
    }
  };
}

function semanticNeedsRetry(line, semanticEntry = {}) {
  if (!line?.prediction || predictionNeedsRetry(line.prediction)) return false;
  if (semanticEntry.equivalentToProblem || semanticEntry.equivalentToPrevious) return false;

  const semanticScore = Number(semanticEntry.semanticScore);
  if (Number.isFinite(semanticScore) && semanticScore >= 2) return false;

  const latex = String(line.latex || line.prediction?.latex || line.prediction?.top?.latex || '').trim();
  const weakFunctionEquation = looksLikeWeakFunctionEquation(latex, semanticEntry);
  const weakVariableEquation = looksLikeWeakVariableEquation(latex, semanticEntry);
  if (
    !looksLikeShortNumericEquation(latex) &&
    !looksMalformedForSemanticRetry(latex) &&
    !weakFunctionEquation &&
    !weakVariableEquation
  ) {
    return false;
  }

  const width = bboxWidth(line.tightBbox);
  return !Number.isFinite(width) || width <= (weakFunctionEquation ? 980 : 620);
}

function looksLikeShortNumericEquation(latex) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || !normalized.includes('=') || !/\d/.test(normalized)) return false;
  if (/[a-zA-Z]/.test(normalized.replace(/\\(?:cdot|times|div|pm)/g, ''))) return false;
  const compact = normalized
    .replace(/\\(?:cdot|times|div|pm)/g, '')
    .replace(/[\s{}()[\].,+\-*/=^_]/g, '');
  return /^\d+$/.test(compact) && compact.length <= 8;
}

function looksMalformedForSemanticRetry(latex) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || !/\d/.test(normalized)) return false;

  const withoutCommands = normalized.replace(/\\[a-zA-Z]+/g, '');
  const letters = withoutCommands.match(/[a-zA-Z]/g) || [];
  const hasMathSyntax = /[=+\-*/()]|\\(?:times|div|frac)/.test(normalized);
  if (!hasMathSyntax) return false;

  if (groupingLooksUnbalanced(normalized)) return true;
  if (/\\(?:times|div)\b/.test(normalized) && /^[a-zA-Z]\s+\d/.test(withoutCommands)) return true;
  if (
    letters.length > 0 &&
    letters.every((letter) => /[oln]/i.test(letter)) &&
    /[()+\-]/.test(normalized)
  ) {
    return true;
  }

  return false;
}

function looksLikeWeakFunctionEquation(latex, semanticEntry = {}) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized.includes('=') || !/[A-Za-z]/.test(normalized)) return false;
  if (!/[A-Za-z]\s*(?:\^\s*\{[^}]+\}\s*)?\(/.test(normalized)) return false;

  const semanticScore = Number(semanticEntry.semanticScore);
  if (Number.isFinite(semanticScore) && semanticScore >= 1.2) return false;
  return !(semanticEntry.equivalentToProblem || semanticEntry.equivalentToPrevious);
}

function looksLikeWeakVariableEquation(latex, semanticEntry = {}) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized.includes('=') || !/[A-Za-z]/.test(normalized) || !/\d/.test(normalized)) return false;
  if (/\\(?:frac|sqrt|log|ln|int|sum|prod)\b/.test(normalized)) return false;
  if (/[A-Za-z]\s*(?:\^\s*\{[^}]+\}\s*)?\(/.test(normalized)) return false;
  const withoutCommands = normalized.replace(/\\[a-zA-Z]+/g, '');
  if (!/[+\-]|\b\d+\s*[A-Za-z]\b|\b[A-Za-z]\s*\^\s*\{/.test(withoutCommands)) return false;

  const semanticScore = Number(semanticEntry.semanticScore);
  if (Number.isFinite(semanticScore) && semanticScore >= 2) return false;
  return !(semanticEntry.equivalentToProblem || semanticEntry.equivalentToPrevious);
}

function repairStandaloneOperationLatex(latex, semanticEntry = {}, { problemLatex = '' } = {}) {
  const semanticScore = Number(semanticEntry.semanticScore);
  if (semanticEntry.equivalentToProblem || semanticEntry.equivalentToPrevious) return null;
  if (Number.isFinite(semanticScore) && semanticScore >= 2) return null;

  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || /[=<>]/.test(normalized)) return null;
  if (/\\(?:frac|sqrt|log|ln|int|sum|prod)\b/.test(normalized)) return null;
  if (looksLikePlainNumericLiteral(normalized)) return null;

  const detachedOperand = detachedOperationOperand(normalized, {
    requireOperationMarker: true
  });
  if (detachedOperand) {
    return `\\times ${detachedOperand} \\times ${detachedOperand}`;
  }

  const subscriptedTimes = normalized.match(
    /^(?:x|X|\\times)\s*_\s*\{\s*(-?\d+)\s*\}\s*\\times\s*(?:_|(?:x|X|\\times)\s*_)\s*\{\s*([a-zA-Z0-9-]+)\s*\}$/
  );
  if (subscriptedTimes) {
    const leftOperand = subscriptedTimes[1];
    let rightOperand = subscriptedTimes[2];
    if (!/^-?\d+$/.test(rightOperand)) rightOperand = leftOperand;
    if (leftOperand === rightOperand) return `\\times ${leftOperand} \\times ${rightOperand}`;
  }

  const spacedOperands = normalized.match(/^(\\(?:times|div))\s+((?:\d\s*){1,5})\s+(?:x|X|\\times|\\div)\s+((?:\d\s*){1,5})$/);
  if (spacedOperands) {
    const operator = spacedOperands[1];
    const leftOperand = (spacedOperands[2].match(/\d/g) || []).join('');
    const rightOperand = (spacedOperands[3].match(/\d/g) || []).join('');
    if (leftOperand && leftOperand === rightOperand) return `${operator} ${leftOperand} ${operator} ${rightOperand}`;
  }

  const plainXOperands = normalized.match(/^(?:x|X)\s+((?:\d\s*){1,5})\s+(?:x|X)\s+((?:\d\s*){1,5})$/);
  if (plainXOperands) {
    const leftOperand = (plainXOperands[1].match(/\d/g) || []).join('');
    const rightOperand = (plainXOperands[2].match(/\d/g) || []).join('');
    if (leftOperand && leftOperand === rightOperand) return `\\times ${leftOperand} \\times ${rightOperand}`;
  }

  const multiplier = contextualProblemDenominatorMultiplier(problemLatex);
  if (multiplier && looksLikeMalformedEqualMultiplierOperation(normalized)) {
    return `\\times ${multiplier} \\times ${multiplier}`;
  }

  const withoutCommands = normalized.replace(/\\(?:times|div|cdot|pm)\b/g, '');
  const letters = withoutCommands.match(/[a-zA-Z]/g) || [];
  if (!letters.length || !letters.every((letter) => letter.toLowerCase() === 'x')) return null;

  const explicit = normalized.match(/\\(times|div)\s+(-?\d+)/);
  const explicitOperator = explicit ? `\\${explicit[1]}` : '';
  const operator = explicitOperator || '\\times';
  const operationPattern = new RegExp(
    `^(?:x|X|\\\\times|\\\\div)\\s+(-?\\d+)(?:\\s+\\d+)?\\s+` +
    `(?:${explicitOperator ? escapeRegExp(explicitOperator) : '(?:x|X|\\\\times)'})\\s+(-?\\d+)$`
  );
  const match = normalized.match(operationPattern);
  if (!match) return null;

  const leftOperand = match[1];
  const rightOperand = match[2];
  if (leftOperand !== rightOperand) return null;
  return `${operator} ${leftOperand} ${operator} ${rightOperand}`;
}

function detachedOperationOperand(latex, { requireOperationMarker = false } = {}) {
  const raw = String(latex || '');
  const hasOperationMarker = /(?:\\(?:cdot|times)\b|[*.]|(?:^|\s)[xX](?:\s|_|\{|$))/.test(raw);
  if (requireOperationMarker && !hasOperationMarker) return '';

  const normalized = raw
    .replace(/\\cdot/g, '*')
    .replace(/\\times/g, '*')
    .replace(/\s+/g, ' ')
    .trim();
  if (!normalized || /[=<>+\-/]/.test(normalized)) return '';

  const subscript = normalized.match(/^(?:x|X|\*)\s*_\s*\{\s*((?:\d\s*){1,5})\s*\}$/);
  if (subscript) return (subscript[1].match(/\d/g) || []).join('');

  const single = normalized.match(/^(?:\*\s*)?((?:\d\s*){1,5})(?:\s*(?:\.|\*))?$/) ||
    normalized.match(/^(?:\.|\*)\s*((?:\d\s*){1,5})$/);
  if (!single) return '';
  return (single[1].match(/\d/g) || []).join('');
}

function looksLikePlainNumericLiteral(latex = '') {
  const compact = String(latex || '').replace(/\s+/g, '');
  return /^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(compact);
}

function repairOperationAnnotationFromPrevious(latex, previousLatex = []) {
  const operand = additiveConstantToRemove(previousLatex);
  if (!operand) return null;

  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || /[<>]/.test(normalized)) return null;
  const match = normalized.match(/^-\s+([a-zA-Z0-9\s]{1,12})\s*(?:-|=)\s*([a-zA-Z0-9\s]{1,12})$/);
  if (!match) return null;
  const left = operationOperandDigits(match[1], operand);
  const right = operationOperandDigits(match[2], operand);
  if (!left || !right) return null;
  const leftRaw = (String(match[1] || '').match(/\d/g) || []).join('');
  const rightRaw = (String(match[2] || '').match(/\d/g) || []).join('');
  if (leftRaw === operand && rightRaw === operand && normalized.includes('-')) return null;
  if (left !== right && right !== operand) return null;
  return `- ${operand} - ${operand}`;
}

function operationOperandDigits(text, contextualOperand = '') {
  const rawDigits = (String(text || '').match(/\d/g) || []).join('');
  if (rawDigits === contextualOperand) return rawDigits;
  const repaired = String(text || '').replace(/[xXlI]/g, '1').replace(/[oO]/g, '0');
  const repairedDigits = (repaired.match(/\d/g) || []).join('');
  if (contextualOperand && repairedDigits === contextualOperand) return repairedDigits;
  return rawDigits;
}

function contextualProblemDenominatorMultiplier(problemLatex = '') {
  const denominators = [];
  const pattern = /\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*((?:\d\s*){1,5})\s*\}/g;
  for (const match of String(problemLatex || '').matchAll(pattern)) {
    const value = Number((match[1].match(/\d/g) || []).join(''));
    if (Number.isFinite(value) && value > 0) denominators.push(value);
  }
  if (denominators.length < 2) return 0;
  return denominators.reduce((product, value) => lcm(product, value), 1);
}

function lcm(a, b) {
  if (!a || !b) return 0;
  return Math.abs(a * b) / gcd(a, b);
}

function gcd(a, b) {
  let left = Math.abs(Number(a) || 0);
  let right = Math.abs(Number(b) || 0);
  while (right) {
    const next = left % right;
    left = right;
    right = next;
  }
  return left;
}

function looksLikeMalformedEqualMultiplierOperation(latex) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || /[=<>]/.test(normalized)) return false;
  if (/^[+\-/]/.test(normalized)) return false;
  if (!/\\(?:times|cdot|div)\b|(?:^|\s)[xX](?:\s|_|$)/.test(normalized)) return false;
  if (/\\(?:frac|sqrt|log|ln|int|sum|prod)\b/.test(normalized)) return false;
  const withoutCommands = normalized.replace(/\\(?:times|cdot|div|pm)\b/g, '');
  const letters = withoutCommands.match(/[a-zA-Z]/g) || [];
  return Boolean(letters.length) && letters.every((letter) => ['x', 'n', 'o', 'l'].includes(letter.toLowerCase()));
}

function additiveConstantToRemove(previousLatex = []) {
  for (let index = previousLatex.length - 1; index >= 0; index -= 1) {
    const latex = String(previousLatex[index] || '').replace(/\s+/g, ' ').trim();
    if (!latex || (latex.match(/=/g) || []).length !== 1) continue;
    for (const side of latex.split('=')) {
      if (!/[A-Za-z\\]/.test(side)) continue;
      const matches = [...side.matchAll(/(?:^|\s)\+\s*((?:\d\s*){1,5})(?=\s|$)/g)];
      if (!matches.length) continue;
      const operand = (matches[matches.length - 1][1].match(/\d/g) || []).join('');
      if (operand && Number(operand) > 0) return operand;
    }
  }
  return '';
}

function groupingLooksUnbalanced(latex) {
  const pairs = { '(': ')', '[': ']', '{': '}' };
  const closers = new Set(Object.values(pairs));
  const stack = [];
  for (const char of String(latex || '')) {
    if (pairs[char]) {
      stack.push(pairs[char]);
    } else if (closers.has(char)) {
      if (stack.pop() !== char) return true;
    }
  }
  return stack.length > 0;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function mergePredictionAttempts(attempts) {
  const usable = (attempts || []).filter(Boolean);
  const mergedCandidates = [];
  const seen = new Set();
  for (const attempt of usable) {
    for (const candidate of attempt.candidates || []) {
      const latex = String(candidate?.latex || '').trim();
      if (!latex || seen.has(latex)) continue;
      seen.add(latex);
      mergedCandidates.push({
        ...candidate,
        retryTargetPixelHeight: attempt.retryTargetPixelHeight ?? null
      });
    }
  }

  const success = usable.find((attempt) => {
    if (attempt.failed || attempt.timedOut) return false;
    const candidates = attempt.candidates || [];
    const latex = String(attempt.latex || attempt.top?.latex || candidates[0]?.latex || '').trim();
    return latex && !isSuspiciousOperationLatex(latex);
  }) || usable.find((attempt) => {
    if (attempt.failed || attempt.timedOut) return false;
    const candidates = attempt.candidates || [];
    return String(attempt.latex || attempt.top?.latex || candidates[0]?.latex || '').trim();
  });
  const best = success || usable[0] || {};
  const top = best.top || mergedCandidates[0] || null;
  return {
    ...best,
    failed: success ? false : Boolean(best.failed),
    timedOut: success ? false : Boolean(best.timedOut),
    retryUsed: usable.some((attempt) => attempt.retryTargetPixelHeight),
    retryAttempts: usable.filter((attempt) => attempt.retryTargetPixelHeight).map((attempt) => ({
      targetPixelHeight: attempt.retryTargetPixelHeight,
      elapsedSeconds: attempt.elapsedSeconds,
      timedOut: Boolean(attempt.timedOut),
      failed: Boolean(attempt.failed),
      latex: attempt.latex || attempt.top?.latex || attempt.candidates?.[0]?.latex || ''
    })),
    candidates: mergedCandidates.length ? mergedCandidates : (best.candidates || []),
    top,
    latex: String(best.latex || top?.latex || '').trim()
  };
}

async function recognizeChunkedLine(candidate, {
  apiUrl,
  model,
  timeoutMs,
  problemLatex = '',
  rasterPadding,
  initialRasterHeight,
  minCssWidth,
  maxChunkCssWidth,
  minGap,
  signal,
  recognizeLine
}) {
  const cssWidth = Math.ceil(candidate?.tightBbox?.xMax ?? 0) - Math.floor(candidate?.tightBbox?.xMin ?? 0) +
    rasterPadding * 2;
  const structural = candidateNeedsExtendedTimeout(candidate);
  const compactFractionStructure = candidateHasLocalFractionStructure(candidate);
  const effectiveMinCssWidth = structural ? Math.min(Number(minCssWidth || 0), 220) : Number(minCssWidth || 0);
  if (!candidate?.strokes?.length || cssWidth < effectiveMinCssWidth) return null;

  const chunks = splitCandidateIntoHorizontalChunks(candidate, {
    maxChunkCssWidth: compactFractionStructure ? Math.min(Number(maxChunkCssWidth || 0) || 140, 140) : maxChunkCssWidth,
    minGap: structural ? 8 : (compactFractionStructure ? Math.min(Number(minGap || 0) || 12, 12) : minGap)
  });
  if (chunks.length < 2) return null;

  const parts = [];
  const attempts = [];
  for (const [index, chunk] of chunks.entries()) {
    throwIfAborted(signal);
    if (chunk.literalLatex) {
      parts.push(chunk.literalLatex);
      attempts.push({ literalLatex: chunk.literalLatex, strokeIds: chunk.strokeIds });
      continue;
    }

    const inferredLiteral = inferContextualChunkLiteral(chunk, chunks, index, problemLatex);
    if (inferredLiteral) {
      parts.push(inferredLiteral);
      attempts.push({ literalLatex: inferredLiteral, inferredLiteral: true, strokeIds: chunk.strokeIds });
      continue;
    }

    if (bboxWidth(chunk.tightBbox || {}) > 220 && splitFractionChunk(chunk)) {
      const fraction = await recognizeFractionChunk(chunk, {
        apiUrl,
        model,
        timeoutMs,
        rasterPadding,
        initialRasterHeight,
        signal,
        recognizeLine
      });
      if (fraction?.attempt) attempts.push(fraction.attempt);
      if (fraction?.latex) {
        parts.push(fraction.latex);
        continue;
      }
    }

    const { prediction, attempts: partAttempts } = await recognizeChunkPart(chunk, {
      apiUrl,
      model,
      timeoutMs,
      rasterPadding,
      initialRasterHeight,
      signal,
      recognizeLine
    });
    attempts.push({
      strokeIds: chunk.strokeIds,
      prediction,
      attempts: partAttempts
    });

    let latex = chooseChunkLatex(prediction);
    if (!latex) {
      const fraction = await recognizeFractionChunk(chunk, {
        apiUrl,
        model,
        timeoutMs,
        rasterPadding,
        initialRasterHeight,
        signal,
        recognizeLine
      });
      if (fraction?.latex) {
        latex = fraction.latex;
        attempts.push(fraction.attempt);
      } else {
        if (fraction?.attempt) attempts.push(fraction.attempt);
        return {
          failed: true,
          chunkFallback: true,
          attempts
        };
      }
    }
    parts.push(latex);
  }

  const latex = normalizeChunkedLatex(parts.join(' '));
  if (!latex) {
    return {
      failed: true,
      chunkFallback: true,
      attempts
    };
  }

  return {
    model,
    latex,
    top: { latex, score: 0, source: 'chunk-fallback' },
    candidates: [{ latex, score: 0, source: 'chunk-fallback' }],
    confidence: 0,
    failed: false,
    timedOut: false,
    chunkFallback: true,
    chunkAttempts: attempts
  };
}

async function recognizeChunkPart(chunk, {
  apiUrl,
  model,
  timeoutMs,
  rasterPadding,
  initialRasterHeight,
  signal,
  recognizeLine
}) {
  const attempts = [];
  let bestPrediction = null;
  for (const targetPixelHeight of chunkRasterHeights(initialRasterHeight)) {
    throwIfAborted(signal);
    const image = rasterizeLineCandidate(chunk, {
      padding: rasterPadding,
      targetPixelHeight
    });
    const prediction = await Promise.resolve()
      .then(() => recognizeLine(image, { apiUrl, model, timeoutMs, signal }))
      .catch((error) => recognitionFailureFromError(error, { model }));
    throwIfAborted(signal);
    const attempt = {
      targetPixelHeight,
      prediction,
      topLatex: chooseChunkLatex(prediction)
    };
    attempts.push(attempt);
    bestPrediction = prediction;
    if (attempt.topLatex && !predictionNeedsRetry(prediction)) {
      return { prediction, attempts };
    }
  }
  return {
    prediction: bestPrediction || {
      model,
      latex: '',
      candidates: [],
      confidence: 0,
      failed: true,
      elapsedSeconds: 0
    },
    attempts
  };
}

async function applyChunkFallbackToEntry(entry, candidate, {
  apiUrl,
  model,
  timeoutMs,
  rasterPadding,
  initialRasterHeight,
  minCssWidth,
  maxChunkCssWidth,
  minGap,
  recognizeLine,
  problemLatex,
  previousLatex,
  signal,
  evidenceByCandidateId
}) {
  const chunked = await recognizeChunkedLine(candidate, {
    apiUrl,
    model,
    timeoutMs,
    problemLatex,
    rasterPadding,
    initialRasterHeight,
    minCssWidth,
    maxChunkCssWidth,
    minGap,
    signal,
    recognizeLine
  });
  if (!chunked || chunked.failed) return false;
  entry.chunkFallback = chunked;
  entry.prediction = chunked;
  entry.candidates = chunked.candidates || [];
  entry.ocrLatex = predictionLatex(chunked);
  entry.latex = chunked.latex || chunked.top?.latex || entry.latex || '';
  entry.evidenceScore = scoreRecognitionEvidence(candidate, chunked, { problemLatex, previousLatex });
  evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
  return true;
}

async function recognizeFractionChunk(chunk, {
  apiUrl,
  model,
  timeoutMs,
  rasterPadding,
  initialRasterHeight,
  signal,
  recognizeLine
}) {
  const split = splitFractionChunk(chunk);
  if (!split) return null;

  const attempt = {
    fractionSubchunk: true,
    strokeIds: chunk.strokeIds,
    barStrokeId: split.bar.id,
    parts: []
  };
  const latexByRole = {};
  for (const role of ['numerator', 'denominator']) {
    throwIfAborted(signal);
    const part = split[role];
    const { latex, attempts } = await recognizeFractionPartChunk(part, {
      role,
      apiUrl,
      model,
      timeoutMs,
      rasterPadding,
      initialRasterHeight,
      signal,
      recognizeLine
    });
    attempt.parts.push(...attempts);
    if (!latex) return { latex: '', attempt };
    latexByRole[role] = latex;
  }

  return {
    latex: `\\frac { ${latexByRole.numerator} } { ${latexByRole.denominator} }`,
    attempt
  };
}

async function recognizeFractionPartChunk(part, {
  role,
  apiUrl,
  model,
  timeoutMs,
  rasterPadding,
  initialRasterHeight,
  signal,
  recognizeLine
}) {
  const attempts = [];
  for (const targetPixelHeight of chunkRasterHeights(initialRasterHeight)) {
    throwIfAborted(signal);
    const image = rasterizeLineCandidate(part, {
      padding: rasterPadding,
      targetPixelHeight
    });
    const prediction = await Promise.resolve()
      .then(() => recognizeLine(image, { apiUrl, model, timeoutMs, signal }))
      .catch((error) => recognitionFailureFromError(error, { model }));
    throwIfAborted(signal);
    const latex = chooseChunkLatex(prediction);
    attempts.push({
      role,
      strokeIds: part.strokeIds,
      targetPixelHeight,
      prediction,
      topLatex: latex
    });
    if (latex && !predictionNeedsRetry(prediction)) {
      return { latex, attempts };
    }
  }
  return { latex: '', attempts };
}

function chunkRasterHeights(targetHeight) {
  const heights = [];
  for (const height of [...FRACTION_CHUNK_RASTER_HEIGHTS, targetHeight]) {
    const value = Number(height);
    if (Number.isFinite(value) && value > 0 && !heights.includes(value)) {
      heights.push(value);
    }
  }
  return heights;
}

function splitFractionChunk(chunk) {
  const strokes = (chunk.strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (strokes.length < 3) return null;
  const box = chunk.tightBbox || bboxForStrokes(strokes);
  const splits = strokes
    .filter((stroke) => strokeLooksLikeFractionBar(stroke.canvasBbox, box))
    .map((stroke) => fractionBarSplit(stroke, strokes))
    .filter((split) => split && split.score >= 0.16)
    .sort((a, b) => b.score - a.score);
  if (!splits.length) return null;
  const { bar, numerator, denominator } = splits[0];

  return {
    bar,
    numerator: makeChunkCandidate(numerator, chunk, 1),
    denominator: makeChunkCandidate(denominator, chunk, 2)
  };
}

function fractionBarSplit(bar, strokes) {
  const barMid = (bar.canvasBbox.yMin + bar.canvasBbox.yMax) / 2;
  const numerator = strokes.filter((stroke) => stroke !== bar && stroke.canvasBbox.yMax <= barMid);
  const denominator = strokes.filter((stroke) => stroke !== bar && stroke.canvasBbox.yMin >= barMid);
  if (!numerator.length || !denominator.length) return null;
  const barWidth = Math.max(1, bboxWidth(bar.canvasBbox));
  const numeratorWidth = bboxWidth(bboxForStrokes(numerator));
  const denominatorWidth = bboxWidth(bboxForStrokes(denominator));
  const widthBalance = Math.min(numeratorWidth, denominatorWidth) / barWidth;
  const countBalance = Math.min(numerator.length, denominator.length) / Math.max(numerator.length, denominator.length);
  return {
    bar,
    numerator,
    denominator,
    score: widthBalance + 0.55 * countBalance
  };
}

function strokeLooksLikeFractionBar(box, parentBox) {
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  const parentWidth = Math.max(1, bboxWidth(parentBox));
  const parentHeight = Math.max(1, bboxHeight(parentBox));
  if (width < Math.max(18, parentWidth * 0.45)) return false;
  if (height <= 18 && width >= height * 3.5) return true;
  if (width >= parentWidth * 0.65 && height <= parentHeight * 0.4 && width >= height * 5) return true;
  return width >= parentWidth * 0.7 && height <= parentHeight * 0.65 && width >= height * 4.5;
}

function candidateCanUseChunking(candidate, {
  rasterPadding,
  minCssWidth
}) {
  const cssWidth = Math.ceil(candidate?.tightBbox?.xMax ?? 0) - Math.floor(candidate?.tightBbox?.xMin ?? 0) +
    rasterPadding * 2;
  if (!Number.isFinite(cssWidth)) return false;
  if (cssWidth >= Number(minCssWidth || 0)) return true;
  return candidateNeedsExtendedTimeout(candidate) && cssWidth >= Math.min(Number(minCssWidth || 0), 220);
}

function splitCandidateIntoHorizontalChunks(candidate, {
  maxChunkCssWidth,
  minGap
}) {
  const strokes = (candidate.strokes || [])
    .filter((stroke) => stroke?.canvasBbox)
    .slice()
    .sort((a, b) => (
      a.canvasBbox.xMin - b.canvasBbox.xMin ||
      a.canvasBbox.yMin - b.canvasBbox.yMin
    ));
  if (strokes.length < 2) return [];

  const atoms = makeFractionAwareChunkAtoms(strokes, { minGap }, candidate);

  const chunks = [];
  let pending = [];
  const flushPending = () => {
    if (!pending.length) return;
    chunks.push(makeChunkCandidate(pending.flatMap((atom) => atom.strokes), candidate, chunks.length));
    pending = [];
  };

  for (const atom of atoms) {
    if (atom.atomicChunk) {
      flushPending();
      chunks.push(makeChunkCandidate(atom.strokes, candidate, chunks.length));
      continue;
    }

    if (isEqualsAtom(atom)) {
      flushPending();
      chunks.push({
        ...makeChunkCandidate(atom.strokes, candidate, chunks.length),
        literalLatex: '='
      });
      continue;
    }

    const proposed = pending.concat(atom);
    const proposedBox = bboxForStrokes(proposed.flatMap((item) => item.strokes));
    if (
      pending.length > 0 &&
      bboxWidth(proposedBox) > maxChunkCssWidth
    ) {
      flushPending();
    }
    pending.push(atom);
  }
  flushPending();

  return chunks.filter((chunk) => chunk.strokeIds.length > 0);
}

function makeFractionAwareChunkAtoms(strokes, { minGap }, candidate) {
  const fractionGroups = findLocalFractionGroups(strokes);
  const assigned = new Set(fractionGroups.flatMap((group) => group));
  const entries = fractionGroups.map((group) => ({
    ...makeChunkAtom(group, candidate),
    atomicChunk: true
  }));

  const remaining = strokes.filter((stroke) => !assigned.has(stroke));
  if (remaining.length) {
    let current = [remaining[0]];
    let currentBox = { ...remaining[0].canvasBbox };
    for (const stroke of remaining.slice(1)) {
      const gap = stroke.canvasBbox.xMin - currentBox.xMax;
      if (gap >= minGap) {
        entries.push(makeChunkAtom(current, candidate));
        current = [stroke];
        currentBox = { ...stroke.canvasBbox };
      } else {
        current.push(stroke);
        currentBox = bboxUnion(currentBox, stroke.canvasBbox);
      }
    }
    entries.push(makeChunkAtom(current, candidate));
  }

  return entries.sort((a, b) => (
    (a.bbox?.xMin ?? 0) - (b.bbox?.xMin ?? 0) ||
    (a.bbox?.yMin ?? 0) - (b.bbox?.yMin ?? 0)
  ));
}

function findLocalFractionGroups(strokes) {
  const groups = [];
  const used = new Set();
  const bars = strokes
    .filter((stroke) => strokeIsHorizontalBarLike(stroke.canvasBbox))
    .sort((a, b) => bboxWidth(b.canvasBbox) - bboxWidth(a.canvasBbox));

  for (const bar of bars) {
    if (used.has(bar)) continue;
    const numerator = nearestFractionSideStrokes(strokes, bar, { side: 'above', used });
    const denominator = nearestFractionSideStrokes(strokes, bar, { side: 'below', used });
    if (!numerator.length || !denominator.length) continue;
    const group = numerator.concat(bar, denominator);
    for (const stroke of group) used.add(stroke);
    groups.push(group.slice().sort((a, b) => (
      a.canvasBbox.xMin - b.canvasBbox.xMin ||
      a.canvasBbox.yMin - b.canvasBbox.yMin
    )));
  }
  return groups;
}

function nearestFractionSideStrokes(strokes, bar, { side, used }) {
  const barBox = bar.canvasBbox;
  const barCenter = bboxXCenter(barBox);
  const matches = [];
  for (const stroke of strokes) {
    if (stroke === bar || used.has(stroke)) continue;
    const box = stroke.canvasBbox;
    const verticalGap = side === 'above'
      ? barBox.yMin - box.yMax
      : box.yMin - barBox.yMax;
    if (verticalGap < -2 || verticalGap > 44) continue;
    const overlap = horizontalOverlapRatio(box, barBox);
    const centerDistance = Math.abs(bboxXCenter(box) - barCenter);
    if (overlap < 0.25 && centerDistance > Math.max(18, bboxWidth(barBox) * 0.75)) {
      continue;
    }
    matches.push({ score: centerDistance + Math.max(0, verticalGap) * 0.15, stroke });
  }
  if (!matches.length) return [];
  matches.sort((a, b) => a.score - b.score);
  const bestScore = matches[0].score;
  const clusterTolerance = Math.max(16, bboxWidth(barBox) * 0.4);
  return matches.filter((item) => item.score <= bestScore + clusterTolerance).map((item) => item.stroke);
}

function strokeIsHorizontalBarLike(box) {
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  return width >= 18 && height <= 12 && width >= height * 3.5;
}

function makeChunkAtom(strokes, parent) {
  return {
    strokes,
    bbox: bboxForStrokes(strokes),
    parentCandidateId: parent.candidateId
  };
}

function makeChunkCandidate(strokes, parent, index) {
  const tightBbox = bboxForStrokes(strokes);
  return {
    candidateId: `${parent.candidateId}_chunk_${index + 1}`,
    id: `${parent.candidateId}_chunk_${index + 1}`,
    profiles: ['chunk-fallback'],
    profile: 'chunk-fallback',
    strokeIds: strokes.map((stroke) => String(stroke.id)),
    strokes,
    tightBbox,
    expandedBbox: tightBbox,
    conflicts: [],
    parentCandidateId: parent.candidateId
  };
}

function inferContextualChunkLiteral(chunk, chunks, index, problemLatex) {
  if (index < 0 || index >= chunks.length - 1) return '';
  if (chunks[index + 1]?.literalLatex !== '=') return '';
  const strokes = (chunk.strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (strokes.length < 1 || strokes.length > 2) return '';
  const box = chunk.tightBbox || bboxForStrokes(strokes);
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  if (width < 8 || height < 10) return '';
  if (width >= height * 2.2 || height >= width * 2.8) return '';
  const variables = contextualLatinVariables(problemLatex);
  return variables.length === 1 ? variables[0] : '';
}

function contextualLatinVariables(latex) {
  const text = String(latex || '').replace(/\\[A-Za-z]+/g, ' ');
  const variables = [];
  for (const match of text.matchAll(/\b[a-z]\b/g)) {
    const variable = match[0];
    if (!variables.includes(variable)) variables.push(variable);
  }
  return variables;
}

function isEqualsAtom(atom) {
  const strokes = atom.strokes || [];
  if (strokes.length !== 2) return false;
  const box = atom.bbox;
  if (!box) return false;
  if (bboxWidth(box) < bboxHeight(box) * 1.35) return false;

  const horizontal = strokes.every((stroke) => (
    bboxWidth(stroke.canvasBbox) >= bboxHeight(stroke.canvasBbox) * 2.5
  ));
  if (!horizontal) return false;

  const overlap = horizontalOverlapRatio(strokes[0].canvasBbox, strokes[1].canvasBbox);
  const verticalGap = Math.max(
    strokes[0].canvasBbox.yMin,
    strokes[1].canvasBbox.yMin
  ) - Math.min(
    strokes[0].canvasBbox.yMax,
    strokes[1].canvasBbox.yMax
  );
  return overlap >= 0.45 && verticalGap > 0;
}

function bboxForStrokes(strokes) {
  return (strokes || []).reduce((box, stroke) => bboxUnion(box, stroke.canvasBbox), {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity
  });
}

function bboxUnion(a, b) {
  if (!a || !Number.isFinite(a.xMin)) return { ...b };
  if (!b) return { ...a };
  return {
    xMin: Math.min(a.xMin, b.xMin),
    yMin: Math.min(a.yMin, b.yMin),
    xMax: Math.max(a.xMax, b.xMax),
    yMax: Math.max(a.yMax, b.yMax)
  };
}

function bboxWidth(box) {
  return Math.max(0, (box?.xMax ?? 0) - (box?.xMin ?? 0));
}

function bboxHeight(box) {
  return Math.max(0, (box?.yMax ?? 0) - (box?.yMin ?? 0));
}

function bboxXCenter(box) {
  return ((box?.xMin ?? 0) + (box?.xMax ?? 0)) / 2;
}

function bboxYCenter(box) {
  return ((box?.yMin ?? 0) + (box?.yMax ?? 0)) / 2;
}

function bboxVerticalOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const yMin = Math.max(a.yMin ?? 0, b.yMin ?? 0);
  const yMax = Math.min(a.yMax ?? 0, b.yMax ?? 0);
  if (yMax <= yMin) return 0;
  const smallerHeight = Math.min(
    Math.max(1, bboxHeight(a)),
    Math.max(1, bboxHeight(b))
  );
  return (yMax - yMin) / smallerHeight;
}

function bboxHorizontalGap(a, b) {
  if (!a || !b) return Infinity;
  return Math.max(
    0,
    Math.max(a.xMin ?? 0, b.xMin ?? 0) - Math.min(a.xMax ?? 0, b.xMax ?? 0)
  );
}

function bboxOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const xMin = Math.max(a.xMin ?? 0, b.xMin ?? 0);
  const yMin = Math.max(a.yMin ?? 0, b.yMin ?? 0);
  const xMax = Math.min(a.xMax ?? 0, b.xMax ?? 0);
  const yMax = Math.min(a.yMax ?? 0, b.yMax ?? 0);
  if (xMax <= xMin || yMax <= yMin) return 0;
  const overlapArea = (xMax - xMin) * (yMax - yMin);
  const smallerArea = Math.min(
    Math.max(1, bboxWidth(a) * bboxHeight(a)),
    Math.max(1, bboxWidth(b) * bboxHeight(b))
  );
  return overlapArea / smallerArea;
}

function normalizeChunkedLatex(latex) {
  return String(latex || '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([=+\-*/^_{}()[\]])/g, ' $1')
    .replace(/([=+\-*/^_{}()[\]])\s+/g, '$1 ')
    .trim();
}

function chooseChunkLatex(prediction = {}) {
  const candidates = prediction?.candidates || [];
  const topLatex = String(prediction?.latex || prediction?.top?.latex || candidates[0]?.latex || '').trim();
  if (!topLatex) return '';

  if (/(\w|\))\s*\^\s*\{\s*f\s*\}/.test(topLatex)) {
    const prime = candidates.find((candidate) => /\\prime/.test(String(candidate.latex || '')));
    if (prime?.latex) return String(prime.latex).trim();
  }

  if (/^t\s+[a-zA-Z\\]/.test(topLatex)) {
    const suffix = topLatex.replace(/^t\s+/, '').trim();
    const plus = candidates.find((candidate) => {
      const latex = String(candidate.latex || '').trim();
      return latex.startsWith('+') && latex.slice(1).trim() === suffix;
    });
    if (plus?.latex) return String(plus.latex).trim();
  }

  return topLatex;
}

export function shouldUseSemanticLatex(currentLatex, semanticEntry = {}) {
  const bestLatex = String(semanticEntry.bestLatex || '').trim();
  if (!bestLatex) return false;
  if (semanticEntry.sound === false) return false;
  const current = String(currentLatex || '').trim();
  if (!current || current === bestLatex) return true;
  if (looksLikeOperationAnnotation(current) && !looksLikeOperationAnnotation(bestLatex)) return false;
  if (looksLikeOperationAnnotation(bestLatex) && !looksLikeOperationAnnotation(current)) return true;
  if (semanticReplacementFightsVisibleRow(current, bestLatex, semanticEntry)) return false;
  const currentScore = (semanticEntry.candidateScores || []).find((candidate) => (
    String(candidate.latex || '').trim() === current
  ));
  if (
    currentScore?.sound === true &&
    (currentScore.equivalentToProblem || currentScore.equivalentToPrevious)
  ) {
    return false;
  }
  const bestScore = semanticCandidateScore(semanticEntry, bestLatex);
  if (semanticBestIsProblemSupportedNumericRepair(bestScore)) return true;
  if (semanticPreviousBestFightsStrongerVisualCurrent(current, bestLatex, semanticEntry, currentScore)) return false;
  if (semanticEntry.equivalentToProblem || semanticEntry.equivalentToPrevious) return true;
  if (bestScore?.detail?.solutionSupportedByProblem) return true;

  const semanticScore = Number(semanticEntry.semanticScore);
  if (Number.isFinite(semanticScore) && semanticScore >= 3) return true;
  if (shouldTrustContextualSemanticBest(current, bestLatex, semanticEntry, currentScore)) return true;

  return currentScore?.sound === false && semanticEntry.sound === true;
}

function gradingSelectedLatex(entry = {}, semanticEntry = {}) {
  const grading = semanticEntry?.grading || entry?.grading || null;
  if (!gradingPreferred(grading)) return '';
  if (!candidateSelectionSafeForGrading(entry, grading)) return '';
  const selectedIndex = Number(grading.selectedCandidateIndex);
  const candidates = Array.isArray(entry?.candidates) ? entry.candidates : [];
  if (Number.isInteger(selectedIndex) && selectedIndex >= 0) {
    const selectedLatex = String(candidates[selectedIndex]?.latex || '').trim();
    if (selectedLatex) return selectedLatex;
  }
  const studentLatex = String(grading.studentLatex || '').trim();
  if (studentLatex) return studentLatex;
  return String(semanticEntry.bestLatex || '').trim();
}

function semanticLatexSafeForReplacement(entry = {}, semanticEntry = {}) {
  const grading = semanticEntry?.grading || entry?.grading || null;
  return !gradingPreferred(grading) || candidateSelectionSafeForGrading(entry, grading);
}

function candidateSelectionSafeForGrading(entry = {}, grading = null) {
  if (!gradingPreferred(grading)) return false;
  if (entry?.excludedFromGrading) return false;

  const selectedIndex = Number(grading?.selectedCandidateIndex);
  const candidates = Array.isArray(entry?.candidates) ? entry.candidates : [];
  const selectedLatex = Number.isInteger(selectedIndex) && selectedIndex >= 0
    ? String(candidates[selectedIndex]?.latex || '').trim()
    : String(grading?.studentLatex || '').trim();
  const topLatex = String(candidates[0]?.latex || entry?.ocrLatex || entry?.latex || '').trim();

  if (!selectedLatex) return true;
  if (!Number.isInteger(selectedIndex) || selectedIndex <= 0 || sameLatexForGrading(selectedLatex, topLatex)) {
    return true;
  }

  if (entry?.ocrRepair?.source) return false;
  if (latexUnsafeForAlternateGrading(entry?.ocrLatex || topLatex || entry?.latex || entry?.acceptedLatex)) return false;

  if (!looksLikeCleanFinalAnswerLatex(selectedLatex)) return false;

  return true;
}

function sameLatexForGrading(left = '', right = '') {
  return compactLatexForGrading(left) === compactLatexForGrading(right);
}

function compactLatexForGrading(latex = '') {
  return String(latex || '').replace(/\s+/g, '').trim();
}

function latexUnsafeForAlternateGrading(latex = '') {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return false;
  return /\\pm\b|±/.test(normalized) ||
    isSuspiciousOperationLatex(normalized) ||
    looksLikePlainNumericLiteral(normalized) && /(?:\.|\*)\s*$/.test(normalized);
}

function selectionContextUnsafeForAlternateGrading(entry = {}) {
  if (entry?.skipReason) return true;
  const profiles = entry?.profiles || [];
  if (profiles.includes('parent') || profiles.includes('row-parent') || profiles.includes('dbnet-parent')) {
    return true;
  }
  return profiles.includes('projection-line') || profiles.includes('loose') || profiles.includes('temporal');
}

function looksLikeCleanFinalAnswerLatex(latex = '') {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return false;
  const simpleNumber = /^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(normalized.replace(/\s+/g, ''));
  if (simpleNumber) return true;
  const assignment = normalized.match(/^[a-zA-Z]\s*=\s*(.+)$/);
  if (!assignment) return false;
  const value = assignment[1].trim();
  if (!value) return false;
  if (/[=<>]/.test(value)) return false;
  const withoutAllowedNames = value
    .replace(/\\(?:frac|sqrt|pi|pm|cdot|times|div)\b/g, '')
    .replace(/\b(?:pi|e)\b/g, '')
    .replace(/[{}()[\]\d\s.+\-*/^]/g, '');
  return !/[A-Za-z]/.test(withoutAllowedNames);
}

function gradingPreferred(grading = null) {
  if (!grading) return false;
  if (grading.solutionCoverage === 'full' || grading.solutionCoverage === 'partial') return true;
  if ((grading.matchedSolutions || []).length > 0) return true;
  return grading.classification === 'valid_step';
}

function gradingEvidenceBoost(entry = {}, grading = null) {
  if (!gradingPreferred(grading)) return 0;
  if (!candidateSelectionSafeForGrading(entry, grading)) return 0;
  if (grading.solutionCoverage === 'full') return 8;
  if (grading.solutionCoverage === 'partial') return 6;
  return 4;
}

function gradableRecognitionLines(lines = []) {
  return (lines || []).filter((line) => !line?.excludedFromGrading);
}

function gradingPayloadCandidates(line = {}) {
  if (!alternateCandidatesSafeForGrading(line)) return [];
  return (line.candidates || []).slice(0, 5);
}

function alternateCandidatesSafeForGrading(line = {}) {
  if (line?.excludedFromGrading) return false;
  if (line?.ocrRepair?.source) return false;
  if (selectionContextUnsafeForAlternateGrading(line)) return false;
  const latex = line.acceptedLatex || line.latex || line.ocrLatex || '';
  if (latexUnsafeForAlternateGrading(latex)) return false;
  const evidenceScore = Number(line.evidenceScore);
  return Number.isFinite(evidenceScore) && evidenceScore >= 3;
}

function safeLineGradingForAggregation(line = {}) {
  if (line?.excludedFromGrading) {
    return {
      studentLatex: line.acceptedLatex || line.latex || '',
      classification: 'other',
      selectedCandidateIndex: null,
      solutionCoverage: 'none',
      matchedSolutions: [],
      answerFinality: 'not_answer',
      countsTowardCompletion: false
    };
  }
  const grading = line.grading || line.sequentialSemantic?.grading || line.contextualSemantic?.grading || null;
  if (!grading) return null;
  if (!gradingPreferred(grading)) return grading;
  if (candidateSelectionSafeForGrading(line, grading)) return grading;
  return {
    ...grading,
    classification: 'other',
    solutionCoverage: 'none',
    matchedSolutions: [],
    answerFinality: 'not_answer',
    countsTowardCompletion: false
  };
}

function buildLiveGradingResult({ problemLatex = '', answerManifest = null, lines = [] } = {}) {
  const steps = (lines || []).map((line, index) => {
    const grading = safeLineGradingForAggregation(line);
    return {
      lineIndex: line.lineIndex ?? index,
      studentLatex: line.acceptedLatex || line.latex || grading?.studentLatex || '',
      classification: grading?.classification || 'other',
      selectedCandidateIndex: grading?.selectedCandidateIndex ?? null,
      solutionCoverage: grading?.solutionCoverage || 'none',
      matchedSolutions: Array.isArray(grading?.matchedSolutions) ? grading.matchedSolutions : [],
      answerFinality: grading?.answerFinality || 'not_answer',
      countsTowardCompletion: grading?.countsTowardCompletion !== false
    };
  });
  const exactSet = Array.isArray(answerManifest?.exact_set)
    ? answerManifest.exact_set.map(String)
    : [];
  const matched = new Set();
  let sawValid = false;
  let firstInvalid = null;
  for (const step of steps) {
    if (step.classification === 'valid_step') sawValid = true;
    if (step.classification === 'invalid_step' && firstInvalid === null) {
      firstInvalid = step.lineIndex;
    }
    if (step.countsTowardCompletion === false) continue;
    for (const solution of step.matchedSolutions || []) {
      if (solution) matched.add(String(solution));
    }
  }
  const cardinality = answerManifest?.cardinality || 'unsupported';
  const complete = cardinality === 'finite'
    ? exactSet.length > 0 && exactSet.every((solution) => matched.has(solution))
    : steps.some((step) => step.countsTowardCompletion !== false && (
        step.solutionCoverage === 'full' || (step.matchedSolutions || []).length > 0
      ));
  const problemStatus = complete
    ? 'correct'
    : firstInvalid !== null
      ? 'incorrect'
      : sawValid
        ? 'incomplete'
        : 'not_started';

  return {
    status: 'complete',
    failed: false,
    problem: {
      latex: problemLatex,
      standardized: answerManifest?.problem_standardized || '',
      solveVariable: answerManifest?.variable || null,
      cardinality,
      solutionSet: exactSet,
      decimalSet: Array.isArray(answerManifest?.decimal_set) ? answerManifest.decimal_set : [],
      tolerance: answerManifest?.tolerance ?? 0.005,
      manifest: answerManifest
    },
    steps,
    result: {
      problemStatus,
      breakdownLineIndex: problemStatus === 'incorrect' ? firstInvalid : null,
      foundSolutions: [...matched],
      missingSolutions: cardinality === 'finite'
        ? exactSet.filter((solution) => !matched.has(solution))
        : []
    }
  };
}

function semanticBestIsProblemSupportedNumericRepair(bestScore = null) {
  const repair = bestScore?.detail?.repair;
  return bestScore?.sound === true &&
    bestScore?.detail?.solutionSupportedByProblem === true &&
    (
      repair === 'contextual_latex_numeric_equivalence' ||
      repair === 'contextual_quadratic_formula_coefficient'
    );
}

function semanticPreviousBestFightsStrongerVisualCurrent(currentLatex, bestLatex, semanticEntry = {}, currentScore = null) {
  if (semanticEntry.equivalentToProblem || !semanticEntry.equivalentToPrevious) return false;
  if (!currentScore || currentScore.sound !== true) return false;
  const bestScore = semanticCandidateScore(semanticEntry, bestLatex);
  if (!bestScore || bestScore.sound !== true) return false;
  if (!bestScore.equivalentToPrevious || bestScore.equivalentToProblem) return false;
  if (currentScore.equivalentToProblem || currentScore.equivalentToPrevious) return false;

  const currentModel = Number(currentScore.detail?.modelScore);
  const bestModel = Number(bestScore.detail?.modelScore);
  if (!Number.isFinite(currentModel) || !Number.isFinite(bestModel) || currentModel < bestModel + 0.5) {
    return false;
  }
  if (latexLineKind(currentLatex) !== latexLineKind(bestLatex)) return false;
  if (currentHasDerivativePrime(currentLatex, bestLatex)) return true;
  return latexTokenOverlap(currentLatex, bestLatex) < 0.8;
}

function currentHasDerivativePrime(currentLatex, bestLatex) {
  const current = String(currentLatex || '');
  const best = String(bestLatex || '');
  return /\\prime/.test(current) &&
    !/\\prime/.test(best) &&
    /\^\s*\{\s*\\prime\s*\}/.test(current);
}

function shouldTrustContextualSemanticBest(currentLatex, bestLatex, semanticEntry = {}, currentScore = null) {
  if (!currentScore || currentScore.equivalentToProblem || currentScore.equivalentToPrevious) return false;
  const bestScore = semanticCandidateScore(semanticEntry, bestLatex);
  if (!bestScore || bestScore.sound !== true) return false;
  if (bestScore.detail?.duplicatePreviousLatex) return false;
  if (latexLineKind(currentLatex) !== latexLineKind(bestLatex)) return false;

  const bestValue = Number(bestScore.score);
  const currentValue = Number(currentScore.score);
  if (!Number.isFinite(bestValue) || !Number.isFinite(currentValue) || bestValue <= currentValue) return false;

  const bestOverlap = Number(bestScore.detail?.characterOverlap);
  const currentOverlap = Number(currentScore.detail?.characterOverlap);
  return Number.isFinite(bestOverlap) &&
    Number.isFinite(currentOverlap) &&
    bestOverlap >= currentOverlap + 0.08;
}

function semanticCandidateScore(semanticEntry = {}, latex = '') {
  const target = String(latex || '').trim();
  return (semanticEntry.candidateScores || []).find((candidate) => (
    String(candidate.latex || '').trim() === target
  ));
}

function semanticReplacementFightsVisibleRow(currentLatex, bestLatex, semanticEntry = {}) {
  const bestScore = semanticCandidateScore(semanticEntry, bestLatex);
  const repair = bestScore?.detail?.repair;
  if (repair !== 'contextual_linear_simplification' && repair !== 'contextual_subtraction_step') return false;

  const bestOverlap = latexTokenOverlap(currentLatex, bestLatex);
  const currentTokens = latexTokens(currentLatex).length;
  const bestTokens = latexTokens(bestLatex).length;
  return currentTokens >= bestTokens + 3 && bestOverlap >= 0.45;
}

function latexTokenOverlap(leftLatex, rightLatex) {
  const left = latexTokens(leftLatex);
  const right = latexTokens(rightLatex);
  if (!left.length || !right.length) return 0;
  const remaining = [...right];
  let matched = 0;
  for (const token of left) {
    const index = remaining.indexOf(token);
    if (index < 0) continue;
    remaining.splice(index, 1);
    matched += 1;
  }
  return matched / Math.max(1, Math.min(left.length, right.length));
}

function latexTokens(latex) {
  return [...String(latex || '').matchAll(/\\[A-Za-z]+|[A-Za-z]+|\d+|[+\-*/=()]/g)]
    .map((match) => match[0].toLowerCase())
    .filter((token) => token !== '\\cdots' && token !== '\\ldots');
}

function latexLineKind(latex) {
  if (looksLikeOperationAnnotation(latex)) return 'operation';
  const text = String(latex || '').trim();
  if (!text) return '';
  if (text.includes('=')) return 'equation';
  return 'expression';
}

function looksLikeOperationAnnotation(latex) {
  const normalized = String(latex || '')
    .replace(/\\times/g, '*')
    .replace(/\\div/g, '/')
    .replace(/\s+/g, '')
    .trim();
  return /^([+\-*/]).+\1.+$/.test(normalized) && !/[=<>]/.test(normalized);
}

function normalizeContextualVariableCase(latex, contextLatex = []) {
  let output = String(latex || '');
  if (!output) return output;

  const variables = contextualLowercaseVariables(contextLatex);
  for (const variable of variables) {
    const upper = variable.toUpperCase();
    if (upper === variable) continue;
    const pattern = new RegExp(`(^|[^\\\\A-Za-z])${escapeRegExp(upper)}(?=$|[^A-Za-z])`, 'g');
    output = output.replace(pattern, `$1${variable}`);
  }
  return output;
}

function repairQuadraticFormulaFromProblem(latex, problemLatex = '') {
  const text = String(latex || '');
  if (!/\\frac\b/.test(text) || !/\\sqrt\b/.test(text) || !/\bx\b/.test(text) || !/=/.test(text)) {
    return null;
  }
  const coefficients = simpleQuadraticCoefficients(problemLatex);
  if (!coefficients) return null;
  const { a, b, c } = coefficients;
  if (a === 0) return null;

  return [
    'x = \\frac {',
    spacedSigned(-b),
    '+ \\sqrt {',
    coefficientMagnitude(b),
    '^ { 2 } - 4 (',
    coefficientLatex(a),
    ') (',
    coefficientLatex(c),
    ') } } { 2 (',
    coefficientLatex(a),
    ') }'
  ].join(' ').replace(/\s+/g, ' ').trim();
}

function repairInitialRationalProblemLine(latex, problemLatex = '', lineIndex = 0) {
  if (lineIndex !== 0) return null;
  const current = String(latex || '').trim();
  const problem = String(problemLatex || '').trim();
  if (!current || !problem || !current.includes('=') || !problem.includes('=')) return null;
  if (!/\\frac\b/.test(current) || !/\\frac\b/.test(problem)) return null;

  const variables = contextualLatinVariables(problem);
  if (variables.length !== 1) return null;
  const variable = variables[0];
  if (new RegExp(`\\b${escapeRegExp(variable)}\\b`).test(current)) return null;
  if (contextualLatinVariables(current).length > 0) return null;

  const currentParts = splitEquationSides(current);
  const problemParts = splitEquationSides(problem);
  if (!currentParts || !problemParts) return null;
  if (normalizeLatexComparable(currentParts.right) !== normalizeLatexComparable(problemParts.right)) return null;

  const currentConstants = latexConstants(currentParts.left).join('|');
  const problemConstants = latexConstants(problemParts.left).join('|');
  if (!currentConstants || currentConstants !== problemConstants) return null;

  if (!latexOperatorMultisetContains(
    latexOperators(currentParts.left),
    latexOperators(problemParts.left)
  )) {
    return null;
  }

  return problem;
}

function splitEquationSides(latex) {
  const parts = String(latex || '').split('=');
  if (parts.length !== 2) return null;
  return {
    left: parts[0].trim(),
    right: parts[1].trim()
  };
}

function normalizeLatexComparable(latex) {
  return String(latex || '')
    .replace(/\\left|\\right/g, '')
    .replace(/\s+/g, '')
    .replace(/[{}]/g, '')
    .trim();
}

function latexConstants(latex) {
  return uniqueStrings([...String(latex || '').matchAll(/\d+/g)].map((match) => match[0])).sort();
}

function latexOperators(latex) {
  return [...String(latex || '').matchAll(/\\frac|[+\-=^]/g)].map((match) => match[0]);
}

function latexOperatorMultisetContains(currentOperators, problemOperators) {
  const remaining = [...(currentOperators || [])];
  for (const operator of problemOperators || []) {
    const index = remaining.indexOf(operator);
    if (index < 0) return false;
    remaining.splice(index, 1);
  }
  return true;
}

function simpleQuadraticCoefficients(problemLatex = '') {
  const compact = String(problemLatex || '')
    .replace(/\\left|\\right/g, '')
    .replace(/\s+/g, '')
    .replace(/\^\{2\}/g, '^2')
    .replace(/\{|\}/g, '');
  const match = compact.match(/^([+-]?\d*)x\^2([+-]\d*)x([+-]\d+)=0$/);
  if (!match) return null;
  return {
    a: parseCoefficient(match[1]),
    b: parseCoefficient(match[2]),
    c: Number(match[3])
  };
}

function parseCoefficient(raw) {
  if (raw === '' || raw === '+') return 1;
  if (raw === '-') return -1;
  if (raw === '+') return 1;
  return Number(raw);
}

function coefficientMagnitude(value) {
  return coefficientLatex(Math.abs(value));
}

function coefficientLatex(value) {
  if (value < 0) return `- ${Math.abs(value)}`;
  return String(value);
}

function spacedSigned(value) {
  return value < 0 ? `- ${Math.abs(value)}` : String(value);
}

function contextualLowercaseVariables(contextLatex = []) {
  const variables = new Set();
  for (const latex of contextLatex || []) {
    const text = String(latex || '').replace(/\\[A-Za-z]+/g, ' ');
    for (const match of text.matchAll(/[a-z]/g)) {
      variables.add(match[0]);
    }
  }
  return variables;
}

function isSuspiciousOperationLatex(latex) {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (/\\(?:ldots|cdots)\b/.test(normalized)) return true;
  return /^\\(?:times|div)\s+\d+\s+\d+\s+\\(?:times|div)\s+\d+\s+\d+$/.test(normalized);
}

function candidatesForRecognition(segmentation) {
  const byId = new Map();
  const keepProfiles = new Set([
    'parent',
    'loose',
    'strict',
    'temporal',
    'row-line',
    'raw-row-line',
    'fraction-stack-line',
    'projection-line',
    'dbnet-parent',
    'dbnet-line'
  ]);

  for (const candidate of segmentation.candidates || []) {
    const profiles = candidate.profiles || [];
    const keep = profiles.some((profile) => keepProfiles.has(profile)) ||
      segmentation.selected.some((selected) => selected.candidateId === candidate.candidateId);
    if (keep) byId.set(candidate.candidateId, candidate);
  }

  if (byId.size === 0) {
    for (const candidate of segmentation.selected || []) {
      byId.set(candidate.candidateId, candidate);
    }
  }

  return [...byId.values()];
}

function recognitionBaselineCover(primaryCover = [], deterministicCover = []) {
  const primary = Array.isArray(primaryCover) ? primaryCover : [];
  const deterministic = Array.isArray(deterministicCover) ? deterministicCover : [];
  if (!primary.length || !deterministic.length) return primary;
  if (deterministic.length <= primary.length) return primary;
  if (coverStrokeKey(primary) !== coverStrokeKey(deterministic)) return primary;
  if (!deterministic.every(isUsableBaselineLineCandidate)) return primary;
  return deterministic;
}

function coverStrokeKey(cover = []) {
  return [...new Set(
    cover.flatMap((candidate) => candidate.strokeIds || []).map(String)
  )].sort().join('|');
}

function isUsableBaselineLineCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'projection-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ));
}

function priorLineContextLatex(entry, candidatePredictions, limit = 3) {
  if (!entry?.tightBbox) return [];
  const chosen = [];
  const priorCandidates = (candidatePredictions || [])
    .filter((candidate) => candidate !== entry)
    .filter(isLineContextCandidate)
    .filter((candidate) => candidate.tightBbox?.yMax <= entry.tightBbox.yMin + 2)
    .sort((a, b) => (
      (b.tightBbox?.yMin ?? 0) - (a.tightBbox?.yMin ?? 0) ||
      (b.evidenceScore ?? 0) - (a.evidenceScore ?? 0)
    ));

  for (const candidate of priorCandidates) {
    if (chosen.some((selected) => verticalOverlapRatio(selected.tightBbox, candidate.tightBbox) >= 0.45)) {
      continue;
    }
    chosen.push(candidate);
    if (chosen.length >= limit) break;
  }

  return uniqueStrings(
    chosen
      .sort((a, b) => (a.tightBbox?.yMin ?? 0) - (b.tightBbox?.yMin ?? 0))
      .map((candidate) => candidate.latex)
      .filter(Boolean)
  );
}

function isLineContextCandidate(entry) {
  const profiles = entry?.profiles || [];
  if (!entry?.latex || !entry?.tightBbox) return false;
  if (profiles.some((profile) => profile.includes('parent') || profile === 'temporal')) return false;
  return profiles.some((profile) => (
    profile === 'row-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'dbnet-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ));
}

function verticalOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const overlap = Math.max(0, Math.min(a.yMax, b.yMax) - Math.max(a.yMin, b.yMin));
  const smaller = Math.min(Math.max(1, a.yMax - a.yMin), Math.max(1, b.yMax - b.yMin));
  return overlap / smaller;
}

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  if (typeof signal.throwIfAborted === 'function') {
    signal.throwIfAborted();
  }
  const error = new Error('Recognition aborted');
  error.name = 'AbortError';
  throw error;
}

function recognitionFailureFromError(error, extra = {}) {
  if (isAbortError(error)) throw error;
  return {
    model: extra.model,
    latex: '',
    candidates: [],
    confidence: 0,
    failed: true,
    error: error instanceof Error ? error.message : String(error),
    elapsedSeconds: 0,
    ...extra
  };
}

function isAbortError(error) {
  return error?.name === 'AbortError';
}

function uniqueStrings(values) {
  return [...new Set((values || []).map(String))];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
