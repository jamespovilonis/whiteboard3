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
    initialRecognitionConcurrency = 1,
    signal = null,
    recognizeLine = recognizeLineImage
  } = options;

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
    problemLatex,
    previousLatex
  });
  const deterministicSegmentation = detection.detections?.length
    ? segmentMathLines(strokes, {
        answerBox,
        detections: [],
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
    baselineCandidateIds: new Set((baselineCover || []).map((candidate) => candidate.candidateId))
  };
  const candidatePredictions = await mapWithConcurrency(
    candidatesToRecognize,
    initialRecognitionConcurrency,
    (candidate) => recognizeInitialCandidate(candidate, {
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
    })
  );
  throwIfAborted(signal);

  const evidenceByCandidateId = new Map(
    candidatePredictions.map((entry) => [entry.candidateId, entry.evidenceScore])
  );
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
    entry.timing.semanticElapsedSeconds = finiteSeconds(semantic.elapsedSeconds);
    entry.timing.submitToFinalPredictionSeconds = secondsSince(pipelineStartedAt);
    entry.evidenceScore += Number(semanticEntry.semanticScore) || 0;
    evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
    if (shouldUseSemanticLatex(entry.latex, semanticEntry)) {
      entry.latex = semanticEntry.bestLatex;
    }
  }

  let selected = recognizeAlternatives
    ? selectCandidateCover(candidatesToRecognize, {
        scoreByCandidateId: evidenceByCandidateId,
        baselineCandidates: baselineCover
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
      entry.timing.contextualSemanticElapsedSeconds = finiteSeconds(contextualCandidateSemantic.elapsedSeconds);
      entry.timing.submitToFinalPredictionSeconds = secondsSince(pipelineStartedAt);
      entry.evidenceScore += evidenceDelta;
      evidenceByCandidateId.set(entry.candidateId, entry.evidenceScore);
      if (shouldUseSemanticLatex(entry.latex, contextualEntry)) {
        entry.latex = contextualEntry.bestLatex;
      }
    }
    if (recognizeAlternatives) {
      selected = selectCandidateCover(candidatesToRecognize, {
        scoreByCandidateId: evidenceByCandidateId,
        baselineCandidates: baselineCover
      });
    }
  }

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
  if (chunkFallback) {
    for (const candidate of selected) {
      throwIfAborted(signal);
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
  if (retryRasterHeights?.length) {
    for (const candidate of selected) {
      throwIfAborted(signal);
      const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
      if (!entry || entry.skippedRecognition || !predictionNeedsRetry(entry.prediction)) continue;
      const retry = await retrySelectedLineRecognition(candidate, {
        apiUrl,
        model,
        timeoutMs,
        problemLatex,
        rasterPadding,
        retryRasterHeights,
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
  if (chunkFallback) {
    for (const candidate of selected) {
      throwIfAborted(signal);
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

  const recognizedLines = selected.map((candidate, index) => {
    const entry = candidatePredictions.find((item) => item.candidateId === candidate.candidateId);
    return {
      ...entry,
      lineIndex: index,
      selected: true
    };
  });

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
  if (semanticRetryRasterHeights?.length && semanticScoring && !selectedLineSemantic.failed) {
    for (const line of recognizedLines) {
      throwIfAborted(signal);
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
    line.timing = {
      ...(line.timing || {}),
      sequentialSemanticElapsedSeconds: finiteSeconds(selectedLineSemantic.elapsedSeconds),
      submitToFinalPredictionSeconds: secondsSince(pipelineStartedAt)
    };
    const operationRepair = repairStandaloneOperationLatex(line.latex, lineSemantic, { problemLatex });
    if (operationRepair) {
      line.ocrRepair = {
        source: 'standalone-operation',
        originalLatex: line.latex,
        repairedLatex: operationRepair
      };
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
    if (shouldUseSemanticLatex(line.latex, lineSemantic)) {
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
    latexLines: recognizedLines.map((line) => line.acceptedLatex),
    latex: recognizedLines.map((line) => line.acceptedLatex).filter(Boolean).join(' \\\\ '),
    timing: {
      totalElapsedSeconds: secondsSince(pipelineStartedAt)
    }
  };
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
  let elapsedSeconds = 0;

  try {
    for (const entry of scorablePredictions) {
      const sameAnswerContext = priorLineContextLatex(entry, scorablePredictions);
      if (sameAnswerContext.length === 0) continue;

      const payload = await scoreSemantics({
        problemLatex,
        problemMetadata,
        previousLatex: [
          ...(previousLatex || []),
          ...sameAnswerContext
        ].filter(Boolean),
        candidateGroups: [{
          candidateId: entry.candidateId,
          latex: entry.latex,
          candidates: semanticCandidateAlternatives(entry.candidates, semanticCandidateLimit),
          elapsedSeconds: entry.prediction?.elapsedSeconds
        }]
      }, { apiUrl, timeoutMs, signal });
      throwIfAborted(signal);

      elapsedSeconds += Number(payload?.elapsedSeconds) || 0;
      const score = (payload?.candidateScores || [])[0];
      if (score) {
        candidateScores.push({
          ...score,
          sameAnswerContext
        });
      }
    }

    return {
      source: 'semantic-service',
      failed: false,
      elapsedSeconds,
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

  try {
    for (const line of scorableLines) {
      const payload = await scoreSemantics({
        problemLatex,
        problemMetadata,
        previousLatex: contextLatex.slice(),
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
      const score = (payload?.candidateScores || [])[0];
      if (score) {
        const lineScore = {
          ...score,
          lineIndex: line.lineIndex,
          candidateId: line.candidateId
        };
        lineScores.push(lineScore);
        const trustedLatex = shouldUseSemanticLatex(line.latex, lineScore)
          ? lineScore.bestLatex
          : line.latex;
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
  signal,
  recognizeLine
}) {
  const attempts = [];
  const seenHeights = new Set((skipRasterHeights || [])
    .map((height) => Number(height))
    .filter((height) => Number.isFinite(height) && height > 0));
  for (const height of retryRasterHeights || []) {
    throwIfAborted(signal);
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
  }
  if (
    attempts.length &&
    attempts.every((attempt) => predictionNeedsRetry(attempt)) &&
    candidateNeedsExtendedTimeout(candidate) &&
    Number(extendedTimeoutMs) > Number(timeoutMs)
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
  if (chunkFallback && attempts.length && attempts.every((attempt) => predictionNeedsRetry(attempt))) {
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

function initialRecognitionSkipReason(candidate, context = {}) {
  if (!context.enabled || !candidate) return '';
  if (shouldSkipSingleStrokeAlternative(candidate, context)) return 'single-stroke-alternative';
  if (shouldDeferContainedNonstructuralAlternative(candidate, context)) {
    return 'contained-nonstructural-alternative';
  }
  return '';
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
    entry.ocrRepair = selectedLine.ocrRepair || null;
    entry.timing = {
      ...(entry.timing || {}),
      ...(selectedLine.timing || {}),
      submitToFinalPredictionSeconds: selectedLine.timing?.submitToFinalPredictionSeconds ?? secondsSince(pipelineStartedAt)
    };
    selectedLine.debugLabel = entry.debugLabel;
  }
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
