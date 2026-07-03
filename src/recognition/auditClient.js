export const DEFAULT_AUDIT_NORMAL_SAMPLE_RATE = 0.10;
export const RECOGNITION_AUDIT_PROMPT_VERSION = 'recognition-audit-v2';
const MAX_COMPACT_STROKE_POINTS = 96;
const previousAuditIdByProblemId = new Map();

export function getRecognitionAuditDecision(result = {}, options = {}) {
  const triggerReasons = [];
  const grading = result?.grading || null;
  const problemStatus = grading?.result?.problemStatus || '';
  const lines = Array.isArray(result?.lines) ? result.lines : [];

  if (!grading || grading.failed) {
    triggerReasons.push('missing_grading');
  }
  if (['incorrect', 'incomplete', 'not_started'].includes(problemStatus)) {
    triggerReasons.push(`problem_status_${problemStatus}`);
  }
  if (hasUnreadLine(result)) {
    triggerReasons.push('unread_or_empty_line');
  }
  if (hasLineSegmentationEmpty(result)) {
    triggerReasons.push('line_segmentation_empty');
  }
  if (hasOcrFailure(result)) {
    triggerReasons.push('ocr_failure_or_timeout');
  }
  if (hasLowConfidenceLine(result)) {
    triggerReasons.push('low_confidence');
  }
  if (hasCorrectAnswerWithInvalidStep(grading)) {
    triggerReasons.push('correct_answer_with_invalid_step');
  }
  if (hasDetachedOperationAnnotation(result)) {
    triggerReasons.push('detached_operation_annotation');
  }
  if (hasCandidateSelectionConflict(result)) {
    triggerReasons.push('candidate_selection_conflict');
  }

  if (triggerReasons.length > 0) {
    return {
      shouldAudit: true,
      sampled: false,
      triggerReasons: unique(triggerReasons)
    };
  }

  const sampleRate = normalizedSampleRate(
    options.normalSampleRate ?? configuredNormalSampleRate()
  );
  const sampleKey = String(options.sampleKey || options.inputSignature || result?.realtime?.inputSignature || result?.latex || '');
  const sampled = problemStatus === 'correct' && deterministicSample(sampleKey, sampleRate);
  return {
    shouldAudit: sampled,
    sampled,
    triggerReasons: sampled ? ['normal_sample'] : []
  };
}

export async function enqueueRecognitionAudit(payload, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const url = `${apiUrl}/audit-recognition`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail || `HTTP ${response.status} from ${url}`);
  }
  if (body?.auditId && payload?.problemId) {
    previousAuditIdByProblemId.set(String(payload.problemId), String(body.auditId));
  }
  return body || {};
}

export async function getRecognitionAuditStatus(auditId, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const encodedAuditId = encodeURIComponent(String(auditId || ''));
  const url = `${apiUrl}/audit-recognition/${encodedAuditId}`;
  const response = await fetch(url);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail || `HTTP ${response.status} from ${url}`);
  }
  return body || {};
}

export async function addRecognitionAuditNote(payload, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const url = `${apiUrl}/audit-recognition-note`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail || `HTTP ${response.status} from ${url}`);
  }
  return body || {};
}

export function buildRecognitionAuditPayload({
  problem = {},
  result = {},
  strokes = [],
  inputSignature = '',
  triggerReasons = []
} = {}) {
  const problemId = problem.id || null;
  const previousAuditId = problemId ? previousAuditIdByProblemId.get(String(problemId)) || null : null;
  return {
    problemId,
    problemLatex: problem.latex || '',
    problemMetadata: problem.metadata || {},
    problemBox: clonePlain(problem.problemBox),
    answerBox: clonePlain(problem.answerBox),
    inputSignature,
    attemptId: buildAttemptId(problemId, inputSignature),
    promptVersion: RECOGNITION_AUDIT_PROMPT_VERSION,
    previousAuditId,
    triggerReasons: triggerReasons.slice(),
    strokes: compactStrokes(strokes),
    fastResult: compactRecognitionResult(result)
  };
}

export function compactRecognitionResult(result = {}) {
  return {
    latex: result.latex || '',
    latexLines: Array.isArray(result.latexLines) ? result.latexLines.slice() : [],
    grading: clonePlain(result.grading || null),
    timing: clonePlain(result.timing || null),
    detection: clonePlain(result.detection || null),
    semantic: clonePlain(stripCandidateImages(result.semantic || null)),
    realtime: clonePlain(result.realtime || null),
    lines: (result.lines || []).map(compactLine),
    candidatePredictions: (result.candidatePredictions || []).map(compactLine),
    selectionSummary: compactSelectionSummary(result),
    annotationAttachments: compactAnnotationAttachments(result),
    segmentation: {
      selected: (result.segmentation?.selected || []).map(compactCandidate),
      candidates: (result.segmentation?.candidates || []).map(compactCandidate),
      partitions: clonePlain(result.segmentation?.partitions || {}),
      parentCandidateId: result.segmentation?.parentCandidateId || null,
      ocrSelectedCandidateIds: (result.segmentation?.ocrSelectedCandidateIds || []).slice()
    }
  };
}

export function buildProblemInputAuditPayload({
  mode = '',
  problemType = '',
  recognizedLatex = '',
  result = {},
  strokes = [],
  answerBox = null,
  inputSignature = '',
  triggerReasons = []
} = {}) {
  const latex = String(recognizedLatex || result?.latex || '').trim();
  const normalizedResult = {
    ...result,
    latex,
    latexLines: Array.isArray(result?.latexLines) && result.latexLines.length
      ? result.latexLines
      : (latex ? [latex] : []),
    grading: null
  };
  return {
    problemId: `problem-input-${String(mode || problemType || 'custom').replace(/[^a-z0-9_-]/gi, '-')}`,
    problemLatex: '',
    problemMetadata: {
      auditSubject: 'problem-input',
      mode,
      problemType,
      source: 'user-handwriting'
    },
    problemBox: clonePlain(answerBox),
    answerBox: clonePlain(answerBox),
    inputSignature,
    attemptId: buildAttemptId(`problem-input-${mode || problemType || 'custom'}`, inputSignature),
    promptVersion: RECOGNITION_AUDIT_PROMPT_VERSION,
    previousAuditId: null,
    triggerReasons: triggerReasons.slice(),
    strokes: compactStrokes(strokes),
    fastResult: compactRecognitionResult(normalizedResult)
  };
}

export function normalizeProblemInputRecognitionResult(result = {}) {
  const lines = Array.isArray(result?.lines) ? result.lines : [];
  const normalizedLines = lines.map((line) => {
    const accepted = chooseProblemInputLineLatex(line);
    return {
      ...line,
      latex: accepted,
      acceptedLatex: accepted,
      ocrLatex: accepted || line?.ocrLatex || ''
    };
  });
  let latexLines = normalizedLines
    .map((line) => String(line.acceptedLatex || line.latex || '').trim())
    .filter((latex) => latex && !containsSplitLineNoise(latex));

  if (!latexLines.length && Array.isArray(result?.latexLines)) {
    latexLines = result.latexLines
      .map((latex) => String(latex || '').trim())
      .filter((latex) => latex && !containsSplitLineNoise(latex));
  }

  if (!latexLines.length) {
    const fallback = firstCleanProblemInputCandidate(result);
    if (fallback) latexLines = [fallback];
  }

  return {
    ...result,
    lines: normalizedLines,
    latexLines,
    latex: latexLines.join(' \\\\ ')
  };
}

export function containsProblemInputSplitLineNoise(latex = '') {
  return containsSplitLineNoise(latex);
}

export function hasCorrectAnswerWithInvalidStep(grading = null) {
  const steps = Array.isArray(grading?.steps) ? grading.steps : [];
  if (steps.length === 0) return false;
  const invalidIndexes = steps
    .filter((step) => step?.classification === 'invalid_step')
    .map((step, index) => Number.isFinite(Number(step.lineIndex)) ? Number(step.lineIndex) : index);
  if (invalidIndexes.length === 0) return false;

  const solutionIndexes = steps
    .filter((step) => (
      step?.solutionCoverage === 'full' ||
      (Array.isArray(step?.matchedSolutions) && step.matchedSolutions.length > 0)
    ))
    .map((step, index) => Number.isFinite(Number(step.lineIndex)) ? Number(step.lineIndex) : index);
  const hasCorrectAnswerSet = grading?.result?.problemStatus === 'correct' ||
    solutionIndexes.length > 0 ||
    (
      Array.isArray(grading?.result?.foundSolutions) &&
      grading.result.foundSolutions.length > 0 &&
      Array.isArray(grading?.result?.missingSolutions) &&
      grading.result.missingSolutions.length === 0
    );
  if (!hasCorrectAnswerSet) return false;
  const lastSolutionIndex = solutionIndexes.length ? Math.max(...solutionIndexes) : Math.max(...steps.map((_, index) => index));
  return invalidIndexes.some((index) => index < lastSolutionIndex || steps.length > 1);
}

export function deterministicSample(key, sampleRate) {
  const rate = normalizedSampleRate(sampleRate);
  if (rate <= 0) return false;
  if (rate >= 1) return true;
  const hash = stableHash32(String(key || ''));
  return hash / 0xffffffff < rate;
}

function compactLine(line = {}) {
  return {
    lineIndex: line.lineIndex ?? null,
    candidateId: line.candidateId || null,
    debugLabel: line.debugLabel || null,
    selected: Boolean(line.selected),
    profiles: Array.isArray(line.profiles) ? line.profiles.slice() : [],
    strokeIds: Array.isArray(line.strokeIds) ? line.strokeIds.map(String) : [],
    tightBbox: clonePlain(line.tightBbox || null),
    latex: line.latex || '',
    acceptedLatex: line.acceptedLatex || '',
    ocrLatex: line.ocrLatex || '',
    excludedFromGrading: Boolean(line.excludedFromGrading),
    candidates: (line.candidates || []).slice(0, 5).map(compactOcrCandidate),
    prediction: compactPrediction(line.prediction || null),
    grading: clonePlain(line.grading || null),
    contextualSemantic: clonePlain(line.contextualSemantic || null),
    sequentialSemantic: clonePlain(line.sequentialSemantic || null),
    ocrRepair: clonePlain(line.ocrRepair || null),
    evidenceScore: Number.isFinite(Number(line.evidenceScore)) ? Number(line.evidenceScore) : null,
    selectedLineIndex: line.selectedLineIndex ?? null,
    discarded: Boolean(line.discarded),
    selectionReason: line.selectionReason || selectionReasonForLine(line),
    timing: clonePlain(line.timing || null),
    realtimeStatus: line.realtimeStatus || null,
    provisional: Boolean(line.provisional),
    annotationAttachment: compactLineAnnotationAttachment(line)
  };
}

function compactAnnotationAttachments(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  const linesByCandidateId = new Map(lines
    .filter((line) => line?.candidateId)
    .map((line) => [String(line.candidateId), line]));
  const equationLines = lines.filter((line) => line?.tightBbox && /[=]/.test(String(line?.acceptedLatex || line?.latex || '')));

  const attachments = lines
    .map((line) => buildAnnotationAttachment(line, { linesByCandidateId, equationLines }))
    .filter(Boolean);

  const pairGroups = new Map();
  for (const attachment of attachments) {
    const pairKey = [
      attachment.targetLineIndex ?? 'na',
      attachment.operand || attachment.operatorLatex || '',
    ].join('|');
    if (!pairGroups.has(pairKey)) pairGroups.set(pairKey, []);
    pairGroups.get(pairKey).push(attachment);
  }
  for (const [pairKey, group] of pairGroups.entries()) {
    const hasLeft = group.some((item) => item.equationSide === 'left');
    const hasRight = group.some((item) => item.equationSide === 'right');
    const hasBoth = group.some((item) => item.equationSide === 'both');
    if (!(hasBoth || (hasLeft && hasRight))) continue;
    for (const attachment of group) {
      attachment.pairedAnnotationId = `pair:${pairKey}`;
    }
  }

  return attachments;
}

function compactLineAnnotationAttachment(line = {}) {
  const attachment = buildAnnotationAttachment(line, {
    linesByCandidateId: new Map(),
    equationLines: []
  });
  return attachment || null;
}

function buildAnnotationAttachment(line = {}, { linesByCandidateId, equationLines } = {}) {
  const repair = line?.ocrRepair || null;
  const source = String(repair?.source || '');
  const rawLatex = String(
    repair?.originalLatex ||
    line?.ocrLatex ||
    line?.acceptedLatex ||
    line?.latex ||
    ''
  ).trim();
  const operand = String(repair?.operand || detachedOperationOperand(rawLatex) || '').trim();
  const annotationBbox = normalizedBox(repair?.annotationBbox || line?.tightBbox || null);
  if (!annotationBbox) return null;

  const derivedFromRepair = source === 'geometry-operation-annotation';
  const inferredDetached = looksLikeDetachedOperationAnnotation(rawLatex);
  if (!derivedFromRepair && !inferredDetached) return null;

  const anchor = resolveAttachmentAnchor(line, { linesByCandidateId, equationLines, repair });
  const repairedLatex = String(repair?.repairedLatex || '').trim();
  const equationSide = (
    derivedFromRepair && hasSymmetricOperationRepair(repairedLatex, operand)
      ? 'both'
      : classifyAnnotationSide(annotationBbox, anchor?.tightBbox || null)
  );
  const operatorLatex = rawLatex || String(line?.acceptedLatex || line?.latex || '').trim();
  return {
    operatorLatex,
    operand: operand || null,
    targetLineIndex: Number.isFinite(Number(anchor?.lineIndex)) ? Number(anchor.lineIndex) : null,
    targetCandidateId: anchor?.candidateId || null,
    equationSide,
    pairedAnnotationId: null,
    attachmentConfidence: derivedFromRepair ? 0.95 : 0.7,
    source: derivedFromRepair ? 'geometry-operation-annotation' : 'detached-operation-heuristic',
    anchorBbox: clonePlain(anchor?.tightBbox || null),
    annotationBbox: clonePlain(annotationBbox),
    repairedLatex: repairedLatex || null,
  };
}

function resolveAttachmentAnchor(line = {}, { linesByCandidateId, equationLines, repair } = {}) {
  const anchorCandidateId = repair?.anchorCandidateId ? String(repair.anchorCandidateId) : '';
  if (anchorCandidateId && linesByCandidateId?.has(anchorCandidateId)) {
    return linesByCandidateId.get(anchorCandidateId) || null;
  }
  if (line?.tightBbox && Array.isArray(equationLines) && equationLines.length) {
    return nearestEquationLine(line.tightBbox, equationLines);
  }
  return null;
}

function nearestEquationLine(annotationBbox, equationLines = []) {
  const annotationCenterY = bboxYCenter(annotationBbox);
  return equationLines
    .map((line) => {
      const anchorBox = normalizedBox(line?.tightBbox || null);
      if (!anchorBox) return null;
      const verticalGap = Math.max(0, Math.max(anchorBox.yMin - annotationBbox.yMax, annotationBbox.yMin - anchorBox.yMax));
      const horizontalSpan = Math.max(0, Math.min(anchorBox.xMax, annotationBbox.xMax) - Math.max(anchorBox.xMin, annotationBbox.xMin));
      const score = verticalGap + Math.abs(annotationCenterY - bboxYCenter(anchorBox)) * 0.2 - horizontalSpan * 0.01;
      return { line, score };
    })
    .filter(Boolean)
    .sort((a, b) => a.score - b.score)[0]?.line || null;
}

function classifyAnnotationSide(annotationBbox, anchorBbox) {
  const annotation = normalizedBox(annotationBbox);
  const anchor = normalizedBox(anchorBbox);
  if (!annotation || !anchor) return 'unknown';

  const anchorMidX = (anchor.xMin + anchor.xMax) / 2;
  const width = Math.max(1, anchor.xMax - anchor.xMin);
  const leftSpan = annotation.xMin <= anchorMidX - width * 0.08;
  const rightSpan = annotation.xMax >= anchorMidX + width * 0.08;
  const centerX = (annotation.xMin + annotation.xMax) / 2;
  const spanningMidpoint = annotation.xMin <= anchorMidX && annotation.xMax >= anchorMidX;

  if (spanningMidpoint || (leftSpan && rightSpan)) return 'both';
  if (centerX < anchorMidX) return 'left';
  if (centerX > anchorMidX) return 'right';
  return 'both';
}

function hasSymmetricOperationRepair(repairedLatex = '', operand = '') {
  const normalized = String(repairedLatex || '').replace(/\s+/g, ' ').trim();
  const normalizedOperand = String(operand || '').trim();
  if (!normalized || !normalizedOperand) return false;
  return normalized === `\\times ${normalizedOperand} \\times ${normalizedOperand}` ||
    normalized === `\\cdot ${normalizedOperand} \\cdot ${normalizedOperand}`;
}

function bboxYCenter(box) {
  const normalized = normalizedBox(box);
  if (!normalized) return 0;
  return (normalized.yMin + normalized.yMax) / 2;
}

function compactSelectionSummary(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  const selectedIds = new Set(lines.map((line) => line?.candidateId).filter(Boolean));
  const predictions = Array.isArray(result.candidatePredictions) ? result.candidatePredictions : [];
  const highConfidenceDiscarded = predictions
    .filter((entry) => entry?.candidateId && !selectedIds.has(entry.candidateId))
    .filter(isHighConfidenceAlternative)
    .slice(0, 8)
    .map((entry) => ({
      candidateId: entry.candidateId || null,
      latex: entry.acceptedLatex || entry.latex || entry.ocrLatex || '',
      tightBbox: clonePlain(entry.tightBbox || null),
      strokeIds: Array.isArray(entry.strokeIds) ? entry.strokeIds.map(String) : [],
      evidenceScore: Number.isFinite(Number(entry.evidenceScore)) ? Number(entry.evidenceScore) : null,
      grading: clonePlain(entry.grading || null),
      prediction: compactPrediction(entry.prediction || null),
    }));

  return {
    selectedCandidateIds: [...selectedIds],
    selectedLines: lines.map((line) => ({
      candidateId: line?.candidateId || null,
      lineIndex: line?.lineIndex ?? null,
      latex: line?.acceptedLatex || line?.latex || '',
      tightBbox: clonePlain(line?.tightBbox || null),
      strokeIds: Array.isArray(line?.strokeIds) ? line.strokeIds.map(String) : [],
      selectionReason: selectionReasonForLine(line || {}),
    })),
    highConfidenceDiscarded,
    rescueSummary: clonePlain(result.selectionRescue || []),
  };
}

function isHighConfidenceAlternative(entry = {}) {
  const grading = entry.grading || entry.semantic?.grading || entry.contextualSemantic?.grading || entry.sequentialSemantic?.grading || null;
  if (grading?.solutionCoverage === 'full' || grading?.solutionCoverage === 'partial') return true;
  const confidence = Number(entry.prediction?.confidence ?? entry.prediction?.top?.confidence);
  if (Number.isFinite(confidence) && confidence >= 0.6) return true;
  const score = Number(entry.prediction?.top?.score);
  if (Number.isFinite(score) && score >= 1.5) return true;
  const evidenceScore = Number(entry.evidenceScore);
  return Number.isFinite(evidenceScore) && evidenceScore >= 8;
}

function selectionReasonForLine(line = {}) {
  if (line.ocrRepair?.source) return `ocr_repair:${line.ocrRepair.source}`;
  const grading = line.grading || line.sequentialSemantic?.grading || line.contextualSemantic?.grading || null;
  if (grading?.solutionCoverage === 'full') return 'grading_full_solution';
  if (grading?.solutionCoverage === 'partial') return 'grading_partial_solution';
  if (grading?.classification === 'valid_step') return 'grading_valid_step';
  if (line.contextualSemantic) return 'contextual_semantic';
  if (line.sequentialSemantic) return 'sequential_semantic';
  if (line.semantic) return 'semantic';
  return 'ocr_evidence';
}

function compactCandidate(candidate = {}) {
  return {
    candidateId: candidate.candidateId || null,
    profiles: Array.isArray(candidate.profiles) ? candidate.profiles.slice() : [],
    strokeIds: Array.isArray(candidate.strokeIds) ? candidate.strokeIds.map(String) : [],
    tightBbox: clonePlain(candidate.tightBbox || null),
    conflicts: clonePlain(candidate.conflicts || [])
  };
}

function compactPrediction(prediction = null) {
  if (!prediction) return null;
  return {
    latex: prediction.latex || '',
    top: compactOcrCandidate(prediction.top || null),
    candidates: (prediction.candidates || []).slice(0, 5).map(compactOcrCandidate),
    confidence: Number.isFinite(Number(prediction.confidence)) ? Number(prediction.confidence) : null,
    failed: Boolean(prediction.failed),
    timedOut: Boolean(prediction.timedOut),
    error: prediction.error || null,
    elapsedSeconds: Number.isFinite(Number(prediction.elapsedSeconds)) ? Number(prediction.elapsedSeconds) : null
  };
}

function compactOcrCandidate(candidate = null) {
  if (!candidate) return null;
  return {
    latex: candidate.latex || '',
    score: Number.isFinite(Number(candidate.score)) ? Number(candidate.score) : null,
    confidence: Number.isFinite(Number(candidate.confidence)) ? Number(candidate.confidence) : null
  };
}

function chooseProblemInputLineLatex(line = {}) {
  const current = String(line?.acceptedLatex || line?.latex || '').trim();
  if (current && !containsSplitLineNoise(current)) return current;
  const candidates = Array.isArray(line?.candidates) ? line.candidates : [];
  const candidate = candidates
    .slice(0, 5)
    .map((item) => String(item?.latex || '').trim())
    .find((latex) => latex && !containsSplitLineNoise(latex));
  return candidate || '';
}

function firstCleanProblemInputCandidate(result = {}) {
  const candidatePredictions = Array.isArray(result?.candidatePredictions)
    ? result.candidatePredictions
    : [];
  for (const prediction of candidatePredictions) {
    const latex = chooseProblemInputLineLatex(prediction);
    if (latex) return latex;
  }
  return '';
}

function containsSplitLineNoise(latex = '') {
  return String(latex || '').includes('//');
}

function compactStrokes(strokes = []) {
  return (strokes || []).filter(Boolean).map((stroke) => ({
    id: String(stroke.id || ''),
    startTime: stroke.startTime ?? null,
    endTime: stroke.endTime ?? null,
    points: compactPoints(stroke.points, { includeTime: true, normalized: true }),
    rawPoints: compactPoints(stroke.rawPoints),
    outlinePoints: compactPoints(stroke.outlinePoints),
    color: stroke.color || '#000000',
    canvasBbox: clonePlain(stroke.canvasBbox || null),
    bbox: clonePlain(stroke.bbox || null),
    relationsToPrev: compactRelations(stroke.relationsToPrev || null)
  }));
}

function compactPoints(points = [], options = {}) {
  if (!Array.isArray(points)) return [];
  return downsamplePoints(points, MAX_COMPACT_STROKE_POINTS).map((point) => {
    const compact = {
      x: quantize(point?.x, options.normalized ? 0.0001 : 0.01),
      y: quantize(point?.y, options.normalized ? 0.0001 : 0.01),
      pressure: Number.isFinite(Number(point?.pressure))
        ? quantize(point.pressure, 0.001)
        : null
    };
    if (options.includeTime && Number.isFinite(Number(point?.t))) {
      compact.t = quantize(point.t, 1);
    }
    return compact;
  });
}

function compactRelations(relations = null) {
  if (!relations || typeof relations !== 'object') return null;
  return {
    dx: quantize(relations.dx, 0.0001),
    dy: quantize(relations.dy, 0.0001),
    dt: quantize(relations.dt, 1),
    overlapRatio: quantize(relations.overlapRatio, 0.0001)
  };
}

function downsamplePoints(points, maxPoints) {
  if (!Array.isArray(points) || points.length <= maxPoints) return points || [];
  if (maxPoints <= 2) return points.slice(0, maxPoints);

  const out = [];
  const lastIndex = points.length - 1;
  for (let index = 0; index < maxPoints; index += 1) {
    const sourceIndex = Math.round((index / (maxPoints - 1)) * lastIndex);
    out.push(points[sourceIndex]);
  }
  return out;
}

function quantize(value, step) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const quantum = Number(step) || 1;
  return Math.round(numeric / quantum) * quantum;
}

function hasUnreadLine(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  if ((Array.isArray(result.latexLines) && result.latexLines.length === 0) && lines.length > 0) return true;
  return lines.some((line) => (
    !String(line?.acceptedLatex || line?.latex || '').trim() ||
    line?.realtimeStatus === 'unread'
  ));
}

function hasLineSegmentationEmpty(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  return lines.some((line) => {
    const latex = String(line?.acceptedLatex || line?.latex || line?.ocrLatex || '').trim();
    return !latex && Array.isArray(line?.strokeIds) && line.strokeIds.length > 0;
  });
}

function hasOcrFailure(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  return lines.some((line) => Boolean(line?.prediction?.failed || line?.prediction?.timedOut));
}

function hasLowConfidenceLine(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  return lines.some((line) => {
    const top = line?.prediction?.top || (Array.isArray(line?.candidates) ? line.candidates[0] : null) || {};
    const confidence = Number(top.confidence ?? line?.prediction?.confidence);
    if (Number.isFinite(confidence) && confidence < 0.55) return true;
    const score = Number(top.score);
    if (Number.isFinite(score) && score <= -0.25) return true;
    const evidenceScore = Number(line?.evidenceScore);
    return Number.isFinite(evidenceScore) && evidenceScore <= -1;
  });
}

function hasDetachedOperationAnnotation(result = {}) {
  const values = [
    ...(Array.isArray(result.lines) ? result.lines : []),
    ...(Array.isArray(result.candidatePredictions) ? result.candidatePredictions : []),
  ];
  return values.some((line) => (
    line?.ocrRepair?.source === 'geometry-operation-annotation' ||
    looksLikeDetachedOperationAnnotation(line?.acceptedLatex || line?.latex || line?.ocrLatex || '')
  ));
}

function hasCandidateSelectionConflict(result = {}) {
  const selected = new Set((result.lines || []).map((line) => line?.candidateId).filter(Boolean));
  return (result.candidatePredictions || []).some((entry) => {
    if (!entry?.candidateId || selected.has(entry.candidateId)) return false;
    const grading = entry.grading || entry.semantic?.grading || entry.contextualSemantic?.grading || null;
    if (grading?.solutionCoverage !== 'full') return false;
    return (result.lines || []).some((line) => boxesOverlap(line?.tightBbox, entry.tightBbox) >= 0.55);
  });
}

function detachedOperationOperand(latex = '') {
  const normalized = String(latex || '')
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

function looksLikeDetachedOperationAnnotation(latex = '') {
  const normalized = String(latex || '').replace(/\s+/g, ' ').trim();
  if (!normalized || normalized.includes('=')) return false;
  if (/^(?:\\times|\\cdot|\*)?\s*\d{1,3}\s*(?:\.|\*|\\times|\\cdot)?$/.test(normalized)) return true;
  if (/^(?:\.|\*|\\times|\\cdot)\s*\d{1,3}$/.test(normalized)) return true;
  return /^(?:x|X|\\times)\s*_\s*\{\s*\d{1,3}\s*\}/.test(normalized);
}

function boxesOverlap(left, right) {
  const a = normalizedBox(left);
  const b = normalizedBox(right);
  if (!a || !b) return 0;
  const xMin = Math.max(a.xMin, b.xMin);
  const yMin = Math.max(a.yMin, b.yMin);
  const xMax = Math.min(a.xMax, b.xMax);
  const yMax = Math.min(a.yMax, b.yMax);
  if (xMax <= xMin || yMax <= yMin) return 0;
  const overlapArea = (xMax - xMin) * (yMax - yMin);
  const smallerArea = Math.min((a.xMax - a.xMin) * (a.yMax - a.yMin), (b.xMax - b.xMin) * (b.yMax - b.yMin));
  return smallerArea > 0 ? overlapArea / smallerArea : 0;
}

function normalizedBox(value) {
  if (!value || typeof value !== 'object') return null;
  const box = {
    xMin: Number(value.xMin),
    yMin: Number(value.yMin),
    xMax: Number(value.xMax),
    yMax: Number(value.yMax)
  };
  if (!Object.values(box).every(Number.isFinite)) return null;
  if (box.xMax <= box.xMin || box.yMax <= box.yMin) return null;
  return box;
}

function configuredNormalSampleRate() {
  const envRate = import.meta.env?.VITE_VLM_AUDIT_NORMAL_SAMPLE_RATE;
  return envRate ?? DEFAULT_AUDIT_NORMAL_SAMPLE_RATE;
}

function normalizedSampleRate(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return DEFAULT_AUDIT_NORMAL_SAMPLE_RATE;
  return Math.max(0, Math.min(1, number));
}

function stableHash32(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function unique(items) {
  return [...new Set(items)];
}

function buildAttemptId(problemId, inputSignature) {
  const key = `${problemId || 'problem'}::${inputSignature || 'input'}`;
  return `attempt_${stableHash32(key).toString(16).padStart(8, '0')}`;
}

function stripCandidateImages(value) {
  if (!value || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(stripCandidateImages);
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (key === 'image' || key === 'dataUrl' || key === 'canvas') continue;
    result[key] = stripCandidateImages(item);
  }
  return result;
}

function clonePlain(value) {
  if (value === null || value === undefined) return value ?? null;
  return JSON.parse(JSON.stringify(stripCandidateImages(value)));
}
