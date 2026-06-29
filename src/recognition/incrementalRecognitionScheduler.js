import { bboxOverlap, padBbox, unionBbox } from '../whiteboard/geometry.js';
import {
  computeTightBbox,
  segmentMathLines,
  strokeBelongsToAnswerBox
} from './lineSegmentation.js';
import { recognizeLineImage } from './ocrClient.js';
import { recognizeStudentWriting } from './studentWritingPipeline.js';

const DEFAULT_DEBOUNCE_MS = 500;
const DEFAULT_CATCHMENT_PADDING = 36;

export class IncrementalRecognitionScheduler {
  constructor(options = {}) {
    this.debounceMs = Number.isFinite(Number(options.debounceMs))
      ? Number(options.debounceMs)
      : DEFAULT_DEBOUNCE_MS;
    this.catchmentPadding = Number.isFinite(Number(options.catchmentPadding))
      ? Number(options.catchmentPadding)
      : DEFAULT_CATCHMENT_PADDING;
    this.recognizeWriting = options.recognizeWriting || recognizeStudentWriting;
    this.baseRecognizeLine = options.recognizeLine || recognizeLineImage;
    this.detectLines = options.detectLines;
    this.scoreSemantics = options.scoreSemantics;
    this.onStateChange = options.onStateChange || (() => {});
    this.onEvent = options.onEvent || (() => {});
    this.apiUrl = options.apiUrl || '';
    this.model = options.model || 'comer';
    this.timeoutMs = options.timeoutMs;
    this.detectionTimeoutMs = options.detectionTimeoutMs;
    this.semanticTimeoutMs = options.semanticTimeoutMs;
    this.semanticScoring = options.semanticScoring !== false;

    this.components = new Map();
    this.ocrCache = new Map();
    this.ocrInflight = new Map();
    this.fullAnswerInFlightSignature = null;
    this.fullAnswerFinalSignature = '';
    this.fullAnswerFinalResult = null;
    this.timer = null;
    this.input = null;
    this.inputSignature = '';
    this.previousStrokes = [];
    this.reconcileVersion = 0;
    this.strokeActive = false;
    this.bufferedSnapshot = null;
    this.paused = false;
  }

  update(input = {}) {
    this.strokeActive = false;
    const normalized = this.normalizeInput(input);
    const nextSignature = this.inputSignatureFor(normalized);
    if (this.paused) {
      this.cancelTimer();
      this.input = normalized;
      this.inputSignature = nextSignature;
      this.previousStrokes = normalized.strokes;
      return;
    }
    if (!normalized.problemId || !normalized.answerBox || normalized.answerStrokes.length === 0) {
      if (nextSignature === this.inputSignature) return;
      this.cancelTimer();
      this.input = normalized;
      this.inputSignature = nextSignature;
      this.previousStrokes = normalized.strokes;
      this.components.clear();
      this.clearFullAnswerState();
      this.emitState();
      return;
    }

    if (nextSignature === this.inputSignature) return;

    if (this.input?.problemId && this.input.problemId !== normalized.problemId) {
      this.components.clear();
      this.ocrInflight.clear();
      this.clearFullAnswerState();
    }

    const changed = diffStrokes(this.previousStrokes, normalized.strokes);
    this.input = normalized;
    this.inputSignature = nextSignature;
    this.previousStrokes = normalized.strokes;
    this.reconcileVersion += 1;

    this.softInvalidateOverlaps(changed);
    this.emitState();
    this.scheduleFlush();
  }

  dispose() {
    this.cancelTimer();
    this.components.clear();
    this.ocrInflight.clear();
    this.clearFullAnswerState();
  }

  beginStroke() {
    this.strokeActive = true;
    this.cancelTimer();
  }

  setPaused(paused) {
    const nextPaused = Boolean(paused);
    if (this.paused === nextPaused) return;
    this.paused = nextPaused;
    if (this.paused) {
      this.cancelTimer();
      return;
    }
    if (this.input?.problemId && this.input.answerBox && this.input.answerStrokes.length > 0) {
      this.scheduleFlush();
    }
  }

  async flushNow() {
    this.cancelTimer();
    await this.flush();
  }

  normalizeInput(input) {
    const strokes = Array.isArray(input.strokes) ? input.strokes.filter(Boolean) : [];
    const answerBox = input.answerBox || null;
    const answerStrokes = strokes.filter((stroke) => strokeBelongsToAnswerBox(stroke, answerBox));
    return {
      problemId: input.problemId || null,
      strokes,
      answerStrokes,
      answerBox,
      problemLatex: input.problemLatex || '',
      problemMetadata: input.problemMetadata || {},
      previousLatex: input.previousLatex || [],
      apiUrl: input.apiUrl || this.apiUrl
    };
  }

  inputSignatureFor(input) {
    return [
      input.problemId || '',
      bboxSignature(input.answerBox),
      ...input.answerStrokes.map(strokeSignature).sort()
    ].join('::');
  }

  scheduleFlush() {
    this.cancelTimer();
    this.timer = setTimeout(() => {
      this.timer = null;
      this.flush();
    }, this.debounceMs);
  }

  cancelTimer() {
    if (!this.timer) return;
    clearTimeout(this.timer);
    this.timer = null;
  }

  softInvalidateOverlaps(changed) {
    if (changed.length === 0) return;

    const changedBoxes = changed
      .map((entry) => entry.stroke?.canvasBbox || entry.previous?.canvasBbox || null)
      .filter(Boolean);
    const changedIds = new Set(changed.map((entry) => entry.stroke?.id || entry.previous?.id).filter(Boolean));

    for (const component of this.components.values()) {
      if (component.status === 'superseded') continue;
      const touchesStroke = component.strokeIds.some((strokeId) => changedIds.has(strokeId));
      const touchesCatchment = changedBoxes.some((box) => bboxOverlap(box, component.catchmentBbox));
      if (!touchesStroke && !touchesCatchment) continue;

      component.contested = true;
      component.status = component.result ? 'contested' : 'pending';
      component.updatedAt = Date.now();
      this.onEvent('recognition-component-contested', {
        problemId: this.input?.problemId || null,
        signature: component.signature,
        strokeIds: component.strokeIds
      });
    }
  }

  async flush() {
    const input = this.input;
    if (!input?.problemId || !input.answerBox || input.answerStrokes.length === 0) {
      this.emitState();
      return;
    }

    const version = this.reconcileVersion;
    const segmentation = segmentMathLines(input.strokes, {
      answerBox: input.answerBox,
      detections: []
    });
    const selected = segmentation.selected || [];
    const currentSignatures = new Set();

    for (const candidate of selected) {
      const component = this.upsertComponentFromCandidate(candidate, {
        source: 'deterministic',
        contested: false
      });
      currentSignatures.add(component.signature);
      this.startRecognitionForCandidate(component, candidate, {
        phase: 'deterministic',
        detectLineBands: false,
        final: false,
        version
      });
    }

    for (const component of this.components.values()) {
      if (component.status === 'superseded') continue;
      if (currentSignatures.has(component.signature)) continue;
      if (component.contested && component.result) continue;
      component.status = 'superseded';
    }

    this.emitState();

    for (const candidate of selected) {
      const component = this.components.get(candidateSignature(candidate));
      if (!component) continue;
      this.startDbnetRefinement(component, candidate, { version });
    }

    this.startFullAnswerFinalRecognition({ version });
  }

  upsertComponentFromCandidate(candidate, options = {}) {
    const signature = candidateSignature(candidate);
    let component = this.components.get(signature);
    const strokeIds = sortedStrokeIds(candidate);
    const tightBbox = cloneBbox(candidate.tightBbox);
    const catchmentBbox = cloneBbox(candidate.expandedBbox) ||
      padBbox(tightBbox, this.catchmentPadding);

    if (!component) {
      component = {
        signature,
        strokeIds,
        tightBbox,
        catchmentBbox,
        candidate,
        status: 'pending',
        source: options.source || 'deterministic',
        phase: null,
        result: null,
        error: null,
        contested: Boolean(options.contested),
        deterministicInFlight: null,
        dbnetInFlight: null,
        updatedAt: Date.now()
      };
      this.components.set(signature, component);
      return component;
    }

    component.strokeIds = strokeIds;
    component.tightBbox = tightBbox;
    component.catchmentBbox = catchmentBbox;
    component.candidate = candidate;
    component.source = options.source || component.source;
    component.contested = Boolean(options.contested);
    if (component.status === 'contested' && !component.contested) {
      component.status = component.result ? component.statusBeforeContest || 'provisional' : 'pending';
    }
    if (component.status === 'superseded') {
      component.status = component.result ? 'provisional' : 'pending';
    }
    component.updatedAt = Date.now();
    return component;
  }

  startRecognitionForCandidate(component, candidate, options = {}) {
    const phase = options.phase || 'deterministic';
    const inFlightKey = phase === 'dbnet' ? 'dbnetInFlight' : 'deterministicInFlight';
    if (component[inFlightKey] === component.signature) return;
    if (component.status === 'final' && phase !== 'dbnet') return;
    if (phase === 'deterministic' && component.result) return;

    component[inFlightKey] = component.signature;
    component.phase = phase;
    if (!component.result) component.status = 'running';
    component.updatedAt = Date.now();
    this.emitState();
    this.onEvent('recognition-component-start', {
      problemId: this.input?.problemId || null,
      phase,
      signature: component.signature,
      strokeIds: component.strokeIds
    });

    const startedSignature = component.signature;
    const startedVersion = options.version;
    this.runRecognition(candidate.strokes, {
      answerBox: candidate.tightBbox,
      detectLineBands: Boolean(options.detectLineBands),
      phase
    }).then((result) => {
      component[inFlightKey] = null;
      if (!this.isFresh(startedVersion)) return;

      if (options.final) {
        this.applyFinalRecognitionResult(result, {
          source: phase,
          affectedSignatures: [startedSignature],
          version: startedVersion
        });
        return;
      }

      const current = this.components.get(startedSignature);
      if (!current || current.status === 'superseded' || current.status === 'final') return;
      current.result = annotateResult(result, {
        status: current.contested ? 'contested' : 'provisional',
        source: phase,
        signature: current.signature
      });
      current.status = current.contested ? 'contested' : 'provisional';
      current.source = phase;
      current.error = null;
      current.updatedAt = Date.now();
      this.emitState();
    }).catch((error) => {
      component[inFlightKey] = null;
      if (!this.isFresh(startedVersion)) return;
      const current = this.components.get(startedSignature);
      if (!current || current.status === 'superseded' || current.status === 'final') return;
      current.status = current.result ? 'provisional' : 'error';
      current.error = error instanceof Error ? error.message : String(error);
      current.updatedAt = Date.now();
      this.emitState();
    });
  }

  startDbnetRefinement(component, candidate, options = {}) {
    if (component.dbnetInFlight === component.signature) return;
    if (component.status === 'final' && !component.contested) return;

    const neighborhood = this.neighborhoodFor(candidate, component);
    const affectedSignatures = [...this.components.values()]
      .filter((entry) => entry.status !== 'superseded')
      .filter((entry) => neighborhood.strokeIds.some((strokeId) => entry.strokeIds.includes(strokeId)) ||
        bboxOverlap(entry.catchmentBbox, neighborhood.bbox))
      .map((entry) => entry.signature);

    component.dbnetInFlight = component.signature;
    component.phase = 'dbnet';
    this.onEvent('recognition-dbnet-start', {
      problemId: this.input?.problemId || null,
      signature: component.signature,
      strokeIds: neighborhood.strokeIds
    });

    this.runRecognition(neighborhood.strokes, {
      answerBox: neighborhood.bbox,
      detectLineBands: true,
      phase: 'dbnet'
    }).then((result) => {
      component.dbnetInFlight = null;
      if (!this.isFresh(options.version)) return;
      this.applyFinalRecognitionResult(result, {
        source: 'dbnet',
        affectedSignatures,
        version: options.version
      });
    }).catch((error) => {
      component.dbnetInFlight = null;
      if (!this.isFresh(options.version)) return;
      const current = this.components.get(component.signature);
      if (!current || current.status === 'superseded') return;
      current.status = current.result ? 'provisional' : 'error';
      current.error = error instanceof Error ? error.message : String(error);
      current.updatedAt = Date.now();
      this.emitState();
    });
  }

  startFullAnswerFinalRecognition(options = {}) {
    const input = this.input;
    if (!input?.problemId || !input.answerBox || input.answerStrokes.length === 0) return;

    const signature = this.inputSignature;
    if (!signature) return;
    if (this.fullAnswerFinalSignature === signature && this.fullAnswerFinalResult) return;
    if (this.fullAnswerInFlightSignature === signature) return;

    this.fullAnswerInFlightSignature = signature;
    this.onEvent('recognition-final-pass-start', {
      problemId: input.problemId,
      answerStrokeCount: input.answerStrokes.length
    });

    this.runRecognition(input.answerStrokes, {
      answerBox: input.answerBox,
      detectLineBands: true,
      phase: 'full-answer'
    }).then((result) => {
      if (this.fullAnswerInFlightSignature === signature) {
        this.fullAnswerInFlightSignature = null;
      }
      if (!this.isFresh(options.version) || signature !== this.inputSignature) return;
      this.applyFullAnswerFinalResult(result, {
        source: 'full-answer',
        signature
      });
    }).catch((error) => {
      if (this.fullAnswerInFlightSignature === signature) {
        this.fullAnswerInFlightSignature = null;
      }
      if (!this.isFresh(options.version) || signature !== this.inputSignature) return;
      this.onEvent('recognition-final-pass-error', {
        problemId: input.problemId,
        error: error instanceof Error ? error.message : String(error)
      });
      this.emitState();
    });
  }

  neighborhoodFor(candidate, component) {
    const input = this.input;
    const seedBbox = component?.contested
      ? unionBbox(component.catchmentBbox, candidate.expandedBbox || candidate.tightBbox)
      : (candidate.expandedBbox || padBbox(candidate.tightBbox, this.catchmentPadding));
    const strokes = input.answerStrokes.filter((stroke) => {
      if (!stroke?.canvasBbox) return false;
      if ((candidate.strokeIds || []).includes(String(stroke.id))) return true;
      return bboxOverlap(stroke.canvasBbox, seedBbox);
    });
    const tight = computeTightBbox(strokes);
    const bbox = tight ? padBbox(tight, 2) : candidate.tightBbox;
    return {
      strokes: strokes.length ? strokes : candidate.strokes,
      strokeIds: (strokes.length ? strokes : candidate.strokes).map((stroke) => String(stroke.id)),
      bbox
    };
  }

  applyFinalRecognitionResult(result, { source, affectedSignatures = [] } = {}) {
    const finalResult = annotateResult(result, {
      status: 'final',
      source,
      signature: null
    });
    const finalSignatures = new Set();

    for (const line of finalResult.lines || []) {
      const signature = lineSignature(line);
      if (!signature) continue;
      finalSignatures.add(signature);
      let component = this.components.get(signature);
      if (!component) {
        component = {
          signature,
          strokeIds: sortedStrokeIds(line),
          tightBbox: cloneBbox(line.tightBbox),
          catchmentBbox: padBbox(line.tightBbox, this.catchmentPadding),
          candidate: null,
          status: 'final',
          source,
          phase: source,
          result: null,
          error: null,
          contested: false,
          deterministicInFlight: null,
          dbnetInFlight: null,
          updatedAt: Date.now()
        };
        this.components.set(signature, component);
      }
      component.result = finalResult;
      component.status = 'final';
      component.source = source;
      component.phase = source;
      component.error = null;
      component.contested = false;
      component.strokeIds = sortedStrokeIds(line);
      component.tightBbox = cloneBbox(line.tightBbox);
      component.catchmentBbox = padBbox(line.tightBbox, this.catchmentPadding);
      component.updatedAt = Date.now();
    }

    for (const signature of affectedSignatures) {
      if (finalSignatures.has(signature)) continue;
      const component = this.components.get(signature);
      if (!component || component.status === 'final') continue;
      component.status = 'superseded';
      component.contested = false;
      component.updatedAt = Date.now();
    }

    this.emitState();
  }

  applyFullAnswerFinalResult(result, { source, signature } = {}) {
    const finalResult = annotateResult(result, {
      status: 'final',
      source,
      signature: null
    });
    const finalSignatures = new Set();

    for (const line of finalResult.lines || []) {
      const lineSig = lineSignature(line);
      if (!lineSig) continue;
      finalSignatures.add(lineSig);
      let component = this.components.get(lineSig);
      if (!component) {
        component = {
          signature: lineSig,
          strokeIds: sortedStrokeIds(line),
          tightBbox: cloneBbox(line.tightBbox),
          catchmentBbox: padBbox(line.tightBbox, this.catchmentPadding),
          candidate: null,
          status: 'final',
          source,
          phase: source,
          result: null,
          error: null,
          contested: false,
          deterministicInFlight: null,
          dbnetInFlight: null,
          updatedAt: Date.now()
        };
        this.components.set(lineSig, component);
      }
      component.result = finalResult;
      component.status = 'final';
      component.source = source;
      component.phase = source;
      component.error = null;
      component.contested = false;
      component.strokeIds = sortedStrokeIds(line);
      component.tightBbox = cloneBbox(line.tightBbox);
      component.catchmentBbox = padBbox(line.tightBbox, this.catchmentPadding);
      component.updatedAt = Date.now();
    }

    for (const component of this.components.values()) {
      if (component.status === 'superseded') continue;
      if (finalSignatures.has(component.signature)) continue;
      component.status = 'superseded';
      component.contested = false;
      component.updatedAt = Date.now();
    }

    this.fullAnswerFinalSignature = signature || this.inputSignature;
    this.fullAnswerFinalResult = finalResult;
    this.emitState();
  }

  runRecognition(strokes, options = {}) {
    const input = this.input;
    return this.recognizeWriting({
      strokes,
      answerBox: options.answerBox,
      problemLatex: input.problemLatex,
      problemMetadata: input.problemMetadata || {},
      previousLatex: input.previousLatex || [],
      apiUrl: input.apiUrl,
      model: this.model,
      timeoutMs: this.timeoutMs,
      detectionTimeoutMs: this.detectionTimeoutMs,
      semanticTimeoutMs: this.semanticTimeoutMs,
      detectLineBands: Boolean(options.detectLineBands),
      semanticScoring: this.semanticScoring,
      recognizeLine: (image, recognizeOptions) => this.cachedRecognizeLine(image, recognizeOptions),
      ...(this.detectLines ? { detectLines: this.detectLines } : {}),
      ...(this.scoreSemantics ? { scoreSemantics: this.scoreSemantics } : {})
    });
  }

  cachedRecognizeLine(image, options = {}) {
    const key = ocrCacheKey(image, {
      apiUrl: options.apiUrl || this.input?.apiUrl || this.apiUrl,
      model: options.model || this.model
    });
    const cached = this.ocrCache.get(key);
    if (cached) {
      return Promise.resolve({
        ...cached,
        cached: true
      });
    }
    const inflight = this.ocrInflight.get(key);
    if (inflight) {
      return inflight.then((result) => ({
        ...result,
        cached: true,
        inFlightReused: true
      }));
    }

    const request = Promise.resolve()
      .then(() => this.baseRecognizeLine(image, options))
      .then((result) => {
        this.ocrCache.set(key, result);
        this.ocrInflight.delete(key);
        return result;
      })
      .catch((error) => {
        this.ocrInflight.delete(key);
        throw error;
      });
    this.ocrInflight.set(key, request);
    return request;
  }

  isFresh(version) {
    return version === this.reconcileVersion;
  }

  emitState() {
    if (!this.input?.problemId) return;
    const snapshot = this.buildSnapshot();
    if (this.strokeActive) {
      this.bufferedSnapshot = snapshot;
      return;
    }
    this.bufferedSnapshot = null;
    this.onStateChange(snapshot);
  }

  buildSnapshot() {
    const activeComponents = [...this.components.values()]
      .filter((component) => component.status !== 'superseded')
      .sort(compareComponents);
    const answerStrokeCount = this.input?.answerStrokes?.length || 0;
    const hasCurrentFullAnswerFinal = this.fullAnswerFinalSignature === this.inputSignature &&
      Boolean(this.fullAnswerFinalResult);
    const hasPending = activeComponents.some((component) => (
      component.status === 'pending' ||
      component.status === 'running' ||
      component.status === 'provisional' ||
      component.status === 'contested'
    ));
    const hasContested = activeComponents.some((component) => component.status === 'contested' || component.contested);
    const allFinal = answerStrokeCount > 0 &&
      hasCurrentFullAnswerFinal &&
      !hasContested;
    const metadata = {
      allFinal,
      answerStrokeCount,
      inputSignature: this.inputSignature
    };
    const aggregate = hasCurrentFullAnswerFinal
      ? fullAnswerAggregate(this.fullAnswerFinalResult, activeComponents, metadata)
      : aggregateComponentResults(activeComponents, metadata);
    const hasAnyResult = activeComponents.some((component) => component.result);

    return {
      problemId: this.input.problemId,
      status: answerStrokeCount === 0
        ? 'idle'
        : (allFinal && !hasContested ? 'complete' : 'pending'),
      result: hasAnyResult || activeComponents.length > 0 || aggregate.lines.length > 0 ? aggregate : null,
      realtime: aggregate.realtime,
      allFinal,
      hasPending: hasCurrentFullAnswerFinal ? false : hasPending,
      hasContested
    };
  }

  clearFullAnswerState() {
    this.fullAnswerInFlightSignature = null;
    this.fullAnswerFinalSignature = '';
    this.fullAnswerFinalResult = null;
  }
}

function aggregateComponentResults(components, metadata) {
  const linesBySignature = new Map();
  const candidatesByKey = new Map();
  const selectedBySignature = new Map();
  const allCandidatesByKey = new Map();
  const latexLines = [];

  for (const component of components) {
    const componentCandidate = componentDebugCandidate(component);
    if (componentCandidate) {
      const componentKey = `component:${component.signature}`;
      candidatesByKey.set(componentKey, componentCandidate);
      selectedBySignature.set(component.signature, componentCandidate);
      allCandidatesByKey.set(componentKey, componentCandidate);
    }

    const result = component.result;
    if (!result) continue;
    for (const line of result.lines || []) {
      const signature = lineSignature(line);
      if (!signature || linesBySignature.has(signature)) continue;
      linesBySignature.set(signature, {
        ...line,
        realtimeStatus: component.status,
        realtimeSource: component.source,
        provisional: component.status !== 'final'
      });
    }
    for (const entry of result.candidatePredictions || []) {
      const key = `${candidateSignature(entry)}:${entry.candidateId || ''}`;
      if (!candidatesByKey.has(key)) {
        candidatesByKey.set(key, {
          ...entry,
          realtimeStatus: component.status,
          realtimeSource: component.source,
          provisional: component.status !== 'final'
        });
      }
    }
    for (const selected of result.segmentation?.selected || []) {
      const signature = candidateSignature(selected);
      if (signature && !selectedBySignature.has(signature)) selectedBySignature.set(signature, selected);
    }
    for (const candidate of result.segmentation?.candidates || []) {
      const key = `${candidateSignature(candidate)}:${candidate.candidateId || ''}`;
      if (!allCandidatesByKey.has(key)) allCandidatesByKey.set(key, candidate);
    }
  }

  const lines = [...linesBySignature.values()].sort(compareLines).map((line, index) => ({
    ...line,
    lineIndex: index
  }));
  for (const line of lines) {
    latexLines.push(line.acceptedLatex || line.latex || '');
  }

  return {
    latexLines,
    lines,
    latex: latexLines.filter(Boolean).join(' \\\\ '),
    detection: {
      source: 'incremental',
      failed: false,
      componentCount: components.length,
      contestedCount: components.filter((component) => component.status === 'contested').length
    },
    semantic: {
      source: 'incremental',
      failed: false
    },
    timing: {
      totalElapsedSeconds: maxTiming(lines)
    },
    candidatePredictions: [...candidatesByKey.values()].sort(compareLines),
    segmentation: {
      selected: [...selectedBySignature.values()].sort(compareLines),
      candidates: [...allCandidatesByKey.values()].sort(compareLines),
      partitions: {},
      parentCandidateId: null,
      ocrSelectedCandidateIds: [...selectedBySignature.values()].map((candidate) => candidate.candidateId)
    },
    realtime: {
      allFinal: metadata.allFinal,
      answerStrokeCount: metadata.answerStrokeCount,
      inputSignature: metadata.inputSignature,
      components: components.map((component) => ({
        signature: component.signature,
        status: component.status,
        source: component.source,
        strokeIds: component.strokeIds,
        tightBbox: component.tightBbox,
        contested: Boolean(component.contested),
        hasResult: Boolean(component.result)
      }))
    }
  };
}

function fullAnswerAggregate(result, components, metadata) {
  const lines = (result?.lines || []).map((line, index) => ({
    ...line,
    lineIndex: index,
    realtimeStatus: 'final',
    realtimeSource: line.realtimeSource || 'full-answer',
    provisional: false
  }));
  const latexLines = Array.isArray(result?.latexLines)
    ? result.latexLines.slice()
    : lines.map((line) => line.acceptedLatex || line.latex || '');
  const candidatePredictions = (result?.candidatePredictions || []).map((entry) => ({
    ...entry,
    realtimeStatus: 'final',
    realtimeSource: entry.realtimeSource || 'full-answer',
    provisional: false
  }));
  const selected = result?.segmentation?.selected || [];
  const candidates = result?.segmentation?.candidates || selected;

  return {
    ...result,
    latexLines,
    lines,
    latex: result?.latex || latexLines.filter(Boolean).join(' \\\\ '),
    candidatePredictions,
    segmentation: {
      selected,
      candidates,
      partitions: result?.segmentation?.partitions || {},
      parentCandidateId: result?.segmentation?.parentCandidateId || null,
      ocrSelectedCandidateIds: result?.segmentation?.ocrSelectedCandidateIds ||
        selected.map((candidate) => candidate.candidateId)
    },
    realtime: {
      ...(result?.realtime || {}),
      allFinal: metadata.allFinal,
      answerStrokeCount: metadata.answerStrokeCount,
      inputSignature: metadata.inputSignature,
      components: components.map((component) => ({
        signature: component.signature,
        status: component.status,
        source: component.source,
        strokeIds: component.strokeIds,
        tightBbox: component.tightBbox,
        contested: Boolean(component.contested),
        hasResult: Boolean(component.result)
      }))
    }
  };
}

function annotateResult(result, { status, source }) {
  return {
    ...result,
    lines: (result.lines || []).map((line) => ({
      ...line,
      realtimeStatus: status,
      realtimeSource: source,
      provisional: status !== 'final'
    })),
    candidatePredictions: (result.candidatePredictions || []).map((entry) => ({
      ...entry,
      realtimeStatus: status,
      realtimeSource: source,
      provisional: status !== 'final'
    })),
    realtime: {
      ...(result.realtime || {}),
      status,
      source
    }
  };
}

function componentDebugCandidate(component) {
  if (!component?.tightBbox || !component.strokeIds?.length) return null;
  if (component.status === 'final' && component.result) return null;
  return {
    candidateId: `realtime_${component.signature}`,
    debugLabel: 'Live',
    selected: false,
    discarded: false,
    selectedLineIndex: null,
    profiles: [component.source || 'deterministic'],
    strokeIds: component.strokeIds.slice(),
    tightBbox: cloneBbox(component.tightBbox),
    image: null,
    latex: '',
    acceptedLatex: null,
    ocrLatex: '',
    candidates: [],
    prediction: null,
    semantic: null,
    contextualSemantic: null,
    sequentialSemantic: null,
    retryPredictions: [],
    semanticRetryPredictions: [],
    ocrRepair: null,
    evidenceScore: null,
    timing: null,
    realtimeStatus: component.status,
    realtimeSource: component.source,
    provisional: component.status !== 'final'
  };
}

function candidateSignature(candidate) {
  if (!candidate) return '';
  return `${sortedStrokeIds(candidate).join('|')}@${bboxSignature(candidate.tightBbox)}`;
}

function lineSignature(line) {
  return candidateSignature(line);
}

function sortedStrokeIds(candidate) {
  return (candidate?.strokeIds || [])
    .map((id) => String(id))
    .sort();
}

function strokeSignature(stroke) {
  return `${String(stroke.id)}@${bboxSignature(stroke.canvasBbox)}`;
}

function bboxSignature(bbox) {
  if (!bbox) return 'none';
  return [
    bbox.xMin,
    bbox.yMin,
    bbox.xMax,
    bbox.yMax
  ].map((value) => Math.round(Number(value) * 10) / 10).join(',');
}

function cloneBbox(bbox) {
  return bbox ? { ...bbox } : null;
}

function diffStrokes(previous = [], next = []) {
  const previousById = new Map(previous.map((stroke) => [String(stroke.id), stroke]));
  const nextById = new Map(next.map((stroke) => [String(stroke.id), stroke]));
  const changed = [];

  for (const [id, stroke] of nextById) {
    const old = previousById.get(id);
    if (!old || bboxSignature(old.canvasBbox) !== bboxSignature(stroke.canvasBbox)) {
      changed.push({ stroke, previous: old || null });
    }
  }
  for (const [id, stroke] of previousById) {
    if (!nextById.has(id)) changed.push({ stroke: null, previous: stroke });
  }

  return changed;
}

function ocrCacheKey(image, options = {}) {
  return [
    options.apiUrl || '',
    options.model || '',
    (image?.strokeIds || []).map(String).sort().join('|'),
    bboxSignature(image?.tightBbox),
    image?.padding ?? '',
    image?.targetPixelHeight ?? '',
    image?.width ?? '',
    image?.height ?? ''
  ].join('::');
}

function compareComponents(a, b) {
  return compareBboxes(a.tightBbox, b.tightBbox) ||
    String(a.signature).localeCompare(String(b.signature));
}

function compareLines(a, b) {
  return compareBboxes(a.tightBbox, b.tightBbox) ||
    String(a.candidateId || '').localeCompare(String(b.candidateId || ''));
}

function compareBboxes(a, b) {
  return (a?.yMin ?? 0) - (b?.yMin ?? 0) ||
    (a?.xMin ?? 0) - (b?.xMin ?? 0) ||
    (a?.yMax ?? 0) - (b?.yMax ?? 0) ||
    (a?.xMax ?? 0) - (b?.xMax ?? 0);
}

function maxTiming(lines) {
  const values = (lines || [])
    .map((line) => Number(line.timing?.submitToFinalPredictionSeconds))
    .filter(Number.isFinite);
  return values.length ? Math.max(...values) : null;
}
