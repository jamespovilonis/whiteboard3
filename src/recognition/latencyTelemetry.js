export const DEFAULT_RECOGNITION_LATENCY_BUDGETS_MS = Object.freeze({
  strokeCapture: 16,
  schedulerFlush: 50,
  segmentation: 35,
  detection: 900,
  ocr: 2500,
  retry: 4500,
  chunkFallback: 3500,
  semantic: 1200,
  grading: 750,
  auditEnqueue: 250,
  totalUiBlocking: 6000
});

export function createRecognitionLatencyTelemetry(options = {}) {
  const budgetsMs = {
    ...DEFAULT_RECOGNITION_LATENCY_BUDGETS_MS,
    ...(options.budgetsMs || {})
  };
  const samples = new Map();

  function record(stage, elapsedMs, metadata = {}) {
    const normalizedStage = String(stage || '').trim();
    const value = Number(elapsedMs);
    if (!normalizedStage || !Number.isFinite(value) || value < 0) return null;
    if (!samples.has(normalizedStage)) samples.set(normalizedStage, []);
    const sample = {
      stage: normalizedStage,
      elapsedMs: roundMs(value),
      budgetMs: finiteBudget(budgetsMs[normalizedStage]),
      ...compactMetadata(metadata)
    };
    sample.overBudget = sample.budgetMs !== null && sample.elapsedMs > sample.budgetMs;
    samples.get(normalizedStage).push(sample);
    return sample;
  }

  function measure(stage, fn, metadata = {}) {
    const startedAt = performanceNow();
    try {
      const value = fn();
      if (value && typeof value.then === 'function') {
        return value.finally(() => record(stage, performanceNow() - startedAt, metadata));
      }
      record(stage, performanceNow() - startedAt, metadata);
      return value;
    } catch (error) {
      record(stage, performanceNow() - startedAt, { ...metadata, failed: true });
      throw error;
    }
  }

  function summary(extra = {}) {
    const stages = {};
    const budgetFailures = [];
    for (const [stage, entries] of samples.entries()) {
      const elapsed = entries
        .map((entry) => Number(entry.elapsedMs))
        .filter((value) => Number.isFinite(value))
        .sort((a, b) => a - b);
      if (!elapsed.length) continue;
      const stageSummary = {
        count: elapsed.length,
        p50Ms: percentile(elapsed, 0.50),
        p95Ms: percentile(elapsed, 0.95),
        maxMs: roundMs(elapsed[elapsed.length - 1]),
        budgetMs: finiteBudget(budgetsMs[stage]),
        overBudgetCount: entries.filter((entry) => entry.overBudget).length
      };
      stageSummary.overBudget = stageSummary.budgetMs !== null &&
        stageSummary.p95Ms > stageSummary.budgetMs;
      stages[stage] = stageSummary;
      for (const entry of entries) {
        if (!entry.overBudget) continue;
        budgetFailures.push({
          stage,
          elapsedMs: entry.elapsedMs,
          budgetMs: entry.budgetMs,
          ...compactMetadata(entry)
        });
      }
    }
    return {
      budgetsMs,
      stages,
      budgetFailures,
      budgetFailureCount: budgetFailures.length,
      overBudget: budgetFailures.length > 0,
      ...extra
    };
  }

  return {
    budgetsMs,
    record,
    measure,
    summary
  };
}

export function strokeCaptureLatencySamples(strokes = []) {
  return (Array.isArray(strokes) ? strokes : [])
    .map((stroke) => {
      const captureElapsed = Number(stroke?.latency?.strokeCaptureElapsedMs);
      if (Number.isFinite(captureElapsed) && captureElapsed >= 0) {
        return {
          elapsedMs: captureElapsed,
          strokeId: stroke.id ? String(stroke.id) : null,
          pointCount: Array.isArray(stroke.points) ? stroke.points.length : null,
          drawDurationMs: finiteNumber(stroke?.latency?.drawDurationMs)
        };
      }
      const start = Number(stroke?.startTime);
      const end = Number(stroke?.endTime);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
      return {
        elapsedMs: 0,
        strokeId: stroke.id ? String(stroke.id) : null,
        pointCount: Array.isArray(stroke.points) ? stroke.points.length : null,
        drawDurationMs: end - start,
        estimated: true
      };
    })
    .filter(Boolean);
}

export function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function percentile(sortedValues, p) {
  if (!sortedValues.length) return null;
  if (sortedValues.length === 1) return roundMs(sortedValues[0]);
  const index = Math.min(sortedValues.length - 1, Math.max(0, Math.ceil(sortedValues.length * p) - 1));
  return roundMs(sortedValues[index]);
}

function finiteBudget(value) {
  const budget = Number(value);
  return Number.isFinite(budget) && budget >= 0 ? roundMs(budget) : null;
}

function roundMs(value) {
  return Math.round(Number(value) * 10) / 10;
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function compactMetadata(metadata = {}) {
  const compacted = {};
  for (const [key, value] of Object.entries(metadata || {})) {
    if (value === undefined || value === null || key === 'stage') continue;
    if (key === 'elapsedMs' || key === 'budgetMs' || key === 'overBudget') continue;
    if (typeof value === 'number') {
      if (Number.isFinite(value)) compacted[key] = value;
      continue;
    }
    if (typeof value === 'string' || typeof value === 'boolean') {
      compacted[key] = value;
      continue;
    }
    if (Array.isArray(value)) {
      compacted[key] = value.slice(0, 12);
    }
  }
  return compacted;
}
