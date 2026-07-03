import { recognitionFetchErrorMessage } from '../recognition/fetchErrors.js';

export const FEEDBACK_PROMPT_VERSION = 'math-feedback-v1';

export async function requestMathFeedback(request, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) ? Number(options.timeoutMs) : 15000;
  const url = `${apiUrl}/feedback/math-work`;
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const externalSignal = options.signal || null;
  if (externalSignal?.aborted) controller?.abort();
  const abortFromExternal = () => controller?.abort();
  if (externalSignal && controller) {
    externalSignal.addEventListener('abort', abortFromExternal, { once: true });
  }
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const startedAt = performanceNow();

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request || {}),
      signal: controller?.signal
    });
    const payload = await response.json().catch(() => null);
    const elapsedSeconds = payload?.elapsedSeconds ?? ((performanceNow() - startedAt) / 1000);
    if (!response.ok || !payload) {
      return fallbackFeedback(request, recognitionFetchErrorMessage(new Error(feedbackHttpErrorMessage(response.status, url, payload)), url), elapsedSeconds);
    }
    return {
      ...payload,
      elapsedSeconds
    };
  } catch (error) {
    if (error?.name === 'AbortError') {
      return {
        attemptId: request?.attemptId || null,
        inputSignature: request?.inputSignature || '',
        status: 'aborted',
        source: 'aborted',
        text: '',
        promptVersion: FEEDBACK_PROMPT_VERSION,
        error: 'Feedback request aborted',
        elapsedSeconds: (performanceNow() - startedAt) / 1000
      };
    }
    return fallbackFeedback(request, recognitionFetchErrorMessage(error, url), (performanceNow() - startedAt) / 1000);
  } finally {
    if (timer) clearTimeout(timer);
    if (externalSignal && controller) {
      externalSignal.removeEventListener('abort', abortFromExternal);
    }
  }
}

export function createCorrectFeedback({ attemptId = null, inputSignature = '', model = '' } = {}) {
  return {
    attemptId,
    inputSignature,
    status: 'complete',
    source: 'deterministic',
    text: 'Correct! Great job!',
    model,
    promptVersion: FEEDBACK_PROMPT_VERSION,
    skippedReason: 'correct'
  };
}

export function trimFeedbackText(text = '') {
  const compact = String(text || '').replace(/\s+/g, ' ').trim();
  if (!compact) return '';
  const sentences = compact.match(/[^.!?]+[.!?]?/g) || [compact];
  return sentences.slice(0, 2).map((sentence) => sentence.trim()).filter(Boolean).join(' ').slice(0, 500).trim();
}

function fallbackFeedback(request, error, elapsedSeconds) {
  return {
    attemptId: request?.attemptId || null,
    inputSignature: request?.inputSignature || '',
    status: 'complete',
    source: 'fallback',
    text: deterministicFallbackText(request),
    model: '',
    promptVersion: FEEDBACK_PROMPT_VERSION,
    error,
    elapsedSeconds
  };
}

function deterministicFallbackText(request = {}) {
  const grading = request.grading || request.fastResult?.grading || {};
  const status = grading?.result?.problemStatus || '';
  const firstInvalid = (grading.steps || []).find((step) => step?.classification === 'invalid_step');
  if (status === 'incorrect' && firstInvalid) {
    const lineNumber = Number(firstInvalid.lineIndex) + 1;
    return `Check line ${Number.isFinite(lineNumber) ? lineNumber : 1}: ${firstInvalid.studentLatex || 'that step'} is not a valid step. Return to the previous valid line and keep the equation balanced.`;
  }
  if (status === 'incomplete') {
    return 'Keep going from your last valid line and preserve the same value on both sides.';
  }
  if (status === 'not_started') {
    return 'Start by rewriting the problem or applying one valid operation to both sides.';
  }
  return 'Return to your latest valid work and continue with a balanced math step.';
}

function feedbackHttpErrorMessage(status, url, payload) {
  if (payload?.detail) return payload.detail;
  return `HTTP ${status} from ${url}`;
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
