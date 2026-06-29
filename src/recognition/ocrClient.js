import { recognitionFetchErrorMessage } from './fetchErrors.js';

export async function recognizeLineImage(lineImage, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const model = options.model || 'comer';
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) ? Number(options.timeoutMs) : 20000;
  const timeoutSeconds = Math.max(0.1, Math.min(20, timeoutMs / 1000));
  const url = `${apiUrl}/recognize?model=${encodeURIComponent(model)}&timeout_seconds=${encodeURIComponent(timeoutSeconds)}`;

  const dataUrl = typeof lineImage === 'string' ? lineImage : lineImage?.dataUrl;
  if (!dataUrl) throw new Error('recognizeLineImage requires a data URL');

  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const externalSignal = options.signal || null;
  if (externalSignal?.aborted) controller?.abort();
  const abortFromExternal = () => controller?.abort();
  if (externalSignal && controller) {
    externalSignal.addEventListener('abort', abortFromExternal, { once: true });
  }
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs + 750) : null;
  const startedAt = performanceNow();

  try {
    const blob = await dataUrlToBlob(dataUrl);
    const body = new FormData();
    body.append('file', blob, 'line.png');

    const response = await fetch(url, {
      method: 'POST',
      body,
      signal: controller?.signal
    });
    const payload = await response.json().catch(() => null);
    const elapsedSeconds = payload?.elapsedSeconds ?? ((performanceNow() - startedAt) / 1000);

    if (!response.ok || !payload) {
      return {
        model,
        latex: '',
        candidates: [],
        confidence: 0,
        failed: true,
        error: payload?.detail || `HTTP ${response.status} from ${url}`,
        elapsedSeconds
      };
    }

    return {
      model,
      latex: payload.top?.latex || '',
      top: payload.top || null,
      candidates: payload.candidates || [],
      confidence: payload.top?.confidence ?? 0,
      timedOut: Boolean(payload.timedOut),
      elapsedSeconds,
      selectionPenalty: payload.selectionPenalty || 0
    };
  } catch (error) {
    return {
      model,
      latex: '',
      candidates: [],
      confidence: 0,
      failed: true,
      error: fetchErrorMessage(error, url),
      elapsedSeconds: (performanceNow() - startedAt) / 1000
    };
  } finally {
    if (timer) clearTimeout(timer);
    if (externalSignal && controller) {
      externalSignal.removeEventListener('abort', abortFromExternal);
    }
  }
}

function fetchErrorMessage(error, url) {
  return recognitionFetchErrorMessage(error, url);
}

async function dataUrlToBlob(dataUrl) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  if (!blob || blob.size === 0) throw new Error('Line image data URL produced an empty blob');
  return blob;
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
