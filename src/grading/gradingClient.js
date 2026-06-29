import { recognitionFetchErrorMessage } from '../recognition/fetchErrors.js';

export async function gradeEquationWork(request, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) ? Number(options.timeoutMs) : 5000;
  const url = `${apiUrl}/grade-equation-work`;
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
      return {
        failed: true,
        error: gradingHttpErrorMessage(response.status, url, payload),
        elapsedSeconds
      };
    }

    return {
      ...payload,
      failed: false,
      elapsedSeconds
    };
  } catch (error) {
    return {
      failed: true,
      error: recognitionFetchErrorMessage(error, url),
      elapsedSeconds: (performanceNow() - startedAt) / 1000
    };
  } finally {
    if (timer) clearTimeout(timer);
    if (externalSignal && controller) {
      externalSignal.removeEventListener('abort', abortFromExternal);
    }
  }
}

function gradingHttpErrorMessage(status, url, payload) {
  if (payload?.detail) return payload.detail;
  if (Number(status) === 404) {
    return `HTTP 404 from ${url}. The recognition gateway is running, but it does not expose grading yet. Restart it with the latest code: python3 testing/semantic_score_server.py --port 8010 --upstream-api-url http://127.0.0.1:8000 --semantic-timeout 2.5`;
  }
  return `HTTP ${status} from ${url}`;
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
