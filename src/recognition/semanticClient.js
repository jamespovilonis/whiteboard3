import { recognitionFetchErrorMessage } from './fetchErrors.js';

export async function scoreLatexCandidates(request, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) ? Number(options.timeoutMs) : 5000;
  const url = `${apiUrl}/score-latex-candidates`;
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const startedAt = performanceNow();

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: controller?.signal
    });
    const payload = await response.json().catch(() => null);
    const elapsedSeconds = payload?.elapsedSeconds ?? ((performanceNow() - startedAt) / 1000);

    if (!response.ok || !payload) {
      return {
        candidateScores: [],
        failed: true,
        error: payload?.detail || `HTTP ${response.status} from ${url}`,
        elapsedSeconds
      };
    }

    return {
      candidateScores: payload.candidateScores || [],
      failed: false,
      elapsedSeconds
    };
  } catch (error) {
    return {
      candidateScores: [],
      failed: true,
      error: fetchErrorMessage(error, url),
      elapsedSeconds: (performanceNow() - startedAt) / 1000
    };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function fetchErrorMessage(error, url) {
  return recognitionFetchErrorMessage(error, url);
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
