export function recognitionFetchErrorMessage(error, url) {
  const message = error instanceof Error ? error.message : String(error || 'Request failed');
  const targetUrl = String(url || '');
  if (isNetworkFetchFailure(message)) {
    return `${message} (${targetUrl}). Recognition API is unreachable. Start it with: python3 -m src.server.app --port 8010 --upstream-api-url http://127.0.0.1:8000 --semantic-timeout 2.5`;
  }
  return `${message} (${targetUrl})`;
}

function isNetworkFetchFailure(message) {
  const normalized = String(message || '').toLowerCase();
  return normalized.includes('failed to fetch') ||
    normalized.includes('networkerror') ||
    normalized.includes('load failed');
}
