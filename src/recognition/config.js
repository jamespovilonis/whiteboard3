export function getRecognitionApiUrl() {
  const envUrl = import.meta.env?.VITE_API_URL || import.meta.env?.VITE_OCR_API_URL;
  if (envUrl) return normalizeConfiguredApiUrl(envUrl);

  if (typeof window !== 'undefined' && window.location?.hostname) {
    const protocol = window.location.protocol || 'http:';
    const hostname = gatewayHostname(window.location.hostname);
    return `${protocol}//${hostname}:8010`;
  }

  return '';
}

function trimTrailingSlash(url) {
  return String(url || '').replace(/\/$/, '');
}

export function normalizeConfiguredApiUrl(url) {
  const configured = trimTrailingSlash(url);
  if (typeof window === 'undefined' || !window.location?.hostname) return configured;

  try {
    const parsed = new URL(configured);
    const pageHostname = String(window.location.hostname || '').trim();
    if (isLoopbackHostname(parsed.hostname) && !isLoopbackHostname(pageHostname)) {
      parsed.hostname = pageHostname;
      return trimTrailingSlash(parsed.toString());
    }
  } catch (_error) {
    return configured;
  }

  return configured;
}

function gatewayHostname(hostname) {
  const value = String(hostname || '').trim();
  if (!value || value === '0.0.0.0' || value === '::') return '127.0.0.1';
  return value;
}

function isLoopbackHostname(hostname) {
  const value = String(hostname || '').trim().toLowerCase();
  return value === 'localhost' ||
    value === '127.0.0.1' ||
    value === '::1' ||
    value === '[::1]';
}
