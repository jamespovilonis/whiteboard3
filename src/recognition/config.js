export function getRecognitionApiUrl() {
  const envUrl = import.meta.env?.VITE_OCR_API_URL;
  if (envUrl) return trimTrailingSlash(envUrl);

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

function gatewayHostname(hostname) {
  const value = String(hostname || '').trim();
  if (!value || value === '0.0.0.0' || value === '::') return '127.0.0.1';
  return value;
}
