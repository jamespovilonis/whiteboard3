export function getRecognitionApiUrl() {
  const envUrl = import.meta.env?.VITE_OCR_API_URL;
  if (envUrl) return trimTrailingSlash(envUrl);

  if (typeof window !== 'undefined' && window.location?.hostname) {
    const protocol = window.location.protocol || 'http:';
    return `${protocol}//${window.location.hostname}:8000`;
  }

  return '';
}

function trimTrailingSlash(url) {
  return String(url || '').replace(/\/$/, '');
}

