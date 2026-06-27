export async function requestLineDetections(candidateImage, options = {}) {
  const apiUrl = String(options.apiUrl || '').replace(/\/$/, '');
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) ? Number(options.timeoutMs) : 10000;
  const dataUrl = candidateImage?.dataUrl || candidateImage;
  if (!dataUrl) throw new Error('requestLineDetections requires a data URL');

  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const startedAt = performanceNow();

  try {
    const blob = await dataUrlToBlob(dataUrl);
    const body = new FormData();
    body.append('file', blob, 'answer.png');

    const response = await fetch(`${apiUrl}/segment-lines`, {
      method: 'POST',
      body,
      signal: controller?.signal
    });
    const payload = await response.json().catch(() => null);
    const elapsedSeconds = payload?.elapsedSeconds ?? ((performanceNow() - startedAt) / 1000);

    if (!response.ok || !payload) {
      return {
        detections: [],
        failed: true,
        error: payload?.detail || `HTTP ${response.status}`,
        elapsedSeconds
      };
    }

    return {
      detections: translateDetections(payload.detections || [], candidateImage),
      rawDetections: payload.detections || [],
      imageWidth: payload.imageWidth,
      imageHeight: payload.imageHeight,
      model: payload.model || 'dbnet',
      elapsedSeconds,
      failed: false
    };
  } catch (error) {
    return {
      detections: [],
      failed: true,
      error: error instanceof Error ? error.message : String(error),
      elapsedSeconds: (performanceNow() - startedAt) / 1000
    };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export function translateDetections(detections, candidateImage = {}) {
  const originX = Number(candidateImage.originX) || 0;
  const originY = Number(candidateImage.originY) || 0;
  const scale = Number(candidateImage.devicePixelRatio) || 1;

  return (detections || [])
    .filter((item) => item?.bbox)
    .map((item) => ({
      ...item,
      bbox: {
        xMin: item.bbox.xMin / scale + originX,
        yMin: item.bbox.yMin / scale + originY,
        xMax: item.bbox.xMax / scale + originX,
        yMax: item.bbox.yMax / scale + originY
      },
      polygon: (item.polygon || []).map((point) => [
        point[0] / scale + originX,
        point[1] / scale + originY
      ])
    }));
}

async function dataUrlToBlob(dataUrl) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  if (!blob || blob.size === 0) throw new Error('Detection image data URL produced an empty blob');
  return blob;
}

function performanceNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

