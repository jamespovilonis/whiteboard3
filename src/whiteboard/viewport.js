export function screenToBoard(point, viewport) {
  return {
    x: viewport.x + point.x / viewport.scale,
    y: viewport.y + point.y / viewport.scale
  };
}

export function boardToScreen(point, viewport) {
  return {
    x: (point.x - viewport.x) * viewport.scale,
    y: (point.y - viewport.y) * viewport.scale
  };
}

export function getViewportDrift(viewport, initialViewport) {
  return {
    x: viewport.x - initialViewport.x,
    y: viewport.y - initialViewport.y
  };
}

export function isAwayFromViewport(viewport, initialViewport, threshold) {
  const drift = getViewportDrift(viewport, initialViewport);
  return Math.abs(drift.x) > threshold || Math.abs(drift.y) > threshold;
}

export function getInitialProblemPosition(viewport, viewportWidth) {
  return {
    x: viewport.x + viewportWidth * 0.25,
    y: viewport.y + 56
  };
}
