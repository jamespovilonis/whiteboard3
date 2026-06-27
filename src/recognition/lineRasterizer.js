export function rasterizeLineCandidate(candidate, options = {}) {
  if (typeof document === 'undefined') {
    throw new Error('rasterizeLineCandidate requires a browser document');
  }
  if (!candidate?.tightBbox || !candidate?.strokes?.length) {
    throw new Error('Cannot rasterize an empty line candidate');
  }

  const padding = Number.isFinite(Number(options.padding)) ? Number(options.padding) : 24;
  const dpr = Number.isFinite(Number(options.devicePixelRatio))
    ? Number(options.devicePixelRatio)
    : (window.devicePixelRatio || 1);
  const targetPixelHeight = Number(options.targetPixelHeight);

  const minX = Math.floor(candidate.tightBbox.xMin);
  const minY = Math.floor(candidate.tightBbox.yMin);
  const maxX = Math.ceil(candidate.tightBbox.xMax);
  const maxY = Math.ceil(candidate.tightBbox.yMax);
  const originX = minX - padding;
  const originY = minY - padding;
  const cssWidth = Math.max(1, maxX - minX + padding * 2);
  const cssHeight = Math.max(1, maxY - minY + padding * 2);
  const renderScale = Number.isFinite(targetPixelHeight) && targetPixelHeight > 0
    ? targetPixelHeight / cssHeight
    : dpr;

  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.ceil(cssWidth * renderScale));
  canvas.height = Math.max(1, Math.ceil(cssHeight * renderScale));

  const ctx = canvas.getContext('2d');
  ctx.scale(renderScale, renderScale);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.translate(padding - minX, padding - minY);
  ctx.fillStyle = '#000000';

  for (const stroke of candidate.strokes) {
    const outline = stroke.outlinePoints || stroke.rawPoints;
    if (!outline || outline.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(outline[0].x, outline[0].y);
    for (let index = 1; index < outline.length; index += 1) {
      ctx.lineTo(outline[index].x, outline[index].y);
    }
    ctx.closePath();
    ctx.fill();
  }

  return {
    candidateId: candidate.candidateId,
    profiles: candidate.profiles.slice(),
    strokeIds: candidate.strokeIds.slice(),
    tightBbox: { ...candidate.tightBbox },
    canvas,
    dataUrl: canvas.toDataURL('image/png'),
    padding,
    originX,
    originY,
    devicePixelRatio: renderScale,
    targetPixelHeight: Number.isFinite(targetPixelHeight) && targetPixelHeight > 0
      ? targetPixelHeight
      : null,
    width: canvas.width,
    height: canvas.height,
    cssWidth,
    cssHeight
  };
}

export function rasterizeLineCandidates(candidates, options = {}) {
  return (candidates || []).map((candidate, index) => ({
    lineIndex: index,
    ...rasterizeLineCandidate(candidate, options)
  }));
}
