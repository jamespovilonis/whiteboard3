export const SMOOTHER_DEFAULTS = Object.freeze({
  size: 12,
  thinning: 0.5,
  smoothing: 0.9,
  streamline: 0.6,
  startTaper: 1,
  endTaper: 1,
  cap: true
});

export class StrokeSmoother {
  constructor(options = {}) {
    this.opts = { ...SMOOTHER_DEFAULTS, ...options };
  }

  smooth(points) {
    if (!points || points.length < 2) return [];

    let pts = points.map((point) => ({
      x: point.x,
      y: point.y,
      pressure: point.pressure || 0.5
    }));

    if (this.opts.streamline > 0 && pts.length > 1) {
      streamlinePoints(pts, this.opts.streamline);
    }

    if (this.opts.smoothing > 0 && pts.length > 2) {
      pts = catmullRomSpline(pts, this.opts.smoothing);
    }

    const thicknessResult = computeThicknesses(pts, this.opts);
    return buildOutline(
      pts,
      thicknessResult.array,
      this.opts,
      thicknessResult.startFull,
      thicknessResult.endFull
    );
  }

  render(ctx, outline, color) {
    if (!outline || outline.length < 3) return;

    ctx.beginPath();
    ctx.moveTo(outline[0].x, outline[0].y);
    for (let i = 1; i < outline.length; i += 1) {
      ctx.lineTo(outline[i].x, outline[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = color || '#000000';
    ctx.fill();
  }
}

function streamlinePoints(pts, amount) {
  const factor = 1 - amount;
  for (let i = 1; i < pts.length; i += 1) {
    const prev = pts[i - 1];
    const curr = pts[i];
    curr.x = prev.x + (curr.x - prev.x) * factor;
    curr.y = prev.y + (curr.y - prev.y) * factor;
  }
}

function catmullRomSpline(pts, tension) {
  const numSegments = pts.length - 1;
  const samplesPerSegment = Math.max(2, Math.round(4 + tension * 8));
  const totalPoints = numSegments * samplesPerSegment + 1;
  const result = new Array(totalPoints);

  for (let i = 0; i < numSegments; i += 1) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];

    for (let j = 0; j < samplesPerSegment; j += 1) {
      const t = j / samplesPerSegment;
      const t2 = t * t;
      const t3 = t2 * t;

      const x = 0.5 * (
        (2 * p1.x) +
        (-p0.x + p2.x) * t +
        (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
        (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
      );

      const y = 0.5 * (
        (2 * p1.y) +
        (-p0.y + p2.y) * t +
        (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
        (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
      );

      const pressure = p1.pressure + (p2.pressure - p1.pressure) * t;
      result[i * samplesPerSegment + j] = { x, y, pressure };
    }
  }

  const last = pts[pts.length - 1];
  result[totalPoints - 1] = {
    x: last.x,
    y: last.y,
    pressure: last.pressure
  };

  return result;
}

function computeThicknesses(pts, opts) {
  const baseSize = opts.size;
  const thinning = opts.thinning;
  const len = pts.length;
  const thicknesses = new Array(len);

  let startFull = baseSize * (0.5 + ((pts[0].pressure || 0.5) - 0.5) * Math.max(0, thinning));
  startFull = Math.max(startFull, baseSize * 0.1);

  let endFull = baseSize * (0.5 + ((pts[len - 1].pressure || 0.5) - 0.5) * Math.max(0, thinning));
  endFull = Math.max(endFull, baseSize * 0.1);

  for (let i = 0; i < len; i += 1) {
    const point = pts[i];
    let t = 0.5;

    if (thinning > 0) {
      const pressure = point.pressure || 0.5;
      t = 0.5 + (pressure - 0.5) * thinning;
      t = Math.max(0.1, Math.min(1, t));
    }

    let thickness = baseSize * t;

    if (opts.startTaper > 0 && i < opts.startTaper) {
      thickness *= i / opts.startTaper;
    }
    if (opts.endTaper > 0 && i >= len - opts.endTaper) {
      thickness *= (len - 1 - i) / opts.endTaper;
    }
    if (thickness < 0.5) thickness = 0.5;

    thicknesses[i] = thickness;
  }

  return { array: thicknesses, startFull, endFull };
}

function buildOutline(pts, thicknesses, opts, startFullThickness, endFullThickness) {
  const len = pts.length;
  if (len < 2) return [];

  const left = new Array(len);
  const right = new Array(len);

  for (let i = 0; i < len; i += 1) {
    const half = thicknesses[i] / 2;
    let px;
    let py;

    if (i === 0) {
      const dx = pts[1].x - pts[0].x;
      const dy = pts[1].y - pts[0].y;
      const mag = Math.sqrt(dx * dx + dy * dy) || 1;
      px = -dy / mag;
      py = dx / mag;
    } else if (i === len - 1) {
      const dx = pts[i].x - pts[i - 1].x;
      const dy = pts[i].y - pts[i - 1].y;
      const mag = Math.sqrt(dx * dx + dy * dy) || 1;
      px = -dy / mag;
      py = dx / mag;
    } else {
      const dx1 = pts[i].x - pts[i - 1].x;
      const dy1 = pts[i].y - pts[i - 1].y;
      const mag1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) || 1;
      const px1 = -dy1 / mag1;
      const py1 = dx1 / mag1;

      const dx2 = pts[i + 1].x - pts[i].x;
      const dy2 = pts[i + 1].y - pts[i].y;
      const mag2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) || 1;
      const px2 = -dy2 / mag2;
      const py2 = dx2 / mag2;

      px = (px1 + px2) * 0.5;
      py = (py1 + py2) * 0.5;
      const mag = Math.sqrt(px * px + py * py) || 1;
      px /= mag;
      py /= mag;
    }

    left[i] = { x: pts[i].x + px * half, y: pts[i].y + py * half };
    right[i] = { x: pts[i].x - px * half, y: pts[i].y - py * half };
  }

  const outline = [];
  const capSegs = opts.cap ? 8 : 0;

  for (let i = 0; i < len; i += 1) {
    outline.push(left[i]);
  }

  if (capSegs > 0 && len > 1) {
    const lastHalf = endFullThickness / 2;
    if (lastHalf > 0.5) {
      const edx = pts[len - 1].x - pts[len - 2].x;
      const edy = pts[len - 1].y - pts[len - 2].y;
      addCap(outline, pts[len - 1], lastHalf, Math.atan2(edy, edx), false, capSegs);
    }
  }

  for (let i = len - 1; i >= 0; i -= 1) {
    outline.push(right[i]);
  }

  if (capSegs > 0 && len > 1) {
    const firstHalf = startFullThickness / 2;
    if (firstHalf > 0.5) {
      const sdx = pts[1].x - pts[0].x;
      const sdy = pts[1].y - pts[0].y;
      addCap(outline, pts[0], firstHalf, Math.atan2(sdy, sdx), true, capSegs);
    }
  }

  return outline;
}

function addCap(outline, center, radius, directionAngle, isStart, segments) {
  let startAngle;
  let endAngle;

  if (isStart) {
    startAngle = directionAngle - Math.PI / 2 + 2 * Math.PI;
    endAngle = directionAngle + Math.PI / 2;
  } else {
    startAngle = directionAngle + Math.PI / 2;
    endAngle = directionAngle - Math.PI / 2;
  }

  for (let i = 0; i <= segments; i += 1) {
    const a = startAngle + (endAngle - startAngle) * (i / segments);
    outline.push({
      x: center.x + Math.cos(a) * radius,
      y: center.y + Math.sin(a) * radius
    });
  }
}
