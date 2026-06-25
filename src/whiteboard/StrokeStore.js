export class StrokeStore {
  constructor() {
    this.strokes = [];
    this.currentStroke = null;
    this.strokeCounter = 0;
    this.strokeStartTimes = new Map();
  }

  startStroke(rawX, rawY, pressure, boardWidth, boardHeight) {
    const id = `stroke_${this.strokeCounter + 1}_${Date.now()}`;
    this.strokeCounter += 1;
    const now = Date.now();

    this.currentStroke = {
      id,
      points: [{
        x: normalize(rawX, boardWidth),
        y: normalize(rawY, boardHeight),
        t: 0,
        pressure: pressure || 0.5
      }],
      startTime: now,
      rawPoints: [{ x: rawX, y: rawY, pressure: pressure || 0.5 }],
      outlinePoints: null,
      color: null
    };

    return id;
  }

  addPoint(rawX, rawY, pressure, boardWidth, boardHeight) {
    if (!this.currentStroke) return;

    this.currentStroke.points.push({
      x: normalize(rawX, boardWidth),
      y: normalize(rawY, boardHeight),
      t: Date.now() - this.currentStroke.startTime,
      pressure: pressure || 0.5
    });

    this.currentStroke.rawPoints.push({
      x: rawX,
      y: rawY,
      pressure: pressure || 0.5
    });
  }

  endStroke(outline, color) {
    if (!this.currentStroke || this.currentStroke.points.length === 0) {
      this.currentStroke = null;
      return null;
    }

    this.currentStroke.outlinePoints = outline || null;
    this.currentStroke.color = color || '#000000';
    this.strokeStartTimes.set(this.currentStroke.id, this.currentStroke.startTime);

    const bbox = computeNormalizedBbox(this.currentStroke.points);
    const canvasBbox = outline && outline.length > 0 ? computeBoardBbox(outline) : null;
    const prev = this.strokes.length > 0 ? this.strokes[this.strokes.length - 1] : null;
    const relationsToPrev = prev
      ? this.computeRelations(bbox, this.currentStroke.startTime, prev)
      : { dx: 0, dy: 0, dt: 0, overlapRatio: 0 };

    const group = {
      id: this.currentStroke.id,
      startTime: this.currentStroke.startTime,
      endTime: Date.now(),
      points: this.currentStroke.points,
      rawPoints: this.currentStroke.rawPoints,
      outlinePoints: this.currentStroke.outlinePoints,
      color: this.currentStroke.color,
      canvasBbox,
      bbox,
      relationsToPrev
    };

    this.strokes.push(group);
    this.currentStroke = null;
    return group;
  }

  removeStrokes(indices) {
    if (!indices || indices.length === 0) return [];

    const sorted = [...indices].sort((a, b) => b - a);
    const removed = [];

    for (const index of sorted) {
      if (index >= 0 && index < this.strokes.length) {
        removed.push(this.strokes.splice(index, 1)[0]);
      }
    }

    return removed;
  }

  getStrokes() {
    return this.strokes;
  }

  getStrokeCount() {
    return this.strokes.length;
  }

  clear() {
    this.strokes = [];
    this.currentStroke = null;
    this.strokeCounter = 0;
    this.strokeStartTimes.clear();
  }

  snapshot() {
    return {
      strokes: deepClone(this.strokes),
      strokeCounter: this.strokeCounter,
      strokeStartTimes: Array.from(this.strokeStartTimes)
    };
  }

  restore(entry) {
    this.strokes = deepClone(entry.strokes || []);
    this.currentStroke = null;
    this.strokeCounter = entry.strokeCounter || 0;
    this.strokeStartTimes = new Map(entry.strokeStartTimes || []);
  }

  computeRelations(currBbox, currStartTime, prev) {
    const cx = (currBbox.xMin + currBbox.xMax) / 2;
    const cy = (currBbox.yMin + currBbox.yMax) / 2;
    const px = (prev.bbox.xMin + prev.bbox.xMax) / 2;
    const py = (prev.bbox.yMin + prev.bbox.yMax) / 2;
    const prevStartTime = this.strokeStartTimes.get(prev.id);

    return {
      dx: cx - px,
      dy: cy - py,
      dt: prevStartTime !== undefined ? currStartTime - prevStartTime : 0,
      overlapRatio: computeIoU(currBbox, prev.bbox)
    };
  }
}

function normalize(value, size) {
  return Math.max(0, Math.min(1, value / size));
}

function computeNormalizedBbox(points) {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;

  for (const point of points) {
    xMin = Math.min(xMin, point.x);
    yMin = Math.min(yMin, point.y);
    xMax = Math.max(xMax, point.x);
    yMax = Math.max(yMax, point.y);
  }

  return { xMin, yMin, xMax, yMax };
}

export function computeBoardBbox(points) {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;

  for (const point of points) {
    xMin = Math.min(xMin, point.x);
    yMin = Math.min(yMin, point.y);
    xMax = Math.max(xMax, point.x);
    yMax = Math.max(yMax, point.y);
  }

  return { xMin, yMin, xMax, yMax };
}

function computeIoU(a, b) {
  const xMin = Math.max(a.xMin, b.xMin);
  const yMin = Math.max(a.yMin, b.yMin);
  const xMax = Math.min(a.xMax, b.xMax);
  const yMax = Math.min(a.yMax, b.yMax);
  const interW = Math.max(0, xMax - xMin);
  const interH = Math.max(0, yMax - yMin);
  const inter = interW * interH;
  const areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin);
  const areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin);
  const union = areaA + areaB - inter;

  return union > 0 ? inter / union : 0;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}
