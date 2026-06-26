export function computePolygonBbox(poly) {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;

  for (const point of poly) {
    xMin = Math.min(xMin, point.x);
    yMin = Math.min(yMin, point.y);
    xMax = Math.max(xMax, point.x);
    yMax = Math.max(yMax, point.y);
  }

  return { xMin, yMin, xMax, yMax };
}

export function bboxOverlap(a, b) {
  return a.xMin <= b.xMax && a.xMax >= b.xMin &&
    a.yMin <= b.yMax && a.yMax >= b.yMin;
}

export function unionBbox(a, b) {
  if (!a) return b ? { ...b } : null;
  if (!b) return { ...a };

  return {
    xMin: Math.min(a.xMin, b.xMin),
    yMin: Math.min(a.yMin, b.yMin),
    xMax: Math.max(a.xMax, b.xMax),
    yMax: Math.max(a.yMax, b.yMax)
  };
}

export function padBbox(bbox, padding) {
  if (!bbox) return null;

  return {
    xMin: bbox.xMin - padding,
    yMin: bbox.yMin - padding,
    xMax: bbox.xMax + padding,
    yMax: bbox.yMax + padding
  };
}

export function polygonsIntersect(polyA, polyB) {
  for (let i = 0; i < polyA.length; i += 1) {
    const a1 = polyA[i];
    const a2 = polyA[(i + 1) % polyA.length];

    for (let j = 0; j < polyB.length; j += 1) {
      const b1 = polyB[j];
      const b2 = polyB[(j + 1) % polyB.length];

      if (segmentsIntersect(a1, a2, b1, b2)) {
        return true;
      }
    }
  }

  for (const point of polyA) {
    if (pointInPolygon(point, polyB)) return true;
  }

  for (const point of polyB) {
    if (pointInPolygon(point, polyA)) return true;
  }

  return false;
}

function segmentsIntersect(p1, p2, p3, p4) {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;
  const cross = d1x * d2y - d1y * d2x;

  if (Math.abs(cross) < 1e-10) return false;

  const dx = p3.x - p1.x;
  const dy = p3.y - p1.y;
  const t = (dx * d2y - dy * d2x) / cross;
  const u = (dx * d1y - dy * d1x) / cross;

  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

function pointInPolygon(point, poly) {
  let inside = false;

  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi);

    if (intersect) inside = !inside;
  }

  return inside;
}
