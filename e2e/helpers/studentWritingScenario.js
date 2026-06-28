const DEFAULT_LAYOUT = Object.freeze({
  promptGap: 140,
  lineGap: 132,
  glyphScale: 1.12,
  equalsOffsetX: 270,
  rhsOffsetX: 468,
  operationRightOffsetX: 300,
  finalRhsOffsetX: 468
});

const ROW_PAUSES_MS = [900, 720, 0];

export function buildEquationSolvingScenario(anchor, options = {}) {
  const variant = options.variant || 'clean';
  const promptLatex = options.promptLatex || options.problem?.latex;
  if (!promptLatex) {
    throw new Error('buildEquationSolvingScenario requires a problem or promptLatex');
  }
  const layout = {
    ...DEFAULT_LAYOUT,
    ...(options.layout || {})
  };
  const messy = variant === 'messy';
  const rand = mulberry32(options.seed ?? (messy ? 982451653 : 104729));
  const left = Math.round(anchor.x);
  const firstLineY = Math.round(anchor.y + layout.promptGap);
  const secondLineY = firstLineY + layout.lineGap;
  const thirdLineY = secondLineY + layout.lineGap;
  const equalsX = left + layout.equalsOffsetX;
  const columns = {
    leftOperandX: left,
    variableX: left + 112,
    equalsX,
    rhsX: equalsX + layout.rhsOffsetX - layout.equalsOffsetX,
    leftOperationX: left + 6,
    rightOperationX: left + layout.operationRightOffsetX,
    finalVariableX: left + 8,
    finalRhsX: equalsX + layout.finalRhsOffsetX - layout.equalsOffsetX
  };
  const scale = layout.glyphScale;
  const rows = [];

  rows.push(row(0, {
    role: 'equation-step',
    latex: '2 x = 8',
    columns: {
      leftOperandX: columns.leftOperandX,
      variableX: columns.variableX,
      equalsX: columns.equalsX,
      rhsX: columns.rhsX
    }
  }, [
    stroke('two', two(left, firstLineY, scale), { symbol: '2' }),
    ...glyphStrokes('x', xGlyph(left + 112, firstLineY + 5, scale)),
    ...equalsStrokes(equalsX, firstLineY + 34, scale, { lineIndex: 0 }),
    stroke('eight', eight(equalsX + layout.rhsOffsetX - layout.equalsOffsetX, firstLineY + 56, scale), { symbol: '8' })
  ]));

  rows.push(row(1, {
    role: 'operation-annotation',
    latex: '/ 2      / 2',
    operation: 'divide-by-2',
    columns: {
      leftOperationX: columns.leftOperationX,
      rightOperationX: columns.rightOperationX
    }
  }, [
    stroke('left-divide', slash(left + 6, secondLineY + 2, scale), { symbol: '/' }),
    stroke('left-two', two(left + 96, secondLineY + 12, scale * 0.9), { symbol: '2' }),
    stroke('right-divide', slash(left + layout.operationRightOffsetX, secondLineY + 2, scale), { symbol: '/' }),
    stroke('right-two', two(left + layout.operationRightOffsetX + 90, secondLineY + 12, scale * 0.9), { symbol: '2' })
  ]));

  rows.push(row(2, {
    role: 'final-answer',
    latex: 'x = 4',
    columns: {
      variableX: columns.finalVariableX,
      equalsX: columns.equalsX,
      rhsX: columns.finalRhsX
    }
  }, [
    ...glyphStrokes('x', xGlyph(left + 8, thirdLineY + 12, scale)),
    ...equalsStrokes(equalsX, thirdLineY + 45, scale, { lineIndex: 2 }),
    ...glyphStrokes('four', four(equalsX + layout.finalRhsOffsetX - layout.equalsOffsetX, thirdLineY, scale))
  ]));

  const normalizedRows = rows.map((entry, lineIndex) => ({
    ...entry,
    strokes: entry.strokes.map((item, strokeIndex) => (
      normalizeStroke(item, {
        messy,
        rand,
        lineIndex,
        strokeIndex
      })
    ))
  }));

  return {
    variant,
    promptLatex,
    equalsX,
    left,
    columns,
    rows: normalizedRows,
    strokes: normalizedRows.flatMap((entry) => entry.strokes)
  };
}

export function scenarioStrokeCountThroughLine(scenario, lineIndex) {
  return scenario.rows
    .slice(0, lineIndex + 1)
    .reduce((count, entry) => count + entry.strokes.length, 0);
}

export function scenarioEqualBarStrokes(scenario) {
  return scenario.strokes.filter((item) => item.kind === 'equals-bar');
}

function row(lineIndex, meta, strokes) {
  return {
    lineIndex,
    role: meta.role,
    latex: meta.latex,
    operation: meta.operation || null,
    columns: meta.columns || {},
    timing: {
      pauseAfterMs: ROW_PAUSES_MS[lineIndex] ?? 650
    },
    strokes: strokes.map((item) => ({
      ...item,
      lineIndex
    }))
  };
}

function stroke(id, points, meta = {}) {
  return {
    id,
    points,
    kind: meta.kind || 'glyph',
    symbol: meta.symbol || null,
    timing: meta.timing || strokeTiming(id, meta)
  };
}

function glyphStrokes(id, paths) {
  return paths.map((points, index) => stroke(`${id}-${index + 1}`, points, { symbol: id }));
}

function equalsStrokes(x, y, scale, meta = {}) {
  return equalsGlyph(x, y, scale).map((points, index) => ({
    id: `equals-${meta.lineIndex ?? 'line'}-${index + 1}`,
    points,
    kind: 'equals-bar',
    symbol: '=',
    equalsBar: index === 0 ? 'top' : 'bottom',
    expectedX: x,
    timing: {
      durationMs: 170,
      pauseAfterMs: index === 0 ? 70 : 180
    }
  }));
}

function normalizeStroke(item, { messy, rand, lineIndex, strokeIndex }) {
  const lineDrift = messy
    ? {
        x: (rand() - 0.5) * 14,
        y: (rand() - 0.5) * 10 + lineIndex * (rand() - 0.5) * 4
      }
    : { x: 0, y: 0 };
  const symbolDrift = messy && item.kind !== 'equals-bar'
    ? {
        x: (rand() - 0.5) * 28,
        y: (rand() - 0.5) * 20
      }
    : { x: 0, y: 0 };
  const crowdingDrift = messy ? messyCrowdingDrift(item) : { x: 0, y: 0 };
  const wobble = messy ? 11 : 0;
  const veer = messy && item.kind !== 'equals-bar'
    ? {
        x: (rand() - 0.5) * 18,
        y: (rand() - 0.5) * 16
      }
    : { x: 0, y: 0 };
  const points = item.points.map((point, pointIndex) => {
    const progress = item.points.length <= 1 ? 0 : pointIndex / (item.points.length - 1);
    const endpointTether = pointIndex === 0 || pointIndex === item.points.length - 1 ? 0.35 : 1;
    const xJitter = item.kind === 'equals-bar' ? 0 : (rand() - 0.5) * wobble * endpointTether;
    const yJitter = (rand() - 0.5) * wobble * endpointTether;
    return {
      x: point.x + lineDrift.x + symbolDrift.x + crowdingDrift.x + xJitter + veer.x * progress,
      y: point.y + lineDrift.y + symbolDrift.y + crowdingDrift.y + yJitter + veer.y * progress
    };
  });

  return {
    ...item,
    id: `${item.lineIndex + 1}-${strokeIndex + 1}-${item.id}`,
    timing: timingForVariant(item.timing, {
      messy,
      rand,
      lineIndex,
      strokeIndex
    }),
    points: item.kind === 'equals-bar'
      ? keepEqualsBarColumn(points, item.expectedX)
      : points
  };
}

function strokeTiming(id, meta = {}) {
  if (id === 'eight') return { durationMs: 760, pauseAfterMs: 170 };
  if (id === 'two' || id.endsWith('-two')) return { durationMs: 560, pauseAfterMs: 120 };
  if (id.includes('divide')) return { durationMs: 210, pauseAfterMs: 70 };
  if (meta.symbol === 'x') return { durationMs: 190, pauseAfterMs: 65 };
  if (meta.symbol === 'four') return { durationMs: 330, pauseAfterMs: 95 };
  return { durationMs: 280, pauseAfterMs: 90 };
}

function timingForVariant(timing, { messy, rand, lineIndex, strokeIndex }) {
  const base = timing || { durationMs: 280, pauseAfterMs: 90 };
  if (!messy) return { ...base };
  const durationScale = 1.08 + rand() * 0.24;
  const pauseWobble = (rand() - 0.35) * 90;
  const thinkingPause = lineIndex === 0 && strokeIndex >= 4 ? 90 : 0;
  return {
    durationMs: Math.round(base.durationMs * durationScale),
    pauseAfterMs: Math.max(40, Math.round(base.pauseAfterMs + pauseWobble + thinkingPause))
  };
}

function messyCrowdingDrift(item) {
  if (item.id === 'left-two' || item.id === 'right-two') {
    return { x: -56, y: 0 };
  }
  if (item.id === 'eight') {
    return { x: -10, y: 0 };
  }
  return { x: 0, y: 0 };
}

function keepEqualsBarColumn(points, expectedX) {
  if (!points.length || !Number.isFinite(expectedX)) return points;
  const dx = expectedX - points[0].x;
  return points.map((point) => ({
    x: point.x + dx,
    y: point.y
  }));
}

function two(x, y, s = 1) {
  return [
    ...arc(x + 34 * s, y + 18 * s, 30 * s, 22 * s, Math.PI * 1.08, Math.PI * 2.18, 18),
    { x: x + 58 * s, y: y + 42 * s },
    { x: x + 12 * s, y: y + 82 * s },
    { x: x + 72 * s, y: y + 82 * s }
  ];
}

function eight(x, y, s = 1) {
  const points = [];
  for (let index = 0; index <= 60; index += 1) {
    const t = (index / 60) * Math.PI * 2;
    points.push({
      x: x + 32 * s * Math.sin(2 * t),
      y: y + 48 * s * Math.sin(t)
    });
  }
  return points;
}

function xGlyph(x, y, s = 1) {
  return [
    [{ x, y }, { x: x + 62 * s, y: y + 72 * s }],
    [{ x: x + 62 * s, y }, { x, y: y + 72 * s }]
  ];
}

function equalsGlyph(x, y, s = 1) {
  return [
    [{ x, y }, { x: x + 76 * s, y }],
    [{ x, y: y + 31 * s }, { x: x + 76 * s, y: y + 31 * s }]
  ];
}

function slash(x, y, s = 1) {
  return [{ x: x + 54 * s, y }, { x, y: y + 88 * s }];
}

function four(x, y, s = 1) {
  return [
    [{ x: x + 58 * s, y }, { x: x + 10 * s, y: y + 54 * s }, { x: x + 78 * s, y: y + 54 * s }],
    [{ x: x + 58 * s, y }, { x: x + 58 * s, y: y + 92 * s }]
  ];
}

function arc(cx, cy, rx, ry, start, end, steps = 20) {
  const points = [];
  for (let index = 0; index <= steps; index += 1) {
    const t = start + (end - start) * (index / steps);
    points.push({
      x: cx + Math.cos(t) * rx,
      y: cy + Math.sin(t) * ry
    });
  }
  return points;
}

function mulberry32(seed) {
  return function rand() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
