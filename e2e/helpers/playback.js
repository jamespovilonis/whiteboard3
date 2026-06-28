const DEFAULT_TIMINGS = Object.freeze({
  pointIntervalMs: 16,
  interStrokeGapMs: 96,
  linePauseMs: 650,
  postStrokeFrameMs: 16
});

export async function waitForE2EBridge(page, options = {}) {
  await page.waitForFunction((expectedLatex) => {
    const snapshot = window.__whiteboardE2E?.snapshot?.();
    if (!snapshot) return false;
    if (!expectedLatex) return true;
    return snapshot.activeProblem?.latex === expectedLatex;
  }, options.activeProblemLatex || null);
}

export async function getE2ESnapshot(page) {
  return page.evaluate(() => window.__whiteboardE2E.snapshot());
}

export async function replayFixtureLines(page, fixture, options = {}) {
  const timings = { ...DEFAULT_TIMINGS, ...(options.timings || {}) };
  const fromLineIndex = options.fromLineIndex ?? 0;
  const throughLineIndex = options.throughLineIndex ?? (fixture.lines || []).length - 1;
  const snapshots = [];
  let expectedStrokeCount = (await getE2ESnapshot(page)).strokes.length;

  for (let lineIndex = fromLineIndex; lineIndex <= throughLineIndex; lineIndex += 1) {
    const line = fixture.lines?.[lineIndex];
    if (!line) continue;

    for (const contour of line.contours || []) {
      await drawContour(page, contour, timings);
      expectedStrokeCount += 1;
      await waitForStrokeCount(page, expectedStrokeCount);
      await advanceClock(page, timings.interStrokeGapMs);
    }

    await advanceClock(page, timings.linePauseMs);
    snapshots.push(await waitForLineSnapshot(page, expectedStrokeCount));
  }

  return snapshots;
}

export async function replayScenarioLines(page, scenario, options = {}) {
  const timings = { ...DEFAULT_TIMINGS, ...(options.timings || {}) };
  const fromLineIndex = options.fromLineIndex ?? 0;
  const throughLineIndex = options.throughLineIndex ?? (scenario.rows || []).length - 1;
  const snapshots = [];
  let expectedStrokeCount = (await getE2ESnapshot(page)).strokes.length;

  for (let lineIndex = fromLineIndex; lineIndex <= throughLineIndex; lineIndex += 1) {
    const line = scenario.rows?.[lineIndex];
    if (!line) continue;

    for (const stroke of line.strokes || []) {
      await drawPath(page, stroke.points, timings, stroke.timing);
      expectedStrokeCount += 1;
      await waitForStrokeCount(page, expectedStrokeCount);
      await advanceClock(page, stroke.timing?.pauseAfterMs ?? timings.interStrokeGapMs);
    }

    await advanceClock(page, line.timing?.pauseAfterMs ?? timings.linePauseMs);
    snapshots.push(await waitForLineSnapshot(page, expectedStrokeCount));
  }

  return snapshots;
}

export async function waitForStrokeCount(page, expectedStrokeCount) {
  await page.waitForFunction((count) => {
    const snapshot = window.__whiteboardE2E?.snapshot?.();
    return (snapshot?.strokes || []).length >= count;
  }, expectedStrokeCount);
}

async function waitForLineSnapshot(page, expectedStrokeCount) {
  await page.waitForFunction((count) => {
    const snapshot = window.__whiteboardE2E?.snapshot?.();
    return (snapshot?.strokes || []).length >= count && Boolean(snapshot.answerBox);
  }, expectedStrokeCount);
  return getE2ESnapshot(page);
}

async function drawContour(page, contour, timings) {
  const path = usablePath(contour);
  if (path.length < 2) return;

  await drawPath(page, path, timings);
}

async function drawPath(page, path, timings, strokeTiming = null) {
  if (!path || path.length < 2) return;

  const points = await page.evaluate((boardPoints) => (
    boardPoints.map((point) => window.__whiteboardE2E.boardToScreen(point))
  ), densifyPath(path, 8));
  const pointIntervalMs = strokeTiming?.durationMs
    ? Math.max(4, strokeTiming.durationMs / Math.max(1, points.length))
    : timings.pointIntervalMs;

  await page.mouse.move(points[0].x, points[0].y);
  await page.mouse.down();
  await advanceClock(page, pointIntervalMs);

  for (const point of points.slice(1)) {
    await page.mouse.move(point.x, point.y);
    await advanceClock(page, pointIntervalMs);
  }

  await page.mouse.up();
  await advanceClock(page, timings.postStrokeFrameMs);
}

async function advanceClock(page, ms) {
  if (ms <= 0) return;
  await page.clock.fastForward(ms);
}

function usablePath(contour) {
  const points = (contour || [])
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y));
  if (points.length <= 24) return points;

  const step = Math.ceil(points.length / 24);
  const sampled = points.filter((_, index) => index % step === 0);
  const last = points[points.length - 1];
  if (sampled[sampled.length - 1] !== last) sampled.push(last);
  return sampled;
}

function densifyPath(points, spacing) {
  const out = [];
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / spacing));
    for (let step = 0; step < steps; step += 1) {
      const t = step / steps;
      out.push({
        x: start.x + dx * t,
        y: start.y + dy * t
      });
    }
  }
  out.push(points[points.length - 1]);
  return out;
}
