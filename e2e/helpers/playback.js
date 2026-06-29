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

export async function waitForProblemSourceLoaded(page) {
  await page.waitForFunction(() => (
    window.__whiteboardE2E?.snapshot?.().events.some((event) => (
      event.type === 'problem-source-loaded'
    ))
  ));
}

export async function enterCustomLatexProblem(page, latex) {
  await expectBridge(page);
  await page.getByTestId('latex-equation-input').fill(latex);
  await page.getByTestId('latex-equation-submit').click();
  await waitForE2EBridge(page, { activeProblemLatex: latex });
  const snapshot = await getE2ESnapshot(page);
  await waitForActiveProblemOnCanvas(page, snapshot.activeProblem?.id || null);
  await page.locator('.problem-print').first().waitFor({ state: 'visible' });
  return getE2ESnapshot(page);
}

export async function drawAnswerStrokeInsideProblemBox(page, problemBox, options = {}) {
  const yOffset = options.yOffset ?? 28;
  const start = await page.evaluate((point) => (
    window.__whiteboardE2E.boardToScreen(point)
  ), {
    x: problemBox.xMin + (options.startXOffset ?? 40),
    y: problemBox.yMin + yOffset
  });
  const end = await page.evaluate((point) => (
    window.__whiteboardE2E.boardToScreen(point)
  ), {
    x: problemBox.xMin + (options.endXOffset ?? 150),
    y: problemBox.yMin + (options.endYOffset ?? yOffset + 16)
  });

  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: options.steps ?? 8 });
  await page.mouse.up();
  await waitForAnswerBox(page);
}

export async function clickAndScrollCanvasDown(page, deltaY = 520) {
  const before = await getE2ESnapshot(page);
  const canvas = before.canvas || { left: 0, top: 0, width: 1280, height: 900 };
  await page.keyboard.press('M');
  await page.locator('.whiteboard-stage.tool-mouse').waitFor({ state: 'visible' });
  await page.mouse.click(
    canvas.left + canvas.width / 2,
    canvas.top + canvas.height / 2
  );
  await page.mouse.wheel(0, deltaY);
  await page.waitForFunction((previousY) => (
    window.__whiteboardE2E?.snapshot?.().viewport.y > previousY
  ), before.viewport.y);
  await page.keyboard.press('P');
  await page.locator('.whiteboard-stage.tool-pen').waitFor({ state: 'visible' });
  return getE2ESnapshot(page);
}

export async function waitForRecognitionComplete(page, problemId = null) {
  await page.waitForFunction((expectedProblemId) => (
    window.__whiteboardE2E?.snapshot?.().recognitionResults.some((entry) => (
      (!expectedProblemId || entry.problemId === expectedProblemId) &&
      (entry.recognition?.status === 'complete' || entry.recognition?.status === 'error')
    ))
  ), problemId);
  return getE2ESnapshot(page);
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
      const drewStroke = await drawContour(page, contour, timings);
      if (!drewStroke) continue;
      expectedStrokeCount += 1;
      await waitForStrokeCount(page, expectedStrokeCount);
      await advanceTiming(page, timings.interStrokeGapMs, timings);
    }

    await advanceTiming(page, timings.linePauseMs, timings);
    snapshots.push(await waitForLineSnapshot(page, expectedStrokeCount));
  }

  return snapshots;
}

async function waitForAnswerBox(page) {
  await page.waitForFunction(() => (
    window.__whiteboardE2E?.snapshot?.().answerBox !== null
  ));
}

async function waitForActiveProblemOnCanvas(page, problemId) {
  if (!problemId) return;
  await page.waitForFunction((expectedProblemId) => {
    const bridge = window.__whiteboardE2E;
    const snapshot = bridge?.snapshot?.();
    const problem = snapshot?.activeProblem || null;
    if (!bridge || !problem || problem.id !== expectedProblemId) return false;

    const screenPoint = bridge.boardToScreen(problem.boardPosition);
    const canvas = snapshot.canvas || {};
    return screenPoint.y >= canvas.top + 16 &&
      screenPoint.y <= canvas.bottom - 120 &&
      screenPoint.x >= canvas.left + 16 &&
      screenPoint.x <= canvas.right - 120;
  }, problemId);
}

async function expectBridge(page) {
  await page.waitForFunction(() => Boolean(window.__whiteboardE2E?.snapshot));
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
      await advanceTiming(page, stroke.timing?.pauseAfterMs ?? timings.interStrokeGapMs, timings);
    }

    await advanceTiming(page, line.timing?.pauseAfterMs ?? timings.linePauseMs, timings);
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
  if (path.length < 2) return false;

  await drawPath(page, path, timings);
  return true;
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
  await advanceTiming(page, pointIntervalMs, timings);

  for (const point of points.slice(1)) {
    await page.mouse.move(point.x, point.y);
    await advanceTiming(page, pointIntervalMs, timings);
  }

  await page.mouse.up();
  await advanceTiming(page, timings.postStrokeFrameMs, timings);
}

async function advanceTiming(page, ms, timings = {}) {
  if (ms <= 0) return;
  if (timings.useRealTime) {
    await page.waitForTimeout(ms);
    return;
  }
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
