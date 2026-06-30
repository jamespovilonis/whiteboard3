import { expect, test } from '@playwright/test';
import {
  getEquationProblemFixture,
  renderEquationProblemFixture
} from './helpers/equationProblemFixtures.js';
import { installProblemSourceRoute } from './helpers/problemSource.js';
import {
  getE2ESnapshot,
  waitForE2EBridge
} from './helpers/playback.js';
import { BOARD_SIZE } from '../src/whiteboard/constants.js';

const REAL_MATRIX_TIMEOUT_MS = 420_000;
const RECOGNITION_TIMEOUT_MS = 150_000;

const CASES = [
  {
    name: 'algebra_prompt_context',
    spacing: 'dense',
    inkStyle: 'messy',
    gapPattern: 'pinched-middle',
    seed: 616
  },
  {
    name: 'rational_quadratic_solve',
    spacing: 'standard',
    inkStyle: 'compact',
    seed: 1720
  },
  {
    name: 'logarithmic_solve',
    spacing: 'mixed',
    inkStyle: 'loose',
    gapPattern: 'pinched-middle',
    seed: 421
  },
  {
    name: 'quadratic_formula_positive_root',
    spacing: 'dense',
    inkStyle: 'messy',
    gapPattern: 'stair-step',
    seed: 808
  }
];

test.describe.configure({ mode: 'serial' });

for (const item of CASES) {
  test(`real reader preserves one-shot accuracy on rendered fixture: ${item.name}`, async ({ page }, testInfo) => {
    test.skip(
      process.env.REAL_OCR_E2E !== '1',
      'Set REAL_OCR_E2E=1 and VITE_API_URL=http://127.0.0.1:8010 to run the live reader matrix.'
    );
    test.fail(Boolean(item.expectedFailure), item.expectedFailure || '');
    test.setTimeout(REAL_MATRIX_TIMEOUT_MS);

    const fixture = getEquationProblemFixture(item.name);
    const rendered = translateFixtureToAnswerArea(
      renderEquationProblemFixture(item.name, item),
      await startProblem(page, fixture)
    );

    await setRealtimeRecognitionPaused(page, true);
    const afterWriting = await injectFixtureLines(page, rendered);
    await setRealtimeRecognitionPaused(page, false);

    expect(afterWriting.strokes.length).toBeGreaterThanOrEqual(rendered.lines.length);
    expect(afterWriting.answerBox).not.toBeNull();
    await expect(page.getByTestId('submit-answer')).toBeEnabled();

    await page.waitForFunction(() => (
      window.__whiteboardE2E.snapshot().events.some((event) => (
        event.type === 'recognition-start'
      ))
    ), null, { timeout: 30_000 });

    await page.waitForFunction((problemId) => {
      const entry = window.__whiteboardE2E.snapshot().recognitionResults.find((result) => (
        result.problemId === problemId
      ));
      return entry?.recognition?.status === 'complete' &&
        entry.recognition?.result?.realtime?.allFinal !== false;
    }, afterWriting.activeProblem.id, { timeout: RECOGNITION_TIMEOUT_MS });

    const finalSnapshot = await getE2ESnapshot(page);
    const recognitionEntry = finalSnapshot.recognitionResults.find((entry) => (
      entry.problemId === afterWriting.activeProblem.id
    ));
    const result = recognitionEntry?.recognition?.result || null;
    expect(result).not.toBeNull();
    expect(result.lines.length).toBeGreaterThanOrEqual(rendered.fixture.expectedLatexLines.length);
    expect(result.realtime?.components?.every((component) => (
      component.status === 'final' && !component.contested
    ))).toBe(true);

    const expectedLines = rendered.fixture.expectedLatexLines;
    const recognizedLines = result.lines.map((line) => line.acceptedLatex || line.latex || '');
    expect(recognizedLines.map(normalizeLatex)).toEqual(
      expect.arrayContaining(expectedLines.map(normalizeLatex))
    );
    await expect(page.getByTestId('next-problem')).toBeDisabled();
    await page.getByTestId('submit-answer').click();
    await expect(page.getByTestId('next-problem')).toBeEnabled();

    const screenshotPath = testInfo.outputPath(`${item.name}-final.png`);
    await page.screenshot({ path: screenshotPath, fullPage: false });
    await testInfo.attach(`${item.name}-final`, {
      path: screenshotPath,
      contentType: 'image/png'
    });
    await testInfo.attach(`${item.name}-lines`, {
      body: JSON.stringify({ expectedLines, recognizedLines }, null, 2),
      contentType: 'application/json'
    });
  });
}

async function startProblem(page, fixture) {
  await installProblemSourceRoute(page, [fixture.name]);
  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });
  return getE2ESnapshot(page);
}

async function setRealtimeRecognitionPaused(page, paused) {
  await page.evaluate((nextPaused) => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(nextPaused);
  }, paused);
}

async function injectFixtureLines(page, fixture) {
  const strokes = [];
  let timestamp = Date.now();

  for (const [lineIndex, line] of (fixture.lines || []).entries()) {
    for (const [contourIndex, contour] of (line.contours || []).entries()) {
      const stroke = strokeFromContour(contour, {
        id: `rendered_${lineIndex + 1}_${contourIndex + 1}`,
        startTime: timestamp,
        lineIndex,
        latex: line.latex
      });
      if (!stroke) continue;
      strokes.push(stroke);
      timestamp += 34;
    }

    await page.evaluate((nextStrokes) => {
      window.__whiteboardE2E?.replaceStrokes?.(nextStrokes, 'e2e-rendered-fixture-line');
    }, strokes);
    await page.waitForFunction((count) => (
      window.__whiteboardE2E?.snapshot?.().strokes.length >= count
    ), strokes.length);
    await page.waitForTimeout(650);
    timestamp += 650;
  }

  return getE2ESnapshot(page);
}

function strokeFromContour(contour, { id, startTime, lineIndex, latex }) {
  const rawPoints = (contour || [])
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y))
    .map((point) => ({
      x: point.x,
      y: point.y,
      pressure: 0.5
    }));
  if (rawPoints.length < 2) return null;

  const canvasBbox = bboxForPoints(rawPoints);
  return {
    id,
    startTime,
    endTime: startTime + 24,
    points: rawPoints.map((point, index) => ({
      x: clamp(point.x / BOARD_SIZE.width, 0, 1),
      y: clamp(point.y / BOARD_SIZE.height, 0, 1),
      t: index,
      pressure: point.pressure
    })),
    rawPoints,
    outlinePoints: rawPoints.map((point) => ({ x: point.x, y: point.y })),
    color: '#000000',
    canvasBbox,
    bbox: {
      xMin: clamp(canvasBbox.xMin / BOARD_SIZE.width, 0, 1),
      yMin: clamp(canvasBbox.yMin / BOARD_SIZE.height, 0, 1),
      xMax: clamp(canvasBbox.xMax / BOARD_SIZE.width, 0, 1),
      yMax: clamp(canvasBbox.yMax / BOARD_SIZE.height, 0, 1)
    },
    relationsToPrev: { dx: 0, dy: 0, dt: 0, overlapRatio: 0 },
    syntheticLineIndex: lineIndex,
    syntheticLatex: latex
  };
}

function translateFixtureToAnswerArea(fixture, snapshot) {
  const active = snapshot.activeProblem;
  const sourceBox = bboxForFixture(fixture);
  const targetX = active.boardPosition.x + 8;
  const targetY = active.problemBox.yMax + 64;
  const dx = targetX - sourceBox.xMin;
  const dy = targetY - sourceBox.yMin;

  return {
    ...fixture,
    lines: fixture.lines.map((line) => ({
      ...line,
      x: line.x + dx,
      y: line.y + dy,
      bbox: shiftBbox(line.bbox, dx, dy),
      contours: line.contours.map((contour) => (
        contour.map((point) => ({
          x: point.x + dx,
          y: point.y + dy
        }))
      ))
    }))
  };
}

function bboxForFixture(fixture) {
  return (fixture.lines || []).reduce((box, line) => ({
    xMin: Math.min(box.xMin, line.bbox.xMin),
    yMin: Math.min(box.yMin, line.bbox.yMin),
    xMax: Math.max(box.xMax, line.bbox.xMax),
    yMax: Math.max(box.yMax, line.bbox.yMax)
  }), {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity
  });
}

function shiftBbox(bbox, dx, dy) {
  return {
    xMin: bbox.xMin + dx,
    yMin: bbox.yMin + dy,
    xMax: bbox.xMax + dx,
    yMax: bbox.yMax + dy
  };
}

function bboxForPoints(points) {
  return (points || []).reduce((box, point) => ({
    xMin: Math.min(box.xMin, point.x),
    yMin: Math.min(box.yMin, point.y),
    xMax: Math.max(box.xMax, point.x),
    yMax: Math.max(box.yMax, point.y)
  }), {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeLatex(value) {
  return String(value || '')
    .replace(/\\left|\\right/g, '')
    .replace(/\\,/g, '')
    .replace(/\\cdot|\\times/g, '')
    .replace(/\s+/g, '')
    .trim();
}
