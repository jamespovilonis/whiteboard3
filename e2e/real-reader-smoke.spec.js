import { expect, test } from '@playwright/test';
import { getEquationProblemFixture } from './helpers/equationProblemFixtures.js';
import { installProblemSourceRoute } from './helpers/problemSource.js';
import { buildEquationSolvingScenario } from './helpers/studentWritingScenario.js';
import {
  getE2ESnapshot,
  replayScenarioLines,
  waitForE2EBridge
} from './helpers/playback.js';

const REAL_READER_TIMEOUT_MS = 180_000;
const RECOGNITION_TIMEOUT_MS = 120_000;

test.describe.configure({ mode: 'serial' });

test('real reader recognizes temporally spaced handwriting replay', async ({ page }, testInfo) => {
  test.skip(
    process.env.REAL_OCR_E2E !== '1',
    'Set REAL_OCR_E2E=1 and VITE_OCR_API_URL=http://127.0.0.1:8010 to run the live reader smoke.'
  );
  test.setTimeout(REAL_READER_TIMEOUT_MS);

  const fixture = getEquationProblemFixture('algebra_prompt_context');
  await installProblemSourceRoute(page, [fixture.name]);

  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });

  const initial = await getE2ESnapshot(page);
  const scenario = buildEquationSolvingScenario(initial.activeProblem.boardPosition, {
    variant: 'clean',
    problem: fixture.problem
  });

  await setRealtimeRecognitionPaused(page, true);
  const snapshots = await replayScenarioLines(page, scenario, {
    timings: { useRealTime: true }
  });
  await setRealtimeRecognitionPaused(page, false);
  const afterWriting = snapshots[snapshots.length - 1];

  assertTimedInkCapture(afterWriting, scenario);
  await expect(page.getByTestId('submit-answer')).toBeDisabled();

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
  }, initial.activeProblem.id, { timeout: RECOGNITION_TIMEOUT_MS });

  const finalSnapshot = await getE2ESnapshot(page);
  const recognitionEntry = finalSnapshot.recognitionResults.find((entry) => (
    entry.problemId === initial.activeProblem.id
  ));
  const result = recognitionEntry?.recognition?.result || null;
  expect(result).not.toBeNull();
  expect(result.lines.length).toBeGreaterThan(0);
  expect(result.realtime?.components?.every((component) => (
    component.status === 'final' && !component.contested
  ))).toBe(true);

  const expectedLines = expectedLatexLines(fixture);
  const recognizedLines = result.lines.map((line) => (
    line.acceptedLatex || line.latex || ''
  ));
  expect(recognizedLines.map(normalizeLatex)).toEqual(
    expect.arrayContaining(expectedLines.map(normalizeLatex))
  );

  const cropStats = await recognitionCropStats(page);
  expect(cropStats.length).toBeGreaterThanOrEqual(result.lines.length);
  for (const crop of cropStats.slice(0, result.lines.length)) {
    expect(crop.loaded).toBe(true);
    expect(crop.width).toBeGreaterThan(20);
    expect(crop.height).toBeGreaterThan(10);
    expect(crop.darkPixels).toBeGreaterThan(10);
  }

  await expect(page.getByTestId('next-problem')).toBeEnabled();

  const screenshotPath = testInfo.outputPath('real-reader-final.png');
  await page.screenshot({ path: screenshotPath, fullPage: false });
  await testInfo.attach('real-reader-final', {
    path: screenshotPath,
    contentType: 'image/png'
  });
  await testInfo.attach('real-reader-lines', {
    body: JSON.stringify({ expectedLines, recognizedLines }, null, 2),
    contentType: 'application/json'
  });
});

function expectedLatexLines(fixture) {
  const configured = String(process.env.REAL_OCR_EXPECTED_LINES || '').trim();
  if (configured) {
    return configured
      .split(/\n|\|/)
      .map((line) => line.trim())
      .filter(Boolean);
  }
  return fixture.recognitionLines;
}

async function setRealtimeRecognitionPaused(page, paused) {
  await page.evaluate((nextPaused) => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(nextPaused);
  }, paused);
}

function assertTimedInkCapture(snapshot, scenario) {
  expect(snapshot.strokes).toHaveLength(scenario.strokes.length);
  expect(snapshot.answerBox).not.toBeNull();

  for (const [index, stroke] of snapshot.strokes.entries()) {
    expect(stroke.points.length).toBeGreaterThanOrEqual(2);
    expect(Number.isFinite(stroke.startTime)).toBe(true);
    expect(Number.isFinite(stroke.endTime)).toBe(true);
    expect(stroke.endTime).toBeGreaterThanOrEqual(stroke.startTime);
    expect(stroke.canvasBbox).toBeTruthy();

    if (index === 0) continue;
    expect(stroke.startTime).toBeGreaterThan(snapshot.strokes[index - 1].startTime);
  }

  const expectedFirst = bboxForPoints(scenario.strokes[0].points);
  const actualFirst = snapshot.strokes[0].canvasBbox;
  expect(Math.abs(centerX(actualFirst) - centerX(expectedFirst))).toBeLessThanOrEqual(18);
  expect(Math.abs(centerY(actualFirst) - centerY(expectedFirst))).toBeLessThanOrEqual(18);
}

async function recognitionCropStats(page) {
  await page.waitForFunction(() => (
    [...document.querySelectorAll('img.recognition-debug-crop')]
      .some((image) => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0)
  ), null, { timeout: 30_000 });

  return page.$$eval('img.recognition-debug-crop', (images) => (
    images.map((image) => {
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let darkPixels = 0;
      for (let index = 0; index < data.length; index += 4) {
        if (
          data[index + 3] > 0 &&
          data[index] < 96 &&
          data[index + 1] < 96 &&
          data[index + 2] < 96
        ) {
          darkPixels += 1;
        }
      }
      return {
        loaded: image.complete && image.naturalWidth > 0,
        width: image.naturalWidth,
        height: image.naturalHeight,
        darkPixels
      };
    })
  ));
}

function normalizeLatex(value) {
  return String(value || '')
    .replace(/\\left|\\right/g, '')
    .replace(/\\,/g, '')
    .replace(/\\cdot|\\times/g, '')
    .replace(/\s+/g, '')
    .trim();
}

function bboxForPoints(points) {
  return points.reduce((box, point) => ({
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

function centerX(bbox) {
  return (bbox.xMin + bbox.xMax) / 2;
}

function centerY(bbox) {
  return (bbox.yMin + bbox.yMax) / 2;
}
