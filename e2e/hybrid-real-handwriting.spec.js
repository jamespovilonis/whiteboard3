import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  createCustomProblemDirectly,
  waitForE2EBridge,
  waitForRecognitionComplete
} from './helpers/playback.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('hybrid real-stroke linear equation trace segments, reads, and grades correctly', async ({ page }) => {
  const fixture = generateHybridLinearFixture();
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: repeatedLatex(fixture.expectedLatexLines),
    defaultLatex: 'x=4',
    problemStatus: 'correct'
  });

  await page.clock.install({ time: new Date('2026-07-03T12:00:00.000Z') });
  await page.goto('/');
  await waitForE2EBridge(page);
  const initial = await createCustomProblemDirectly(page, fixture.problemLatex, {
    problemType: 'equation-solving'
  });

  await injectTrace(page, fixture, initial.activeProblem);
  await page.clock.fastForward(1800);
  const recognized = await waitForRecognitionComplete(page, initial.activeProblem.id);
  const entry = recognized.recognitionResults.find((item) => item.problemId === initial.activeProblem.id);
  const result = entry?.recognition?.result || null;

  expect(result).not.toBeNull();
  expect(entry.recognition.status).toBe('complete');
  expect(result.lines.length).toBeGreaterThanOrEqual(1);
  expect(result.lines.map((line) => line.acceptedLatex || line.latex || '')).toContain('x=4');
  expect(result.grading?.result?.problemStatus).toBe('correct');
  expect(mockRecognition.endpoints()).toContain('/segment-lines');
  expect(mockRecognition.endpoints()).toContain('/recognize');
  expect(mockRecognition.endpoints()).toEqual(
    expect.arrayContaining([expect.stringMatching(/^\/grade-(equation|math)-work$/)])
  );

  const selectedKeys = (result.segmentation?.selected || []).map((candidate) => (
    strokeGroupKey(candidate.strokeIds || [])
  ));
  const expectedKeys = fixture.expectedLineGroups.map((group) => strokeGroupKey(group.strokeIds));
  expect(new Set(selectedKeys)).toEqual(new Set(expectedKeys));
});

function generateHybridLinearFixture() {
  const raw = execFileSync('python3', [
    'testing/realistic_handwriting.py',
    '--hybrid-linear',
    '--linear-a', '3',
    '--linear-b', '2',
    '--linear-x', '4',
    '--seed', '31',
    '--include-fixture'
  ], {
    cwd: ROOT,
    encoding: 'utf8',
    maxBuffer: 20 * 1024 * 1024
  });
  return JSON.parse(raw).fixture;
}

async function injectTrace(page, fixture, activeProblem) {
  await page.evaluate(() => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(true);
  });

  const dx = activeProblem.boardPosition.x + 8 - Number(fixture.answerBox?.xMin || 0);
  const dy = activeProblem.problemBox.yMax + 56 - Number(fixture.answerBox?.yMin || 0);
  const translated = fixture.strokes.map((stroke) => translateStroke(stroke, dx, dy));
  await page.evaluate((strokes) => {
    window.__whiteboardE2E?.replaceStrokes?.(strokes, 'hybrid-real-stroke-linear-equation');
  }, translated);
  await page.waitForFunction((count) => (
    window.__whiteboardE2E?.snapshot?.().strokes.length === count
  ), translated.length);
  await page.waitForFunction(() => (
    window.__whiteboardE2E?.snapshot?.().answerBox !== null
  ));

  await page.evaluate(() => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(false);
  });
}

function translateStroke(stroke, dx, dy) {
  return {
    ...stroke,
    rawPoints: shiftPoints(stroke.rawPoints, dx, dy),
    outlinePoints: shiftPoints(stroke.outlinePoints, dx, dy),
    canvasBbox: shiftBox(stroke.canvasBbox, dx, dy)
  };
}

function shiftPoints(points = [], dx, dy) {
  return points.map((point) => ({
    ...point,
    x: Number(point.x || 0) + dx,
    y: Number(point.y || 0) + dy
  }));
}

function shiftBox(box, dx, dy) {
  return {
    xMin: Number(box.xMin) + dx,
    yMin: Number(box.yMin) + dy,
    xMax: Number(box.xMax) + dx,
    yMax: Number(box.yMax) + dy
  };
}

function repeatedLatex(lines) {
  return Array.from({ length: 80 }, (_, index) => lines[index % lines.length] || 'x=4');
}

function strokeGroupKey(strokeIds = []) {
  return strokeIds.map(String).sort().join('|');
}
