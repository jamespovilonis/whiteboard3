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
  await runHybridFixtureE2E(page, fixture);
});

test('hybrid real-stroke complex math trace segments, reads, and grades correctly', async ({ page }) => {
  const fixture = generateHybridComplexMathFixture();
  await runHybridFixtureE2E(page, fixture);
});

async function runHybridFixtureE2E(page, fixture) {
  const answerLatex = fixture.problemMetadata.answerLatex || 'x=4';
  const solution = String(fixture.problemMetadata.solution ?? (answerLatex.replace(/^x\s*=\s*/, '') || '4'));
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: repeatedLatex(fixture.expectedLatexLines),
    defaultLatex: answerLatex,
    problemStatus: 'correct',
    answerManifest: {
      problem_raw: fixture.problemLatex,
      responseKind: 'solution_set',
      variable: 'x',
      cardinality: 'finite',
      exact_set: [solution],
      decimal_set: [Number(solution)],
      tolerance: 0.005,
      acceptable_strings: [answerLatex, solution]
    }
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
  expect(result.lines.map((line) => line.acceptedLatex || line.latex || '')).toContain(answerLatex);
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
}

function generateHybridLinearFixture() {
  return generateHybridFixture([
    '--hybrid-linear',
    '--random-linear',
    '--seed', '31'
  ]);
}

function generateHybridComplexMathFixture() {
  return generateHybridFixture([
    '--hybrid-complex-math',
    '--seed', '41',
    '--include-crossout'
  ]);
}

function generateHybridFixture(args) {
  const raw = execFileSync('python3', [
    'testing/realistic_handwriting.py',
    ...args,
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
  await page.evaluate(({ strokes, fixtureKind }) => {
    window.__whiteboardE2E?.replaceStrokes?.(strokes, fixtureKind);
  }, { strokes: translated, fixtureKind: fixture.fixtureKind });
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
  const fallback = lines[lines.length - 1] || 'x=4';
  return Array.from({ length: 80 }, (_, index) => lines[index % lines.length] || fallback);
}

function strokeGroupKey(strokeIds = []) {
  return strokeIds.map(String).sort().join('|');
}
