import { expect, test } from '@playwright/test';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  enterCustomLatexProblem,
  getE2ESnapshot,
  waitForE2EBridge,
  waitForRecognitionComplete
} from './helpers/playback.js';
import {
  loadRealHandwritingFixture,
  translateRealTraceFixture
} from '../testing/real_handwriting_fixtures.mjs';

const MOCK_TRACE_SLUG = 'crossout-scratch-division';
const LIVE_TRACE_SLUG = 'compact-plus-minus-solution';
const LIVE_TIMEOUT_MS = 180_000;

test('mocked recognition replays a distilled real handwriting trace in the browser', async ({ page }) => {
  test.skip(
    process.env.REAL_OCR_E2E === '1',
    'Mock trace replay is covered by the default E2E run; live OCR mode uses the opt-in trace test.'
  );
  const fixture = loadRealHandwritingFixture(MOCK_TRACE_SLUG);
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: repeatedMockLatex(fixture),
    defaultLatex: fixture.fastLatexLines[0] || 'x',
    problemStatus: 'correct'
  });

  await page.clock.install({ time: new Date('2026-06-28T12:00:00.000Z') });
  await page.goto('/');
  await waitForE2EBridge(page);
  const initial = await enterCustomLatexProblem(page, fixture.problemLatex, {
    problemType: problemTypeFor(fixture)
  });

  await injectTrace(page, fixture, initial.activeProblem);
  await page.clock.fastForward(1300);
  const recognized = await waitForRecognitionComplete(page, initial.activeProblem.id);
  const entry = recognized.recognitionResults.find((item) => item.problemId === initial.activeProblem.id);
  const result = entry?.recognition?.result || null;

  expect(result).not.toBeNull();
  expect(result.lines.length).toBeGreaterThan(0);
  expect(result.segmentation.selected.length).toBeGreaterThan(0);
  expect(mockRecognition.endpoints()).toContain('/segment-lines');
  expect(mockRecognition.endpoints()).toContain('/recognize');
  expect(recognized.answerBox).not.toBeNull();
  expect(recognized.answerBox.yMax).toBeGreaterThan(recognized.answerBox.yMin);
  expect(recognized.strokes).toHaveLength(fixture.strokes.length);
});

test('live OCR can replay a distilled real handwriting trace', async ({ page }, testInfo) => {
  test.skip(
    process.env.REAL_OCR_E2E !== '1',
    'Set REAL_OCR_E2E=1 and VITE_OCR_API_URL=http://127.0.0.1:8010 to run live trace replay.'
  );
  test.setTimeout(LIVE_TIMEOUT_MS);

  const fixture = loadRealHandwritingFixture(LIVE_TRACE_SLUG);
  await installInvalidProblemSourceRoute(page, { problems: [] });
  await page.goto('/');
  await waitForE2EBridge(page);
  const initial = await enterCustomLatexProblem(page, fixture.problemLatex, {
    problemType: problemTypeFor(fixture)
  });

  await injectTrace(page, fixture, initial.activeProblem);
  const recognized = await waitForRecognitionComplete(page, initial.activeProblem.id);
  const entry = recognized.recognitionResults.find((item) => item.problemId === initial.activeProblem.id);
  const result = entry?.recognition?.result || null;

  expect(result).not.toBeNull();
  expect(entry.recognition.status).toBe('complete');
  expect(result.lines.length).toBeGreaterThan(0);
  await testInfo.attach(`${fixture.slug}-live-lines`, {
    body: JSON.stringify({
      expectedLatexLines: fixture.expectedLatexLines,
      fastLatexLines: fixture.fastLatexLines,
      recognizedLines: result.lines.map((line) => line.acceptedLatex || line.latex || '')
    }, null, 2),
    contentType: 'application/json'
  });
});

async function injectTrace(page, fixture, activeProblem) {
  await page.evaluate(() => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(true);
  });

  const target = {
    x: activeProblem.boardPosition.x + 8,
    y: activeProblem.problemBox.yMax + 56
  };
  const translated = translateRealTraceFixture(fixture, target);
  await page.evaluate((strokes) => {
    window.__whiteboardE2E?.replaceStrokes?.(strokes, 'real-handwriting-trace');
  }, translated.strokes);
  await page.waitForFunction((count) => (
    window.__whiteboardE2E?.snapshot?.().strokes.length === count
  ), translated.strokes.length);
  await page.waitForFunction(() => (
    window.__whiteboardE2E?.snapshot?.().answerBox !== null
  ));

  await page.evaluate(() => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(false);
  });
}

function problemTypeFor(fixture) {
  return String(fixture.problemLatex || '').includes('=')
    ? 'equation-solving'
    : 'evaluate-expression';
}

function repeatedMockLatex(fixture) {
  const labels = fixture.expectedLineGroups?.map((group) => group.latex).filter(Boolean) || [];
  const fallback = fixture.fastLatexLines?.length ? fixture.fastLatexLines : fixture.expectedLatexLines;
  const values = labels.length ? labels : fallback;
  return Array.from({ length: 40 }, (_, index) => values[index % values.length] || 'x');
}
