import { expect, test } from '@playwright/test';
import { getEquationProblemFixture } from './helpers/equationProblemFixtures.js';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installProblemSourceRoute } from './helpers/problemSource.js';
import { buildEquationSolvingScenario } from './helpers/studentWritingScenario.js';
import {
  getE2ESnapshot,
  replayScenarioLines,
  waitForE2EBridge
} from './helpers/playback.js';

test('mocked recognition calls start after realtime debounce and return timing metadata', async ({ page }) => {
  const fixture = getEquationProblemFixture('algebra_prompt_context');
  await installProblemSourceRoute(page, [fixture.name]);
  await page.clock.install({ time: new Date('2026-06-28T12:00:00.000Z') });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: fixture.recognitionLines
  });

  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });

  const initial = await getE2ESnapshot(page);
  expect(initial.activeProblem.latex).toBe(fixture.problem.latex);
  expect(initial.activeProblem.metadata.family).toBe(fixture.problem.family);
  expect(initial.activeProblem.metadata.source).toBe('testing/fixture_catalog.py');
  expect(initial.activeProblem.metadata.expectedLatexLines).toEqual(fixture.recognitionLines);
  const scenario = buildEquationSolvingScenario(initial.activeProblem.boardPosition, {
    variant: 'clean',
    problem: fixture.problem
  });
  const snapshots = await replayScenarioLines(page, withShortLinePauses(scenario));
  const afterWriting = snapshots[snapshots.length - 1];

  expect(afterWriting.strokes).toHaveLength(scenario.strokes.length);
  expect(mockRecognition.calls).toHaveLength(0);
  await expect(page.getByTestId('submit-answer')).toBeEnabled();

  await page.clock.fastForward(650);
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().events.some((event) => (
      event.type === 'recognition-start'
    ))
  ));
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().recognitionResults.some((entry) => (
      entry.recognition?.status === 'complete'
    ))
  ));

  const afterRecognition = await getE2ESnapshot(page);
  const endpoints = mockRecognition.endpoints();

  expect(endpoints).toContain('/segment-lines');
  expect(endpoints).toContain('/recognize');
  expect(endpoints).toContain('/score-latex-candidates');
  expect(mockRecognition.calls.some((call) => (
    call.postDataJson?.problemLatex === fixture.problem.latex ||
    call.postDataJson?.context?.problemLatex === fixture.problem.latex
  ))).toBe(true);
  expect(mockRecognition.calls.some((call) => (
    call.postDataJson?.problemMetadata?.family === fixture.problem.family &&
    call.postDataJson?.problemMetadata?.source === 'testing/fixture_catalog.py'
  ))).toBe(true);

  const started = afterRecognition.events.findIndex((event) => event.type === 'recognition-start');
  const completed = afterRecognition.events.findIndex((event) => event.type === 'recognition-complete');
  expect(started).toBeGreaterThanOrEqual(0);
  expect(completed).toBeGreaterThan(started);
  await expect(page.getByTestId('next-problem')).toBeEnabled();

  const completedResult = afterRecognition.recognitionResults.find((entry) => (
    entry.recognition?.status === 'complete'
  ));
  const result = completedResult.recognition.result;
  expect(result.lines.length).toBeGreaterThan(0);
  expect(Number.isFinite(result.timing.totalElapsedSeconds)).toBe(true);

  for (const line of result.lines) {
    expect(line.timing).not.toBeNull();
    expect(Number.isFinite(line.timing.submitToFinalPredictionSeconds)).toBe(true);
    expect(Number.isFinite(line.timing.ocrElapsedSeconds)).toBe(true);
  }

  const answerBoxBeforeSubmit = afterRecognition.activeProblem.answerBox;
  const recognitionStartsBeforeSubmit = afterRecognition.events.filter((event) => (
    event.type === 'recognition-start'
  )).length;
  await page.getByTestId('submit-answer').click();
  await expect(page.getByTestId('submit-answer')).toBeDisabled();

  const afterSubmit = await getE2ESnapshot(page);
  expect(afterSubmit.activeProblem.status).toBe('submitted');
  expect(afterSubmit.activeProblem.answerBoxFrozen).toBe(true);

  await page.mouse.move(answerBoxBeforeSubmit.xMin + 16, answerBoxBeforeSubmit.yMax + 80);
  await page.mouse.down();
  await page.mouse.move(answerBoxBeforeSubmit.xMin + 180, answerBoxBeforeSubmit.yMax + 96, { steps: 4 });
  await page.mouse.up();
  await page.clock.fastForward(750);

  const afterLateInk = await getE2ESnapshot(page);
  expect(afterLateInk.activeProblem.status).toBe('submitted');
  expect(afterLateInk.activeProblem.answerBox).toEqual(answerBoxBeforeSubmit);
  expect(afterLateInk.events.filter((event) => event.type === 'recognition-start')).toHaveLength(
    recognitionStartsBeforeSubmit
  );
});

function withShortLinePauses(scenario) {
  return {
    ...scenario,
    rows: scenario.rows.map((row) => ({
      ...row,
      timing: {
        ...(row.timing || {}),
        pauseAfterMs: 120
      }
    }))
  };
}
