import { expect, test } from '@playwright/test';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  getE2ESnapshot,
  waitForE2EBridge
} from './helpers/playback.js';

const CUSTOM_LATEX = '\\frac{x}{2} + 5 = 13';

test('opens a latex prompt before rendering a custom problem', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);

  await expect(page.getByTestId('latex-equation-input')).toBeVisible();
  const snapshot = await getE2ESnapshot(page);
  expect(snapshot.activeProblem).toBeNull();
  expect(snapshot.problemFlow.problems).toHaveLength(0);
});

test('renders user latex and sends it as recognition semantic context', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    defaultLatex: 'x = 16'
  });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);

  await page.getByTestId('latex-equation-input').fill(CUSTOM_LATEX);
  await page.getByTestId('latex-equation-submit').click();
  await waitForE2EBridge(page, { activeProblemLatex: CUSTOM_LATEX });

  const initial = await getE2ESnapshot(page);
  expect(initial.activeProblem.id).toBe('problem-1');
  expect(initial.activeProblem.latex).toBe(CUSTOM_LATEX);
  expect(initial.activeProblem.metadata.source).toBe('user-latex');
  await expect(page.locator('.problem-print[data-problem-id="problem-1"]')).toBeVisible();

  await drawAnswerStrokeInsideProblemBox(page, initial.activeProblem.problemBox);
  await page.getByTestId('submit-answer').click();
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().recognitionResults.some((entry) => (
      entry.recognition?.status === 'complete' || entry.recognition?.status === 'error'
    ))
  ));

  expect(mockRecognition.calls.some((call) => (
    call.postDataJson?.problemLatex === CUSTOM_LATEX ||
    call.postDataJson?.context?.problemLatex === CUSTOM_LATEX
  ))).toBe(true);

  const afterSubmit = await getE2ESnapshot(page);
  expect(afterSubmit.activeProblem.id).toBe('problem-1');
  expect(afterSubmit.activeProblem.status).toBe('submitted');
  expect(afterSubmit.problemFlow.awaitingEquation).toBe(false);
  await expect(page.getByTestId('latex-equation-input')).toHaveCount(0);

  await page.getByTestId('next-problem').click();
  const afterNext = await getE2ESnapshot(page);
  expect(afterNext.activeProblem).toBeNull();
  expect(afterNext.problemFlow.awaitingEquation).toBe(true);
  await expect(page.getByTestId('latex-equation-input')).toBeVisible();
});

async function waitForProblemSourceLoaded(page) {
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().events.some((event) => (
      event.type === 'problem-source-loaded'
    ))
  ));
}

async function drawAnswerStrokeInsideProblemBox(page, problemBox) {
  const start = await page.evaluate((point) => (
    window.__whiteboardE2E.boardToScreen(point)
  ), {
    x: problemBox.xMin + 40,
    y: problemBox.yMin + 28
  });
  const end = await page.evaluate((point) => (
    window.__whiteboardE2E.boardToScreen(point)
  ), {
    x: problemBox.xMin + 130,
    y: problemBox.yMin + 42
  });

  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 8 });
  await page.mouse.up();
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().answerBox !== null
  ));
}
