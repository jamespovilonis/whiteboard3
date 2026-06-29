import { expect, test } from '@playwright/test';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  drawAnswerStrokeInsideProblemBox,
  enterCustomLatexProblem,
  getE2ESnapshot,
  waitForE2EBridge,
  waitForProblemSourceLoaded,
  waitForRecognitionComplete
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

  const initial = await enterCustomLatexProblem(page, CUSTOM_LATEX);
  expect(initial.activeProblem.id).toBe('problem-1');
  expect(initial.activeProblem.latex).toBe(CUSTOM_LATEX);
  expect(initial.activeProblem.metadata.source).toBe('user-latex');
  await expect(page.locator('.problem-print[data-problem-id="problem-1"]')).toBeVisible();

  await drawAnswerStrokeInsideProblemBox(page, initial.activeProblem.problemBox);
  await expect(page.getByTestId('submit-answer')).toBeEnabled();
  await waitForRecognitionComplete(page, initial.activeProblem.id);

  expect(mockRecognition.calls.some((call) => (
    call.postDataJson?.problemLatex === CUSTOM_LATEX ||
    call.postDataJson?.context?.problemLatex === CUSTOM_LATEX
  ))).toBe(true);

  const afterSubmit = await getE2ESnapshot(page);
  expect(afterSubmit.activeProblem.id).toBe('problem-1');
  expect(afterSubmit.activeProblem.status).toBe('solving');
  expect(afterSubmit.problemFlow.awaitingEquation).toBe(false);
  await expect(page.getByTestId('latex-equation-input')).toHaveCount(0);

  await expect(page.getByTestId('next-problem')).toBeEnabled();
  await page.getByTestId('next-problem').click();
  const afterNext = await getE2ESnapshot(page);
  expect(afterNext.activeProblem).toBeNull();
  expect(afterNext.problemFlow.awaitingEquation).toBe(true);
  await expect(page.getByTestId('latex-equation-input')).toBeVisible();
});
