import { expect, test } from '@playwright/test';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  drawAnswerStrokeInsideProblemBox,
  enterHandwrittenProblem,
  getE2ESnapshot,
  submitHandwrittenProblemPreview,
  waitForE2EBridge,
  waitForProblemSourceLoaded,
  waitForRecognitionComplete
} from './helpers/playback.js';

const CUSTOM_LATEX = '\\frac{x}{2} + 5 = 13';

test('opens a handwriting prompt before rendering a custom problem', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);

  await expect(page.getByTestId('handwritten-problem-canvas')).toBeVisible();
  await expect(page.getByTestId('latex-equation-input')).toHaveCount(0);
  await expect(page.getByTestId('handwritten-problem-type-solve')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('handwritten-problem-type-evaluate')).toHaveAttribute('aria-pressed', 'false');
  const snapshot = await getE2ESnapshot(page);
  expect(snapshot.activeProblem).toBeNull();
  expect(snapshot.problemFlow.problems).toHaveLength(0);
});

test('renders user latex and sends it as recognition semantic context', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: [CUSTOM_LATEX, 'x = 16']
  });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);

  await submitHandwrittenProblemPreview(page, CUSTOM_LATEX);
  expect(mockRecognition.endpoints()).toContain('/recognize');
  await expect(page.getByTestId('handwritten-problem-action')).toHaveText('Solve');
  await expect(page.getByTestId('handwritten-problem-latex')).toHaveAttribute('data-latex', CUSTOM_LATEX);
  expect((await getE2ESnapshot(page)).activeProblem).toBeNull();
  await page.getByTestId('handwritten-problem-confirm').click();
  await waitForE2EBridge(page, { activeProblemLatex: CUSTOM_LATEX });
  const initial = await getE2ESnapshot(page);
  expect(initial.activeProblem.id).toBe('problem-1');
  expect(initial.activeProblem.latex).toBe(CUSTOM_LATEX);
  expect(initial.activeProblem.metadata.source).toBe('user-handwriting');
  expect(initial.activeProblem.metadata.problemType).toBe('equation-solving');
  await expect(page.locator('.problem-print[data-problem-id="problem-1"]')).toBeVisible();

  await drawAnswerStrokeInsideProblemBox(page, initial.activeProblem.problemBox);
  await expect(page.getByTestId('submit-answer')).toBeEnabled();
  await waitForRecognitionComplete(page, initial.activeProblem.id);

  expect(mockRecognition.calls.some((call) => (
    call.postDataJson?.problemLatex === CUSTOM_LATEX ||
    call.postDataJson?.context?.problemLatex === CUSTOM_LATEX
  ))).toBe(true);

  await expect(page.getByTestId('next-problem')).toBeDisabled();
  await page.getByTestId('submit-answer').click();
  const afterSubmit = await getE2ESnapshot(page);
  expect(afterSubmit.activeProblem.id).toBe('problem-1');
  expect(afterSubmit.activeProblem.status).toBe('submitted');
  expect(afterSubmit.problemFlow.awaitingEquation).toBe(false);
  await expect(page.getByTestId('latex-equation-input')).toHaveCount(0);

  await expect(page.getByTestId('next-problem')).toBeEnabled();
  await page.getByTestId('next-problem').click();
  const afterNext = await getE2ESnapshot(page);
  expect(afterNext.activeProblem).toBeNull();
  expect(afterNext.problemFlow.awaitingEquation).toBe(true);
  await expect(page.getByTestId('handwritten-problem-canvas')).toBeVisible();
});

test('creates evaluate-expression custom problems and sends the type to grading', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const latex = '\\frac{1}{2} + \\frac{2}{4}';
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: [latex, '1']
  });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);

  const initial = await enterHandwrittenProblem(page, latex, {
    problemType: 'evaluate-expression'
  });
  expect(initial.activeProblem.id).toBe('problem-1');
  expect(initial.activeProblem.kind).toBe('evaluate-expression');
  expect(initial.activeProblem.latex).toBe(latex);
  expect(initial.activeProblem.metadata.source).toBe('user-handwriting');
  expect(initial.activeProblem.metadata.problemType).toBe('evaluate-expression');
  await page.waitForFunction(() => (
    window.__whiteboardE2E?.snapshot?.().activeProblem?.initialGrading
  ));
  expect(mockRecognition.calls.some((call) => (
    call.endpoint === '/grade-math-work' &&
    call.postDataJson?.problemLatex === latex &&
    call.postDataJson?.problemMetadata?.problemType === 'evaluate-expression' &&
    (call.postDataJson?.lines || []).length === 0
  ))).toBe(true);

  await drawAnswerStrokeInsideProblemBox(page, initial.activeProblem.problemBox);
  await waitForRecognitionComplete(page, initial.activeProblem.id);

  expect(mockRecognition.calls.some((call) => (
    call.endpoint === '/grade-math-work' &&
    call.postDataJson?.problemLatex === latex &&
    call.postDataJson?.problemMetadata?.problemType === 'evaluate-expression'
  ))).toBe(true);
});
