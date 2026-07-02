import { expect, test } from '@playwright/test';
import { installMockRecognitionRoutes } from './helpers/mockRecognition.js';
import { installInvalidProblemSourceRoute } from './helpers/problemSource.js';
import {
  clickAndScrollCanvasDown,
  drawAnswerStrokeInsideProblemBox,
  enterHandwrittenProblem,
  getE2ESnapshot,
  waitForE2EBridge,
  waitForProblemSourceLoaded,
  waitForRecognitionComplete
} from './helpers/playback.js';

const CUSTOM_PROBLEMS = [
  {
    latex: '\\frac{x}{2} + 5 = 13',
    recognizedLatex: 'x = 16'
  },
  {
    latex: '\\sqrt{x + 9} = 7',
    recognizedLatex: 'x = 40'
  },
  {
    latex: '\\frac{3x - 2}{5} + \\frac{x + 1}{3} = 4',
    recognizedLatex: 'x = \\frac{23}{14}',
    scrollBeforeWriting: true
  }
];

test('solves multiple handwritten problems and can scroll the canvas for longer work', async ({ page }) => {
  await installInvalidProblemSourceRoute(page, { problems: [] });
  const mockRecognition = await installMockRecognitionRoutes(page, {
    latexLines: CUSTOM_PROBLEMS.flatMap((problem) => [
      problem.latex,
      problem.recognizedLatex
    ])
  });

  await page.goto('/');
  await waitForE2EBridge(page);
  await waitForProblemSourceLoaded(page);
  await expect(page.getByTestId('handwritten-problem-canvas')).toBeVisible();

  for (const [index, customProblem] of CUSTOM_PROBLEMS.entries()) {
    const problemId = `problem-${index + 1}`;
    const initial = await enterHandwrittenProblem(page, customProblem.latex);
    expect(initial.activeProblem.id).toBe(problemId);
    expect(initial.activeProblem.latex).toBe(customProblem.latex);
    expect(initial.activeProblem.metadata.source).toBe('user-handwriting');
    await expect(page.locator(`.problem-print[data-problem-id="${problemId}"]`)).toBeVisible();
    await expect(page.getByTestId('submit-answer')).toBeEnabled();

    let problemBox = initial.activeProblem.problemBox;
    if (customProblem.scrollBeforeWriting) {
      const scrolled = await clickAndScrollCanvasDown(page, 120);
      expect(scrolled.viewport.y).toBeGreaterThan(initial.viewport.y);
      expect(scrolled.strokes).toHaveLength(initial.strokes.length);
      expect(scrolled.activeProblem.id).toBe(problemId);
      problemBox = scrolled.activeProblem.problemBox;
    }

    await drawAnswerStrokeInsideProblemBox(page, problemBox, {
      yOffset: customProblem.scrollBeforeWriting ? 168 : 28,
      endYOffset: customProblem.scrollBeforeWriting ? 184 : 44,
      endXOffset: customProblem.scrollBeforeWriting ? 210 : 150
    });

    const recognized = await waitForRecognitionComplete(page, problemId);
    const entry = recognized.recognitionResults.find((result) => result.problemId === problemId);
    expect(entry?.recognition?.status).toBe('complete');
    expect(entry?.recognition?.result?.latex || '').toContain(customProblem.recognizedLatex);
    expect(mockRecognition.calls.some((call) => (
      call.postDataJson?.problemLatex === customProblem.latex ||
      call.postDataJson?.context?.problemLatex === customProblem.latex
    ))).toBe(true);

    await expect(page.getByTestId('next-problem')).toBeDisabled();
    await page.getByTestId('submit-answer').click();
    await expect(page.getByTestId('next-problem')).toBeEnabled();
    await page.getByTestId('next-problem').click();
    await page.waitForFunction(() => {
      const snapshot = window.__whiteboardE2E?.snapshot?.();
      return snapshot?.activeProblem === null && snapshot?.problemFlow?.awaitingEquation === true;
    });
    await expect(page.getByTestId('handwritten-problem-canvas')).toBeVisible();
  }

  const finalSnapshot = await getE2ESnapshot(page);
  expect(finalSnapshot.problemFlow.problems).toHaveLength(CUSTOM_PROBLEMS.length);
  expect(finalSnapshot.problemFlow.problems.every((problem) => problem.status === 'submitted')).toBe(true);
});
