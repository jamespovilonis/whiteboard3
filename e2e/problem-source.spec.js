import { expect, test } from '@playwright/test';
import { getEquationProblemFixture } from './helpers/equationProblemFixtures.js';
import {
  installInvalidProblemSourceRoute,
  installProblemSourceRoute
} from './helpers/problemSource.js';
import {
  getE2ESnapshot,
  waitForE2EBridge
} from './helpers/playback.js';

test('renders equation-solving problems from the testing fixture catalog by default', async ({ page }) => {
  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: '2 x + 3 = 11' });

  const snapshot = await getE2ESnapshot(page);
  expect(snapshot.activeProblem.id).toBe('algebra_prompt_context');
  expect(snapshot.activeProblem.latex).toBe('2 x + 3 = 11');
});

test('renders a logarithmic equation from the Playwright problem source', async ({ page }) => {
  const fixture = getEquationProblemFixture('logarithmic_solve');
  await installProblemSourceRoute(page, [fixture.name]);

  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });

  const snapshot = await getE2ESnapshot(page);
  expect(snapshot.activeProblem.id).toBe(fixture.problem.id);
  expect(snapshot.activeProblem.latex).toBe(fixture.problem.latex);
  await expect(page.locator(`.problem-print[data-problem-id="${fixture.problem.id}"]`)).toBeVisible();
});

test('does not render original hard-coded problems when the E2E problem source is invalid', async ({ page }) => {
  const problemSource = await installInvalidProblemSourceRoute(page, {
    problems: [
      { id: 'simplify-ignored', kind: 'simplify', latex: 'x + x' },
      { id: 'missing-latex', kind: 'equation-solving' }
    ]
  });

  await page.goto('/');
  await waitForE2EBridge(page);
  await page.waitForFunction(() => (
    window.__whiteboardE2E.snapshot().events.some((event) => event.type === 'problem-source-loaded')
  ));

  const snapshot = await getE2ESnapshot(page);
  expect(problemSource.calls.length).toBeGreaterThanOrEqual(1);
  expect(snapshot.activeProblem).toBeNull();
  expect(snapshot.problemFlow.problems).toHaveLength(0);
});
