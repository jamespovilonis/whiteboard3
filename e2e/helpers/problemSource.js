import { equationProblemDefinitions } from './equationProblemFixtures.js';

export const E2E_PROBLEM_SOURCE_PATH = '/api/e2e/equation-solving-problems';

export async function installProblemSourceRoute(page, fixtureNames) {
  const problems = equationProblemDefinitions(fixtureNames);
  const calls = [];

  await page.route(`**${E2E_PROBLEM_SOURCE_PATH}`, async (route) => {
    calls.push({
      method: route.request().method(),
      url: route.request().url()
    });
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ problems })
    });
  });

  return {
    calls,
    problems
  };
}

export async function installInvalidProblemSourceRoute(page, payload = { problems: [] }) {
  const calls = [];

  await page.route(`**${E2E_PROBLEM_SOURCE_PATH}`, async (route) => {
    calls.push({
      method: route.request().method(),
      url: route.request().url()
    });
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(payload)
    });
  });

  return { calls };
}
