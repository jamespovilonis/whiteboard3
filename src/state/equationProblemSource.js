import { normalizeProblemDefinitions } from './problemFlow.js';

const E2E_PROBLEM_SOURCE_URL = '/api/e2e/equation-solving-problems';

export const E2E_PROBLEM_SOURCE_ENABLED = import.meta.env?.VITE_E2E_TEST === '1';
export const TEST_PROBLEM_SOURCE_ENABLED = import.meta.env?.VITE_TEST_PROBLEM_SOURCE === '1';

export async function loadE2EEquationSolvingProblems() {
  if ((!E2E_PROBLEM_SOURCE_ENABLED && !TEST_PROBLEM_SOURCE_ENABLED) || typeof fetch !== 'function') {
    return [];
  }

  try {
    const response = await fetch(E2E_PROBLEM_SOURCE_URL, {
      headers: {
        Accept: 'application/json'
      }
    });
    if (!response.ok) return [];

    const payload = await response.json();
    return normalizeProblemDefinitions(payload?.problems);
  } catch (_error) {
    return [];
  }
}
