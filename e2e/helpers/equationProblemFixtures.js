import { execFileSync } from 'node:child_process';

let cachedCatalog = null;

export function getEquationProblemFixture(name) {
  const fixture = equationProblemFixtures().find((entry) => entry.name === name);
  if (!fixture) {
    throw new Error(`Unknown testing equation problem fixture: ${name}`);
  }
  return fixture;
}

export function equationProblemDefinitions(names) {
  const fixtures = equationProblemFixtures();
  const selectedNames = new Set(names || []);
  return fixtures
    .filter((fixture) => !selectedNames.size || selectedNames.has(fixture.name))
    .map((fixture) => fixture.problem);
}

export function equationProblemFixtures() {
  if (!cachedCatalog) {
    cachedCatalog = loadTestingEquationProblemCatalog().map((problem) => ({
      name: problem.name || problem.id,
      problem,
      recognitionLines: problem.expectedLatexLines || [],
      scenario: {
        type: problem.id === 'algebra_prompt_context' ? 'linear-basic' : 'render-only'
      }
    }));
  }
  return cachedCatalog;
}

function loadTestingEquationProblemCatalog() {
  const output = execFileSync('python3', ['testing/export_equation_problem_catalog.py'], {
    cwd: process.cwd(),
    encoding: 'utf8'
  });
  return JSON.parse(output).problems || [];
}
