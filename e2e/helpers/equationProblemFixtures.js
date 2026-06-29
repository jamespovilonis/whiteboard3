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

export function renderEquationProblemFixture(name, options = {}) {
  const script = `
import json
import sys
from pathlib import Path
root = Path.cwd()
testing_dir = root / "testing"
sys.path.insert(0, str(testing_dir))
from fixture_catalog import build_board, fixture_payload, get_problem, line_gaps_for_pattern, placements_for
problem = get_problem(sys.argv[1])
spacing = sys.argv[2]
ink_style = sys.argv[3]
seed = int(sys.argv[4])
gap_pattern = sys.argv[5] if len(sys.argv) > 5 else ""
line_gaps = line_gaps_for_pattern(len(problem.lines), gap_pattern) if gap_pattern else None
board = build_board(problem.name, spacing=spacing, line_gaps=line_gaps, seed=seed, ink_style=ink_style)
_, _, _, gaps = placements_for(problem, spacing, line_gaps)
payload = fixture_payload(problem, spacing, gaps, board, ink_style=ink_style)
if gap_pattern:
    payload["fixture"]["gapPattern"] = gap_pattern
print(json.dumps(payload))
`;
  const output = execFileSync('python3', [
    '-c',
    script,
    name,
    options.spacing || 'standard',
    options.inkStyle || 'normal',
    String(options.seed ?? 101),
    options.gapPattern || ''
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024
  });
  return JSON.parse(output);
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
