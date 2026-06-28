import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const PROBLEM_SOURCE_PATH = '/api/e2e/equation-solving-problems';
const ROOT_DIR = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [react(), testProblemSourcePlugin()]
});

function testProblemSourcePlugin() {
  return {
    name: 'whiteboard-test-problem-source',
    configureServer(server) {
      const enabled = process.env.VITE_E2E_TEST === '1' || process.env.VITE_TEST_PROBLEM_SOURCE === '1';
      if (!enabled) return;

      server.middlewares.use(PROBLEM_SOURCE_PATH, (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ detail: 'Method not allowed' }));
          return;
        }

        const problemNames = String(process.env.VITE_TEST_PROBLEM_FIXTURES || '')
          .split(',')
          .map((name) => name.trim())
          .filter(Boolean);
        const { problems } = loadTestingEquationProblems(problemNames);

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ problems }));
      });
    }
  };
}

function loadTestingEquationProblems(problemNames = []) {
  const output = execFileSync(
    'python3',
    ['testing/export_equation_problem_catalog.py', ...problemNames],
    {
      cwd: ROOT_DIR,
      encoding: 'utf8'
    }
  );
  return JSON.parse(output);
}
