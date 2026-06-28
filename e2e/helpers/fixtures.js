import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export function loadSyntheticFixture(name = 'algebra-prompt-context_standard') {
  const fixturePath = path.join(ROOT, 'testing', 'results', `${name}.json`);
  return JSON.parse(readFileSync(fixturePath, 'utf8'));
}

export function contourCountThroughLine(fixture, lineIndex) {
  return (fixture.lines || [])
    .slice(0, lineIndex + 1)
    .reduce((count, line) => count + (line.contours || []).length, 0);
}

export function totalContourCount(fixture) {
  return contourCountThroughLine(fixture, (fixture.lines || []).length - 1);
}
