import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const TESTING_DIR = path.dirname(fileURLToPath(import.meta.url));
export const REAL_HANDWRITING_FIXTURE_DIR = path.join(TESTING_DIR, 'fixtures', 'real_handwriting');

export function loadRealHandwritingFixtures(options = {}) {
  const fixtureDir = options.fixtureDir || REAL_HANDWRITING_FIXTURE_DIR;
  if (!existsSync(fixtureDir)) return [];
  return readdirSync(fixtureDir)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => loadRealHandwritingFixture(path.join(fixtureDir, name)));
}

export function loadRealHandwritingFixture(filePathOrSlug) {
  const filePath = filePathOrSlug.endsWith?.('.json')
    ? filePathOrSlug
    : path.join(REAL_HANDWRITING_FIXTURE_DIR, `${filePathOrSlug}.json`);
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

export function strokeGroupKey(strokeIds = []) {
  return (strokeIds || []).map(String).sort().join('|');
}

export function bboxForStrokes(strokes = []) {
  return (strokes || []).reduce((box, stroke) => {
    const item = stroke.canvasBbox || {};
    return {
      xMin: Math.min(box.xMin, Number(item.xMin)),
      yMin: Math.min(box.yMin, Number(item.yMin)),
      xMax: Math.max(box.xMax, Number(item.xMax)),
      yMax: Math.max(box.yMax, Number(item.yMax))
    };
  }, {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity
  });
}

export function padBbox(bbox, padding) {
  return {
    xMin: bbox.xMin - padding,
    yMin: bbox.yMin - padding,
    xMax: bbox.xMax + padding,
    yMax: bbox.yMax + padding
  };
}

export function translateRealTraceFixture(fixture, target = {}) {
  const dx = Number(target.x || 0) - Number(fixture.answerBox?.xMin || 0);
  const dy = Number(target.y || 0) - Number(fixture.answerBox?.yMin || 0);
  const translated = cloneJson(fixture);
  translated.answerBox = shiftBox(translated.answerBox, dx, dy);
  translated.problemBox = shiftBox(translated.problemBox, dx, dy);
  translated.strokes = (translated.strokes || []).map((stroke) => translateStroke(stroke, dx, dy));
  return translated;
}

export function expectedLineGroupKeys(fixture) {
  return (fixture.expectedLineGroups || []).map((group) => strokeGroupKey(group.strokeIds));
}

function translateStroke(stroke, dx, dy) {
  return {
    ...stroke,
    rawPoints: shiftPoints(stroke.rawPoints, dx, dy),
    outlinePoints: shiftPoints(stroke.outlinePoints, dx, dy),
    canvasBbox: shiftBox(stroke.canvasBbox, dx, dy)
  };
}

function shiftPoints(points = [], dx, dy) {
  return (points || []).map((point) => ({
    ...point,
    x: Number(point.x || 0) + dx,
    y: Number(point.y || 0) + dy
  }));
}

function shiftBox(box, dx, dy) {
  if (!box) return null;
  return {
    xMin: Number(box.xMin) + dx,
    yMin: Number(box.yMin) + dy,
    xMax: Number(box.xMax) + dx,
    yMax: Number(box.yMax) + dy
  };
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}
