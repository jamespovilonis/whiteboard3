import { expect, test } from '@playwright/test';
import { getEquationProblemFixture } from './helpers/equationProblemFixtures.js';
import { installProblemSourceRoute } from './helpers/problemSource.js';
import {
  buildEquationSolvingScenario,
  scenarioEqualBarStrokes,
  scenarioStrokeCountThroughLine
} from './helpers/studentWritingScenario.js';
import {
  getE2ESnapshot,
  replayScenarioLines,
  waitForE2EBridge
} from './helpers/playback.js';

test('replays temporally spaced student handwriting into app stroke state', async ({ page }) => {
  const fixture = getEquationProblemFixture('algebra_prompt_context');
  await installProblemSourceRoute(page, [fixture.name]);
  await page.clock.install({ time: new Date('2026-06-28T12:00:00.000Z') });
  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });
  await pauseRealtimeRecognition(page);

  const initial = await getE2ESnapshot(page);
  expect(initial.activeProblem.latex).toBe(fixture.problem.latex);
  const scenario = buildScenarioFromSnapshot(initial, fixture, 'clean');
  assertWorkedEquationStructure(scenario, fixture.problem.latex);
  expect(scenario.promptLatex).toBe(initial.activeProblem.latex);
  assertScenarioStartsUnderProblem(scenario, initial);
  const [afterLine1] = await replayScenarioLines(page, scenario, { throughLineIndex: 0 });
  const [afterLine2] = await replayScenarioLines(page, scenario, {
    fromLineIndex: 1,
    throughLineIndex: 1
  });
  const [afterFinalLine] = await replayScenarioLines(page, scenario, {
    fromLineIndex: 2,
    throughLineIndex: 2
  });

  expect(afterLine1.strokes).toHaveLength(scenarioStrokeCountThroughLine(scenario, 0));
  expect(afterLine2.strokes).toHaveLength(scenarioStrokeCountThroughLine(scenario, 1));
  expect(afterFinalLine.strokes).toHaveLength(scenario.strokes.length);
  assertPartialProgressSnapshots([afterLine1, afterLine2, afterFinalLine], scenario);

  expect(afterLine1.answerBox).not.toBeNull();
  expect(afterLine2.answerBox.yMax).toBeGreaterThan(afterLine1.answerBox.yMax);
  expect(afterFinalLine.answerBox.yMax).toBeGreaterThan(afterLine2.answerBox.yMax);
  expect(afterFinalLine.answerBox.xMax).toBeGreaterThan(afterFinalLine.answerBox.xMin);

  assertStrokeTiming(afterFinalLine.strokes);
  assertScenarioTimingDynamics(afterFinalLine.strokes, scenario);
  assertEqualsBarsAligned(afterFinalLine.strokes, scenario);
  expect(afterFinalLine.events.filter((event) => event.type === 'pen-stroke-start')).toHaveLength(scenario.strokes.length);
  expect(afterFinalLine.events.filter((event) => event.type === 'stroke-finalized')).toHaveLength(scenario.strokes.length);
});

test('messy handwriting still preserves the aligned equation-bar column', async ({ page }) => {
  const fixture = getEquationProblemFixture('algebra_prompt_context');
  await installProblemSourceRoute(page, [fixture.name]);
  await page.clock.install({ time: new Date('2026-06-28T12:00:00.000Z') });
  await page.goto('/');
  await waitForE2EBridge(page, { activeProblemLatex: fixture.problem.latex });
  await pauseRealtimeRecognition(page);

  const initial = await getE2ESnapshot(page);
  expect(initial.activeProblem.latex).toBe(fixture.problem.latex);
  const cleanScenario = buildScenarioFromSnapshot(initial, fixture, 'clean');
  const messyScenario = buildScenarioFromSnapshot(initial, fixture, 'messy');
  assertWorkedEquationStructure(cleanScenario, fixture.problem.latex);
  assertWorkedEquationStructure(messyScenario, fixture.problem.latex);
  expect(messyScenario.strokes.length).toBe(cleanScenario.strokes.length);
  expect(averagePointDeviation(cleanScenario, messyScenario)).toBeGreaterThan(7);
  expect(averageEndpointDeviation(cleanScenario, messyScenario)).toBeGreaterThan(7);
  assertMessyOperationPairsCrowd(cleanScenario, messyScenario);

  const snapshots = await replayScenarioLines(page, messyScenario);
  const afterWriting = snapshots[snapshots.length - 1];

  expect(afterWriting.strokes).toHaveLength(messyScenario.strokes.length);
  assertPartialProgressSnapshots(snapshots, messyScenario);
  assertScenarioTimingDynamics(afterWriting.strokes, messyScenario);
  assertEqualsBarsAligned(afterWriting.strokes, messyScenario, 7);
});

function assertStrokeTiming(strokes) {
  for (const [index, stroke] of strokes.entries()) {
    expect(Number.isFinite(stroke.startTime)).toBe(true);
    expect(Number.isFinite(stroke.endTime)).toBe(true);
    expect(stroke.endTime).toBeGreaterThanOrEqual(stroke.startTime);
    expect(stroke.points.length).toBeGreaterThanOrEqual(2);
    expect(stroke.points[0].t).toBe(0);

    let previousPointTime = 0;
    for (const point of stroke.points.slice(1)) {
      expect(point.t).toBeGreaterThanOrEqual(0);
      expect(point.t).toBeGreaterThanOrEqual(previousPointTime);
      previousPointTime = point.t;
    }

    if (index === 0) {
      expect(stroke.relationsToPrev.dt).toBe(0);
      continue;
    }

    const previous = strokes[index - 1];
    expect(stroke.startTime).toBeGreaterThan(previous.startTime);
    expect(stroke.relationsToPrev.dt).toBe(stroke.startTime - previous.startTime);
  }
}

function assertScenarioTimingDynamics(strokes, scenario) {
  const firstStroke = strokes[0];
  const finalStroke = strokes[strokes.length - 1];
  expect(finalStroke.endTime - firstStroke.startTime).toBeGreaterThan(6000);

  const eight = renderedStrokeByScenarioId(strokes, scenario, 'eight');
  const slash = renderedStrokeByScenarioId(strokes, scenario, 'left-divide');
  const two = renderedStrokeByScenarioId(strokes, scenario, 'left-two');
  expect(strokeDuration(eight)).toBeGreaterThan(strokeDuration(slash) + 300);
  expect(strokeDuration(two)).toBeGreaterThan(strokeDuration(slash) + 180);

  const firstBoundary = lineBoundaryStartGap(strokes, scenario, 0, 1);
  const secondBoundary = lineBoundaryStartGap(strokes, scenario, 1, 2);
  const withinLineGap = maxWithinLineStartGap(strokes, scenario);
  expect(firstBoundary).toBeGreaterThan(withinLineGap + 600);
  expect(secondBoundary).toBeGreaterThan(withinLineGap + 450);
}

function assertWorkedEquationStructure(scenario, promptLatex) {
  expect(scenario.promptLatex).toBe(promptLatex);
  expect(scenario.rows.map((row) => row.role)).toEqual([
    'equation-step',
    'operation-annotation',
    'final-answer'
  ]);
  expect(scenario.rows.map((row) => row.latex)).toEqual([
    '2 x = 8',
    '/ 2      / 2',
    'x = 4'
  ]);

  const [firstRow, operationRow, finalRow] = scenario.rows;
  expect(firstRow.columns.equalsX).toBe(scenario.columns.equalsX);
  expect(finalRow.columns.equalsX).toBe(scenario.columns.equalsX);
  expect(operationRow.operation).toBe('divide-by-2');
  expect(operationRow.columns.leftOperationX).toBeLessThan(scenario.columns.equalsX);
  expect(operationRow.columns.rightOperationX).toBeGreaterThanOrEqual(scenario.columns.equalsX);

  const firstTwo = strokeBoxBySuffix(firstRow.strokes, 'two');
  const firstX = strokeBoxBySuffix(firstRow.strokes, 'x-1');
  const firstEquals = bboxForStrokes(firstRow.strokes.filter((stroke) => stroke.kind === 'equals-bar'));
  const firstEight = strokeBoxBySuffix(firstRow.strokes, 'eight');
  expect(centerX(firstTwo)).toBeLessThan(centerX(firstX));
  expect(centerX(firstX)).toBeLessThan(centerX(firstEquals));
  expect(centerX(firstEquals)).toBeLessThan(centerX(firstEight));

  const finalX = strokeBoxBySuffix(finalRow.strokes, 'x-1');
  const finalEquals = bboxForStrokes(finalRow.strokes.filter((stroke) => stroke.kind === 'equals-bar'));
  const finalFour = bboxForStrokes(finalRow.strokes.filter((stroke) => stroke.id.includes('four')));
  expect(centerX(finalX)).toBeLessThan(centerX(finalEquals));
  expect(centerX(finalEquals)).toBeLessThan(centerX(finalFour));
  expect(Math.abs(centerX(firstEquals) - centerX(finalEquals))).toBeLessThanOrEqual(4);
}

function assertPartialProgressSnapshots(snapshots, scenario) {
  expect(snapshots).toHaveLength(scenario.rows.length);

  for (const [lineIndex, snapshot] of snapshots.entries()) {
    expect(snapshot.strokes).toHaveLength(scenarioStrokeCountThroughLine(scenario, lineIndex));
    expect(snapshot.answerBox).not.toBeNull();
  }

  expect(snapshots[0].strokes.length).toBeLessThan(scenario.strokes.length);
  expect(snapshots[1].strokes.length).toBeLessThan(scenario.strokes.length);
  expect(snapshots[0].answerBox.yMax).toBeLessThan(snapshots[1].answerBox.yMax);
  expect(snapshots[1].answerBox.yMax).toBeLessThan(snapshots[2].answerBox.yMax);
}

function buildScenarioFromSnapshot(snapshot, fixture, variant) {
  const anchor = snapshot.activeProblem?.boardPosition || { x: 240, y: -24 };
  return buildEquationSolvingScenario(anchor, {
    variant,
    problem: fixture.problem
  });
}

function renderedStrokeByScenarioId(strokes, scenario, idSuffix) {
  const index = scenario.strokes.findIndex((item) => item.id.endsWith(idSuffix));
  expect(index).toBeGreaterThanOrEqual(0);
  return strokes[index];
}

function strokeDuration(stroke) {
  return stroke.endTime - stroke.startTime;
}

function lineBoundaryStartGap(strokes, scenario, fromLineIndex, toLineIndex) {
  const fromLastIndex = scenarioStrokeCountThroughLine(scenario, fromLineIndex) - 1;
  const toFirstIndex = scenarioStrokeCountThroughLine(scenario, toLineIndex - 1);
  return strokes[toFirstIndex].startTime - strokes[fromLastIndex].startTime;
}

function maxWithinLineStartGap(strokes, scenario) {
  let maxGap = 0;
  let offset = 0;
  for (const row of scenario.rows) {
    for (let index = 1; index < row.strokes.length; index += 1) {
      const gap = strokes[offset + index].startTime - strokes[offset + index - 1].startTime;
      maxGap = Math.max(maxGap, gap);
    }
    offset += row.strokes.length;
  }
  return maxGap;
}

function assertScenarioStartsUnderProblem(scenario, snapshot) {
  const problem = snapshot.activeProblem;
  expect(problem).not.toBeNull();
  const firstRowBox = bboxForStrokes(scenario.rows[0].strokes);
  expect(Math.abs(firstRowBox.xMin - problem.boardPosition.x)).toBeLessThanOrEqual(8);
  expect(firstRowBox.yMin).toBeGreaterThan(problem.boardPosition.y + 100);
}

function assertEqualsBarsAligned(strokes, scenario, tolerance = 4) {
  const scenarioBars = scenarioEqualBarStrokes(scenario);
  const flatScenarioStrokes = scenario.strokes;
  const renderedBars = scenarioBars.map((bar) => {
    const strokeIndex = flatScenarioStrokes.indexOf(bar);
    expect(strokeIndex).toBeGreaterThanOrEqual(0);
    return {
      scenario: bar,
      rendered: strokes[strokeIndex]
    };
  });

  const lineOneBars = renderedBars.filter((entry) => entry.scenario.lineIndex === 0);
  const finalLineBars = renderedBars.filter((entry) => entry.scenario.lineIndex === 2);
  expect(lineOneBars).toHaveLength(2);
  expect(finalLineBars).toHaveLength(2);

  for (const reference of lineOneBars) {
    for (const candidate of finalLineBars) {
      expect(Math.abs(centerX(reference.rendered.canvasBbox) - centerX(candidate.rendered.canvasBbox))).toBeLessThanOrEqual(tolerance);
      expect(Math.abs(width(reference.rendered.canvasBbox) - width(candidate.rendered.canvasBbox))).toBeLessThanOrEqual(tolerance + 5);
    }
  }
}

async function pauseRealtimeRecognition(page) {
  await page.evaluate(() => {
    window.__whiteboardE2E?.setRealtimeRecognitionPaused?.(true);
  });
}

function averagePointDeviation(cleanScenario, messyScenario) {
  let total = 0;
  let count = 0;
  for (const [index, cleanStroke] of cleanScenario.strokes.entries()) {
    const messyStroke = messyScenario.strokes[index];
    if (!messyStroke || cleanStroke.kind === 'equals-bar') continue;
    const length = Math.min(cleanStroke.points.length, messyStroke.points.length);
    for (let pointIndex = 0; pointIndex < length; pointIndex += 1) {
      const clean = cleanStroke.points[pointIndex];
      const messy = messyStroke.points[pointIndex];
      total += Math.hypot(clean.x - messy.x, clean.y - messy.y);
      count += 1;
    }
  }
  return count ? total / count : 0;
}

function averageEndpointDeviation(cleanScenario, messyScenario) {
  let total = 0;
  let count = 0;
  for (const [index, cleanStroke] of cleanScenario.strokes.entries()) {
    const messyStroke = messyScenario.strokes[index];
    if (!messyStroke || cleanStroke.kind === 'equals-bar') continue;
    const cleanStart = cleanStroke.points[0];
    const cleanEnd = cleanStroke.points[cleanStroke.points.length - 1];
    const messyStart = messyStroke.points[0];
    const messyEnd = messyStroke.points[messyStroke.points.length - 1];
    total += Math.hypot(cleanStart.x - messyStart.x, cleanStart.y - messyStart.y);
    total += Math.hypot(cleanEnd.x - messyEnd.x, cleanEnd.y - messyEnd.y);
    count += 2;
  }
  return count ? total / count : 0;
}

function assertMessyOperationPairsCrowd(cleanScenario, messyScenario) {
  const cleanLeftGap = operationPairGap(cleanScenario, 'left');
  const cleanRightGap = operationPairGap(cleanScenario, 'right');
  const messyLeftGap = operationPairGap(messyScenario, 'left');
  const messyRightGap = operationPairGap(messyScenario, 'right');

  expect(cleanLeftGap).toBeGreaterThan(20);
  expect(cleanRightGap).toBeGreaterThan(20);
  expect(messyLeftGap).toBeLessThan(cleanLeftGap - 20);
  expect(messyRightGap).toBeLessThan(cleanRightGap - 20);
  expect(Math.min(messyLeftGap, messyRightGap)).toBeLessThanOrEqual(6);
}

function operationPairGap(scenario, side) {
  const slashStroke = scenario.strokes.find((item) => item.id.endsWith(`${side}-divide`));
  const twoStroke = scenario.strokes.find((item) => item.id.endsWith(`${side}-two`));
  expect(slashStroke).toBeTruthy();
  expect(twoStroke).toBeTruthy();
  const slashBox = bboxForPoints(slashStroke.points);
  const twoBox = bboxForPoints(twoStroke.points);
  return twoBox.xMin - slashBox.xMax;
}

function strokeBoxBySuffix(strokes, suffix) {
  const stroke = strokes.find((item) => item.id.endsWith(suffix));
  expect(stroke).toBeTruthy();
  return bboxForPoints(stroke.points);
}

function bboxForStrokes(strokes) {
  return strokes.reduce((box, stroke) => unionBox(box, bboxForPoints(stroke.points)), null);
}

function bboxForPoints(points) {
  return points.reduce((box, point) => unionBox(box, {
    xMin: point.x,
    yMin: point.y,
    xMax: point.x,
    yMax: point.y
  }), null);
}

function unionBox(a, b) {
  if (!a) return { ...b };
  return {
    xMin: Math.min(a.xMin, b.xMin),
    yMin: Math.min(a.yMin, b.yMin),
    xMax: Math.max(a.xMax, b.xMax),
    yMax: Math.max(a.yMax, b.yMax)
  };
}

function centerX(box) {
  return (box.xMin + box.xMax) / 2;
}

function width(box) {
  return box.xMax - box.xMin;
}
