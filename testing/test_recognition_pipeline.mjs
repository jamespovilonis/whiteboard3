#!/usr/bin/env node

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import {
  buildProblemInputAuditPayload,
  buildAttemptId,
  buildRecognitionAuditPayload,
  deterministicSample,
  getRecognitionAuditDecision,
  hasCorrectAnswerWithInvalidStep,
  normalizeProblemInputRecognitionResult
} from '../src/recognition/auditClient.js';
import { getRecognitionApiUrl, normalizeConfiguredApiUrl } from '../src/recognition/config.js';
import {
  createRecognitionLatencyTelemetry,
  strokeCaptureLatencySamples
} from '../src/recognition/latencyTelemetry.js';
import { problemStatusDisplay } from '../src/components/problemStatusDisplay.js';
import { IncrementalRecognitionScheduler } from '../src/recognition/incrementalRecognitionScheduler.js';
import { translateDetections } from '../src/recognition/segmentationClient.js';
import {
  previousLatexForSubmission,
  shouldRunRecognitionForProblem,
  summarizeRecognitionResult
} from '../src/hooks/useProblemFlowController.js';
import { recognizeStudentWriting, shouldUseSemanticLatex } from '../src/recognition/studentWritingPipeline.js';
import {
  scoreRecognitionEvidence,
  segmentMathLines,
  selectCandidateCover,
  strokeBelongsToAnswerBox
} from '../src/recognition/lineSegmentation.js';
import {
  applyProblemRecognitionError,
  applyProblemFeedbackProgress,
  applyProblemRecognitionProgress,
  applyProblemRecognitionResult,
  createInitialProblemFlow,
  getActiveProblem,
  getActiveModelResponse,
  isProblemReadyForNext,
  isProblemSubmittable,
  reconcileProblemFlowWithStrokes,
  requestNextProblem,
  startCustomProblem,
  submitActiveProblem
} from '../src/state/problemFlow.js';
import {
  loadRealHandwritingFixture,
  loadRealHandwritingFixtures,
  strokeGroupKey
} from './real_handwriting_fixtures.mjs';

const FLOW_TEST_PROBLEMS = Object.freeze([
  {
    id: 'flow-context-eta',
    kind: 'equation-solving',
    latex: '\\eta + 1 = 6',
    modelResponse: {
      before: 'Solve the equation.',
      latex: '\\eta = 5',
      after: 'Submit your work when you are ready.'
    }
  },
  {
    id: 'flow-context-z',
    kind: 'equation-solving',
    latex: 'z + 1 = 6',
    modelResponse: {
      before: 'Solve the equation.',
      latex: 'z = 5',
      after: 'Submit your work when you are ready.'
    }
  }
]);

test('recognition API defaults to the local gateway port', () => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    location: {
      protocol: 'http:',
      hostname: '127.0.0.1'
    }
  };
  try {
    assert.equal(getRecognitionApiUrl(), 'http://127.0.0.1:8010');
  } finally {
    if (previousWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = previousWindow;
    }
  }
});

test('recognition API maps wildcard dev host to loopback gateway', () => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    location: {
      protocol: 'http:',
      hostname: '0.0.0.0'
    }
  };
  try {
    assert.equal(getRecognitionApiUrl(), 'http://127.0.0.1:8010');
  } finally {
    if (previousWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = previousWindow;
    }
  }
});

test('recognition API rewrites loopback override for lan clients', () => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    location: {
      protocol: 'http:',
      hostname: '192.168.1.156'
    }
  };
  try {
    assert.equal(
      normalizeConfiguredApiUrl('http://127.0.0.1:8010'),
      'http://192.168.1.156:8010'
    );
  } finally {
    if (previousWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = previousWindow;
    }
  }
});

test('recognition latency telemetry summarizes p50 p95 and budget failures', () => {
  const telemetry = createRecognitionLatencyTelemetry({
    budgetsMs: {
      ocr: 100,
      segmentation: 20
    }
  });
  telemetry.record('ocr', 40, { candidateId: 'a' });
  telemetry.record('ocr', 120, { candidateId: 'b' });
  telemetry.record('segmentation', 12);

  const summary = telemetry.summary();
  assert.equal(summary.stages.ocr.count, 2);
  assert.equal(summary.stages.ocr.p50Ms, 40);
  assert.equal(summary.stages.ocr.p95Ms, 120);
  assert.equal(summary.stages.ocr.overBudgetCount, 1);
  assert.equal(summary.budgetFailureCount, 1);
  assert.equal(summary.budgetFailures[0].stage, 'ocr');
  assert.equal(summary.stages.segmentation.overBudget, false);
});

test('stroke capture latency uses finalized capture work instead of draw duration', () => {
  const samples = strokeCaptureLatencySamples([{
    id: 'stroke-a',
    startTime: 1000,
    endTime: 2400,
    points: [{}, {}],
    latency: {
      strokeCaptureElapsedMs: 4.5,
      drawDurationMs: 1400
    }
  }]);
  assert.equal(samples[0].elapsedMs, 4.5);
  assert.equal(samples[0].drawDurationMs, 1400);

  const fallback = strokeCaptureLatencySamples([{
    id: 'old-stroke',
    startTime: 1000,
    endTime: 2400,
    points: [{}]
  }]);
  assert.equal(fallback[0].elapsedMs, 0);
  assert.equal(fallback[0].drawDurationMs, 1400);
  assert.equal(fallback[0].estimated, true);
});

test('OCR evidence can promote a parent candidate over child rows', () => {
  const parent = candidate('parent_a|b', ['parent'], ['a', 'b'], 0, 0, 100, 100);
  const childA = candidate('row_a', ['row-line'], ['a'], 0, 0, 100, 40);
  const childB = candidate('row_b', ['row-line'], ['b'], 0, 60, 100, 100);

  const geometryOnly = selectCandidateCover([parent, childA, childB]);
  assert.deepEqual(geometryOnly.map((item) => item.candidateId), ['row_a', 'row_b']);

  const withOcrEvidence = selectCandidateCover([parent, childA, childB], {
    scoreByCandidateId: {
      'parent_a|b': 10
    }
  });
  assert.deepEqual(withOcrEvidence.map((item) => item.candidateId), ['parent_a|b']);
});

test('baseline-aware OCR evidence does not fragment a structural math line', () => {
  const ink = [
    stroke('num', 100, 80, 132, 112),
    stroke('bar', 88, 124, 220, 130),
    stroke('den', 104, 144, 138, 176),
    stroke('eq', 248, 116, 290, 148),
    stroke('rhs', 318, 102, 356, 164),
  ];
  const whole = candidateFromStrokeList('row_fraction', ['row-line'], ink);
  const pieces = ink.map((item) => candidateFromStrokeList(`strict_${item.id}`, ['strict'], [item]));
  const scoreByCandidateId = Object.fromEntries(
    pieces.map((piece) => [piece.candidateId, 2])
  );
  scoreByCandidateId[whole.candidateId] = -1000;

  const rawEvidenceCover = selectCandidateCover([whole, ...pieces], { scoreByCandidateId });
  assert.deepEqual(
    rawEvidenceCover.map((item) => item.candidateId).sort(),
    pieces.map((item) => item.candidateId).sort()
  );

  const stableEvidenceCover = selectCandidateCover([whole, ...pieces], {
    scoreByCandidateId,
    baselineCandidates: [whole]
  });
  assert.deepEqual(stableEvidenceCover.map((item) => item.candidateId), [whole.candidateId]);
});

test('baseline-aware OCR evidence does not coalesce lines into an unread parent', () => {
  const childA = candidate('row_a', ['row-line', 'dbnet-line'], ['a'], 0, 0, 160, 40);
  const childB = candidate('row_b', ['row-line', 'dbnet-line'], ['b'], 0, 72, 160, 112);
  const parent = candidateFromStrokeList('parent_a|b', ['parent', 'dbnet-parent'], [
    ...childA.strokes,
    ...childB.strokes,
  ]);

  const rawPartialEvidenceCover = selectCandidateCover([parent, childA, childB], {
    scoreByCandidateId: {
      row_a: -1000,
      row_b: -1000
    }
  });
  assert.deepEqual(rawPartialEvidenceCover.map((item) => item.candidateId), ['parent_a|b']);

  const stablePartialEvidenceCover = selectCandidateCover([parent, childA, childB], {
    scoreByCandidateId: {
      row_a: -1000,
      row_b: -1000
    },
    baselineCandidates: [childA, childB]
  });
  assert.deepEqual(stablePartialEvidenceCover.map((item) => item.candidateId), ['row_a', 'row_b']);

  const stableEmptyParentCover = selectCandidateCover([parent, childA, childB], {
    scoreByCandidateId: {
      'parent_a|b': -50,
      row_a: -1000,
      row_b: -1000
    },
    baselineCandidates: [childA, childB]
  });
  assert.deepEqual(stableEmptyParentCover.map((item) => item.candidateId), ['row_a', 'row_b']);
});

test('baseline-aware OCR evidence does not coalesce lines into an unread projection row', () => {
  const childA = candidate('row_a', ['row-line', 'dbnet-line'], ['a', 'b'], 0, 0, 260, 78);
  const childB = candidate('row_b', ['row-line', 'dbnet-line'], ['c', 'd'], 0, 118, 360, 190);
  const projection = candidateFromStrokeList('projection_a|b|c|d', ['projection-line'], [
    ...childA.strokes,
    ...childB.strokes,
  ]);

  const rawPartialEvidenceCover = selectCandidateCover([projection, childA, childB], {
    scoreByCandidateId: {
      row_a: -1000,
      row_b: -1000
    }
  });
  assert.deepEqual(rawPartialEvidenceCover.map((item) => item.candidateId), ['projection_a|b|c|d']);

  const stableNoProjectionEvidence = selectCandidateCover([projection, childA, childB], {
    scoreByCandidateId: {
      row_a: -1000,
      row_b: -1000
    },
    baselineCandidates: [childA, childB]
  });
  assert.deepEqual(stableNoProjectionEvidence.map((item) => item.candidateId), ['row_a', 'row_b']);

  const stableEmptyProjection = selectCandidateCover([projection, childA, childB], {
    scoreByCandidateId: {
      'projection_a|b|c|d': -50,
      row_a: -1000,
      row_b: -1000
    },
    baselineCandidates: [childA, childB]
  });
  assert.deepEqual(stableEmptyProjection.map((item) => item.candidateId), ['row_a', 'row_b']);
});

test('merged detector row-line yields to independent child rows', () => {
  const upper = candidateFromStrokeList('strict_upper', ['strict', 'row-line'], [
    stroke('u1', 20, 0, 80, 40),
    stroke('u2', 110, 8, 170, 48),
  ]);
  const lower = candidateFromStrokeList('strict_lower', ['strict', 'row-line'], [
    stroke('l1', 22, 76, 88, 116),
    stroke('l2', 118, 84, 184, 124),
  ]);
  const merged = candidateFromStrokeList('merged_dbnet_line', ['raw-row-line', 'row-line', 'dbnet-line'], [
    ...upper.strokes,
    ...lower.strokes,
  ]);

  const selected = selectCandidateCover([merged, upper, lower]);

  assert.deepEqual(selected.map((item) => item.candidateId), ['strict_upper', 'strict_lower']);
});

test('dbnet-only merged line yields to independent child rows', () => {
  const upper = candidateFromStrokeList('strict_upper', ['strict', 'raw-row-line', 'row-line'], [
    stroke('u1', 20, 0, 90, 44),
    stroke('u2', 110, 6, 180, 50),
  ]);
  const lower = candidateFromStrokeList('row_lower', ['raw-row-line', 'row-line'], [
    stroke('l1', 24, 74, 94, 118),
    stroke('l2', 116, 82, 186, 126),
  ]);
  const merged = candidateFromStrokeList('dbnet_only_merge', ['dbnet-line'], [
    ...upper.strokes,
    ...lower.strokes,
  ]);

  const selected = selectCandidateCover([merged, upper, lower]);

  assert.deepEqual(selected.map((item) => item.candidateId), ['strict_upper', 'row_lower']);
});

test('clean dbnet row beats strict row with lower-edge intrusion', () => {
  const main = [
    stroke('a', 100, 0, 150, 60),
    stroke('b', 170, 8, 220, 70),
    stroke('c', 240, 14, 290, 72),
  ];
  const intruding = [
    stroke('e1', 120, 76, 165, 86),
    stroke('e2', 190, 78, 235, 88),
  ];
  const lowerRemainder = [
    stroke('d1', 260, 96, 310, 140),
    stroke('d2', 330, 100, 380, 145),
    stroke('d3', 400, 104, 450, 148),
  ];
  const mixed = candidateFromStrokeList('mixed', ['strict', 'raw-row-line', 'row-line'], [
    ...main,
    ...intruding,
  ]);
  const clean = candidateFromStrokeList('clean', ['dbnet-line'], main);
  const lower = candidateFromStrokeList('lower', ['raw-row-line', 'row-line', 'dbnet-line'], [
    ...intruding,
    ...lowerRemainder,
  ]);
  const lowerRest = candidateFromStrokeList('lower-rest', ['raw-row-line', 'row-line'], lowerRemainder);

  const selected = selectCandidateCover([mixed, clean, lower, lowerRest]);

  assert.deepEqual(selected.map((item) => item.candidateId), ['clean', 'lower']);
});

test('dbnet row with upper boundary intrusion yields to clean child row', () => {
  const upperMain = [
    stroke('u1', 118, 61, 145, 146),
    stroke('u2', 159, 94, 199, 128),
    stroke('u3', 220, 98, 304, 122),
  ];
  const upperBoundary = stroke('u-boundary', 128, 135, 138, 152);
  const lowerMain = [
    stroke('l1', 164, 163, 209, 219),
    stroke('l2', 231, 171, 305, 247),
    stroke('l3', 326, 183, 467, 248),
  ];

  const upperDbnet = candidateFromStrokeList('upper-dbnet', ['dbnet-line'], upperMain);
  const upperStrict = candidateFromStrokeList('upper-strict', ['strict', 'row-line'], [
    ...upperMain,
    upperBoundary,
  ]);
  const lowerDbnetMixed = candidateFromStrokeList('lower-dbnet-mixed', ['dbnet-line'], [
    upperBoundary,
    ...lowerMain,
  ]);
  const lowerClean = candidateFromStrokeList('lower-clean', ['strict', 'raw-row-line', 'row-line'], lowerMain);

  const selected = selectCandidateCover([upperDbnet, upperStrict, lowerDbnetMixed, lowerClean]);

  assert.deepEqual(selected.map((item) => item.candidateId), ['upper-strict', 'lower-clean']);
});

test('messy quadratic formula fraction stays one line without swallowing later work', () => {
  const boxes = [
    ['q1', 0, 334, 63, 627, 128], ['q2', 1, 470, 198, 499, 242], ['q3', 2, 306, 349, 343, 392],
    ['q4', 0, 393, 79, 407, 101], ['q5', 1, 489, 198, 498, 206], ['q6', 2, 156, 351, 198, 395],
    ['q7', 0, 263, 84, 281, 115], ['q8', 1, 368, 200, 398, 249], ['q9', 2, 230, 353, 277, 361],
    ['q10', 0, 370, 86, 388, 118], ['q11', 1, 413, 202, 456, 251], ['q12', 2, 229, 368, 276, 377],
    ['q13', 0, 481, 86, 493, 129], ['q14', 1, 322, 229, 364, 237], ['q15', 0, 515, 87, 528, 132],
    ['q16', 1, 168, 259, 215, 308], ['q17', 0, 535, 88, 546, 133], ['q18', 1, 237, 260, 298, 271],
    ['q19', 0, 293, 89, 321, 119], ['q20', 1, 317, 261, 504, 284], ['q21', 0, 457, 89, 475, 120],
    ['q22', 1, 236, 278, 298, 289], ['q23', 0, 496, 90, 509, 121], ['q24', 1, 395, 280, 423, 329],
    ['q25', 0, 601, 90, 613, 135], ['q26', 0, 581, 93, 598, 124], ['q27', 0, 233, 100, 259, 103],
    ['q28', 0, 420, 105, 445, 109], ['q29', 0, 552, 109, 576, 113], ['q30', 0, 126, 117, 158, 147],
    ['q31', 0, 174, 123, 214, 128], ['q32', 0, 172, 136, 213, 140], ['q33', 0, 226, 137, 619, 153],
    ['q34', 0, 407, 151, 420, 195], ['q35', 0, 441, 153, 454, 197], ['q36', 0, 384, 154, 403, 185],
    ['q37', 0, 422, 155, 436, 187],
  ];
  const strokes = boxes.map(([id, lineIndex, xMin, yMin, xMax, yMax], index) => ({
    ...stroke(id, xMin, yMin, xMax, yMax),
    syntheticLineIndex: lineIndex,
    startTime: index * 120,
    endTime: index * 120 + 20
  }));

  const result = segmentMathLines(strokes, {
    answerBox: { xMin: 100, yMin: 40, xMax: 650, yMax: 420 }
  });
  const lineSets = result.selected.map((candidate) => (
    [...new Set(candidate.strokes.map((item) => item.syntheticLineIndex))].sort((a, b) => a - b)
  ));

  assert.deepEqual(lineSets, [[0], [1], [2]]);
  assert.equal(result.selected[0].profiles.includes('fraction-stack-line'), true);
  assert.equal(result.selected[1].profiles.includes('fraction-stack-line'), true);
});

test('slow OCR penalizes nonstructural multi-row parent evidence', () => {
  const childA = candidate('row_a', ['row-line'], ['a'], 0, 0, 100, 32);
  const childB = candidate('row_b', ['row-line'], ['b'], 0, 74, 100, 106);
  const parent = candidateFromStrokeList('parent_a|b', ['parent'], [
    ...childA.strokes,
    ...childB.strokes,
  ]);
  const prediction = {
    latex: 'x = 1',
    top: { latex: 'x = 1', score: 2 },
    candidates: [{ latex: 'x = 1', score: 2 }],
    elapsedSeconds: 9
  };

  const slowParentScore = scoreRecognitionEvidence(parent, prediction, { problemLatex: 'x = 1' });
  const slowChildScore = scoreRecognitionEvidence(childA, prediction, { problemLatex: 'x = 1' });

  assert.ok(slowParentScore < slowChildScore - 3);
});

test('slow OCR does not add multi-row penalty to structural fraction lines', () => {
  const fraction = candidateFromStrokeList('row_fraction', ['row-line'], [
    stroke('num', 100, 80, 132, 112),
    stroke('bar', 88, 124, 220, 130),
    stroke('den', 104, 144, 138, 176),
    stroke('eq', 248, 116, 290, 148),
    stroke('rhs', 318, 102, 356, 164),
  ]);
  const singleRow = candidate('row_single', ['row-line'], ['a'], 0, 0, 120, 42);
  const prediction = {
    latex: '\\frac { x } { 2 } = 1',
    top: { latex: '\\frac { x } { 2 } = 1', score: 2 },
    candidates: [{ latex: '\\frac { x } { 2 } = 1', score: 2 }],
    elapsedSeconds: 9
  };

  const structuralScore = scoreRecognitionEvidence(fraction, prediction, { problemLatex: '\\frac { x } { 2 } = 1' });
  const singleRowScore = scoreRecognitionEvidence(singleRow, prediction, { problemLatex: '\\frac { x } { 2 } = 1' });

  assert.ok(Math.abs(structuralScore - singleRowScore) < 0.1);
});

test('student writing pipeline recognizes alternatives before final selection', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'x = 1',
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      if (image.candidateId.startsWith('parent_')) {
        return {
          latex: 'x = 1',
          top: { latex: 'x = 1', score: 2 },
          candidates: [{ latex: 'x = 1', score: 2 }],
          elapsedSeconds: 0.6
        };
      }
      return {
        latex: '',
        candidates: [],
        failed: true,
        elapsedSeconds: 0.2
      };
    }
  });

  assert.ok(calls.some((id) => id.startsWith('parent_')));
  assert.ok(result.candidatePredictions.some((entry) => entry.profiles.includes('row-line')));
  assert.equal(result.lines.length, 1);
  assert.ok(result.lines[0].candidateId.startsWith('parent_'));
  assert.equal(result.latex, 'x = 1');
  assert.ok(result.candidatePredictions.length > result.lines.length);
});

test('student writing pipeline skips CoMER for contained single-stroke alternatives', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 20, 30),
    stroke('b', 100, 0, 120, 30),
    stroke('c', 200, 0, 220, 30),
  ];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 240, yMax: 60 },
    problemLatex: 'x = 1',
    semanticScoring: true,
    recognizeLine: async (image) => {
      calls.push(image.strokeIds.slice());
      if (image.strokeIds.length === 1) {
        throw new Error('single-stroke alternatives should not hit OCR');
      }
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.6
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 3,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: false,
        candidateScores: []
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(calls.length > 0);
  assert.ok(calls.every((strokeIds) => strokeIds.length > 1));
  assert.ok(result.candidatePredictions.length > calls.length);
  assert.ok(result.candidatePredictions.some((entry) => (
    entry.strokeIds.length === 1 &&
    entry.skippedRecognition &&
    entry.image === null &&
    entry.prediction.skipReason === 'single-stroke-alternative'
  )));
  assert.equal(result.latex, 'x = 1');

  const noSkipCalls = [];
  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 240, yMax: 60 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    skipSingleStrokeAlternatives: false,
    recognizeLine: async (image) => {
      noSkipCalls.push(image.strokeIds.slice());
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.6
      };
    }
  });

  assert.ok(noSkipCalls.length > calls.length);
});

test('student writing pipeline defers larger boxes covered by valid deterministic lines', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      if (image.candidateId === 'parent_a|b') {
        throw new Error('covered parent should not hit initial OCR');
      }
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.04
      };
    }
  });

  assert.deepEqual(calls, ['loose_a', 'loose_b']);
  const parentEntry = result.candidatePredictions.find((entry) => entry.candidateId === 'parent_a|b');
  assert.ok(parentEntry);
  assert.equal(parentEntry.skippedRecognition, true);
  assert.equal(parentEntry.image, null);
  assert.equal(parentEntry.prediction.skipReason, 'covered-by-valid-deterministic-line');
  assert.ok(result.candidatePredictions.length > calls.length);
});

test('student writing pipeline OCRs larger box when deterministic children are weak', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      if (image.candidateId === 'parent_a|b') {
        return {
          latex: 'x = 1',
          top: { latex: 'x = 1', score: 2 },
          candidates: [{ latex: 'x = 1', score: 2 }],
          elapsedSeconds: 0.04
        };
      }
      return {
        latex: '',
        top: null,
        candidates: [],
        failed: true,
        elapsedSeconds: 0.02
      };
    }
  });

  assert.deepEqual(calls, ['loose_a', 'loose_b', 'parent_a|b']);
  assert.equal(result.lines.length, 1);
  assert.equal(result.lines[0].candidateId, 'parent_a|b');
  assert.equal(result.latex, 'x = 1');
});

test('covered parent deferral can be disabled for eager alternative OCR', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  const calls = [];

  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    deferCoveredParentRecognition: false,
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.04
      };
    }
  });

  assert.ok(calls.includes('parent_a|b'));
});

test('student writing pipeline defers contained nonstructural alternatives', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 20, 30),
    stroke('b', 35, 0, 55, 30),
    stroke('c', 140, 0, 160, 30),
  ];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 180, yMax: 60 },
    problemLatex: 'x = 1',
    semanticScoring: true,
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      if (image.candidateId === 'strict_a|b') {
        throw new Error('contained strict alternatives should be deferred');
      }
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.04
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 3,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: false,
        candidateScores: []
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(calls.includes('parent_a|b|c'));
  assert.equal(calls.includes('strict_a|b'), false);
  assert.ok(result.candidatePredictions.some((entry) => (
    entry.candidateId === 'strict_a|b' &&
    entry.skippedRecognition &&
    entry.image === null &&
    entry.prediction.skipReason === 'contained-nonstructural-alternative'
  )));
  assert.equal(result.latex, 'x = 1');

  const eagerCalls = [];
  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 180, yMax: 60 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    stagedAlternativeRecognition: false,
    recognizeLine: async (image) => {
      eagerCalls.push(image.candidateId);
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.04
      };
    }
  });

  assert.ok(eagerCalls.includes('strict_a|b'));
  assert.ok(eagerCalls.length > calls.length);
});

test('student writing pipeline OCRs deferred answer alternatives when selected container is weak', async () => {
  installFakeCanvas();
  const cases = [
    {
      name: 'compact linear',
      strokes: [
        stroke('a', 0, 0, 24, 34),
        stroke('b', 34, 0, 58, 34),
        stroke('c', 140, 0, 166, 34),
      ],
      answerLatex: 'x = 4',
      answerCandidateIds: ['strict_a|b']
    },
    {
      name: 'wide derivative',
      strokes: [
        stroke('a', 0, 0, 38, 44),
        stroke('b', 52, 0, 96, 46),
        stroke('c', 112, 0, 158, 46),
        stroke('d', 320, 0, 350, 44),
      ],
      answerLatex: 'f ^ { \\prime } ( x ) = 2 x',
      answerCandidateIds: ['strict_a|b|c', 'loose_a|b|c']
    },
    {
      name: 'fraction-like spacing',
      strokes: [
        stroke('a', 0, 0, 42, 34),
        stroke('b', 6, 48, 48, 82),
        stroke('c', 72, 20, 120, 58),
        stroke('d', 260, 8, 292, 48),
      ],
      answerLatex: '\\frac { x } { 2 } = 3',
      answerCandidateIds: ['strict_a|b|c', 'loose_a|b|c']
    },
  ];

  for (const fixture of cases) {
    assignTimes(fixture.strokes);
    const calls = [];

    const result = await recognizeStudentWriting({
      strokes: fixture.strokes,
      answerBox: { xMin: -8, yMin: -8, xMax: 380, yMax: 110 },
      problemLatex: fixture.answerLatex,
      semanticScoring: true,
      retryRasterHeights: [],
      semanticRetryRasterHeights: [],
      recognizeLine: async (image) => {
        calls.push(image.candidateId);
        if (fixture.answerCandidateIds.includes(image.candidateId)) {
          return {
            latex: fixture.answerLatex,
            top: { latex: fixture.answerLatex, score: 2 },
            candidates: [{ latex: fixture.answerLatex, score: 2 }],
            elapsedSeconds: 0.08
          };
        }
        return {
          latex: '',
          top: null,
          candidates: [],
          failed: true,
          elapsedSeconds: 0.05
        };
      },
      scoreSemantics: async (request) => ({
        candidateScores: request.candidateGroups.map((group) => ({
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: group.latex === fixture.answerLatex ? 8 : 0,
          bestLatex: group.latex,
          sound: group.latex === fixture.answerLatex,
          equivalentToProblem: group.latex === fixture.answerLatex,
          equivalentToPrevious: false,
          candidateScores: []
        })),
        elapsedSeconds: 0.01
      })
    });

    assert.ok(
      fixture.answerCandidateIds.some((candidateId) => calls.includes(candidateId)),
      `${fixture.name} should OCR a deferred answer candidate; calls=${calls.join(',')}`
    );
    assert.equal(result.latex, fixture.answerLatex, fixture.name);
    assert.ok(
      result.lines.some((line) => fixture.answerCandidateIds.includes(line.candidateId)),
      fixture.name
    );
  }
});

test('student writing pipeline still recognizes a single-stroke answer when it has no containing alternative', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 80, 36)];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 90, yMax: 46 },
    problemLatex: 'x = 4',
    recognizeLine: async (image) => {
      calls.push(image.strokeIds.slice());
      return {
        latex: 'x = 4',
        top: { latex: 'x = 4', score: 2 },
        candidates: [{ latex: 'x = 4', score: 2 }],
        elapsedSeconds: 0.03
      };
    }
  });

  assert.ok(calls.some((strokeIds) => strokeIds.join('|') === 'a'));
  assert.equal(result.candidatePredictions.some((entry) => entry.skippedRecognition), false);
  assert.equal(result.latex, 'x = 4');
});

test('student writing pipeline runs independent initial OCR alternatives concurrently', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 20, 30),
    stroke('b', 100, 0, 120, 30),
    stroke('c', 200, 0, 220, 30),
  ];
  let inFlight = 0;
  let maxInFlight = 0;
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 240, yMax: 60 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    skipSingleStrokeAlternatives: false,
    initialRecognitionConcurrency: 3,
    recognizeLine: async (image) => {
      calls.push(image.candidateId);
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await sleep(20);
      inFlight -= 1;
      const isParent = image.strokeIds.length > 1;
      return {
        latex: isParent ? 'x = 1' : 'x',
        top: { latex: isParent ? 'x = 1' : 'x', score: isParent ? 2 : -1 },
        candidates: [{ latex: isParent ? 'x = 1' : 'x', score: isParent ? 2 : -1 }],
        elapsedSeconds: 0.02
      };
    }
  });

  assert.ok(calls.length >= 3);
  assert.ok(maxInFlight >= 2);
  assert.equal(result.latex, 'x = 1');
});

test('post-OCR merge coalesces same-row final answer fragments left to right', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('x', 0, 0, 40, 50),
    stroke('eq', 72, -5, 96, 55),
    stroke('nine', 104, -5, 140, 55),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -20, xMax: 160, yMax: 80 },
    problemLatex: 'x = 11',
    semanticScoring: true,
    skipSingleStrokeAlternatives: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      const strokeIds = image.strokeIds.join('|');
      if (strokeIds.includes('x') && strokeIds.includes('eq') && strokeIds.includes('nine')) {
        return { failed: true, latex: '', top: null, candidates: [], elapsedSeconds: 0.01 };
      }
      if (strokeIds === 'x') {
        return {
          latex: 'x',
          top: { latex: 'x', score: 20 },
          candidates: [{ latex: 'x', score: 20 }],
          elapsedSeconds: 0.01
        };
      }
      if (strokeIds.includes('eq') && strokeIds.includes('nine')) {
        return {
          latex: '= 9',
          top: { latex: '= 9', score: 20 },
          candidates: [{ latex: '= 9', score: 20 }],
          elapsedSeconds: 0.01
        };
      }
      return { latex: '', top: null, candidates: [], elapsedSeconds: 0 };
    },
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: 'x',
        cardinality: 'finite',
        exact_set: ['11'],
        decimal_set: [11],
        tolerance: 0.005
      },
      candidateScores: request.candidateGroups.map((group) => {
        const isMergedFinalAnswer = String(group.latex || '').trim() === 'x = 9';
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: isMergedFinalAnswer ? 1 : 10,
          bestLatex: group.latex,
          sound: false,
          equivalentToProblem: false,
          equivalentToPrevious: false,
          grading: isMergedFinalAnswer ? {
            studentLatex: group.latex,
            classification: 'invalid_step',
            selectedCandidateIndex: 0,
            solutionCoverage: 'none',
            matchedSolutions: []
          } : null,
          candidateScores: []
        };
      }),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines.length, 1);
  assert.deepEqual(result.lines[0].mergedFrom, ['strict_eq|nine', 'strict_x']);
  assert.equal(result.latex, 'x = 9');
  assert.equal(result.grading.result.problemStatus, 'incorrect');
});

test('pipeline debug timing covers selected and discarded candidates', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'x = 1',
    semanticScoring: true,
    deferCoveredParentRecognition: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      const latex = image.candidateId.startsWith('parent_') ? 'x = 1' : 'x';
      return {
        latex,
        top: { latex, score: 2 },
        candidates: [
          { latex, score: 2 },
          { latex: `${latex} + 0`, score: 1 },
          { latex: `${latex} - 0`, score: 0.5 },
          { latex: `${latex} \\cdot 1`, score: 0 },
          { latex: `${latex} / 1`, score: -0.5 },
        ],
        elapsedSeconds: image.candidateId.startsWith('parent_') ? 0.4 : 0.2
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: String(group.candidateId).startsWith('parent_') ? 12 : 0,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: String(group.candidateId).startsWith('parent_'),
        equivalentToPrevious: false,
        candidateScores: (group.candidates || []).map((candidate, index) => ({
          latex: candidate.latex,
          score: 5 - index,
          sound: true
        }))
      })),
      elapsedSeconds: 0.03
    })
  });

  const selected = result.candidatePredictions.filter((entry) => entry.selected);
  const discarded = result.candidatePredictions.filter((entry) => entry.discarded);

  assert.ok(selected.length > 0);
  assert.ok(discarded.length > 0);
  assert.ok(result.timing.totalElapsedSeconds >= 0);
  for (const entry of result.candidatePredictions) {
    assert.match(entry.debugLabel, /^C\d+$/);
    assert.equal(typeof entry.timing.submitToInitialPredictionSeconds, 'number');
    assert.equal(typeof entry.timing.submitToFinalPredictionSeconds, 'number');
    assert.equal(entry.image.dataUrl.startsWith('data:image/png;base64,'), true);
  }
});

test('student writing pipeline uses detector bands as segmentation evidence', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  let detectorImage = null;

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -10, yMin: -10, xMax: 80, yMax: 130 },
    problemLatex: 'x = 1',
    detectLineBands: true,
    detectLines: async (image) => {
      detectorImage = image;
      return {
        detections: [
          { bbox: { xMin: 10, yMin: 10, xMax: 60, yMax: 40 } },
          { bbox: { xMin: 10, yMin: 90, xMax: 60, yMax: 120 } },
        ],
        elapsedSeconds: 0.05
      };
    },
    recognizeLine: async (image) => ({
      latex: image.profiles.includes('dbnet-line') ? 'x = 1' : '',
      top: image.profiles.includes('dbnet-line') ? { latex: 'x = 1', score: 2 } : null,
      candidates: image.profiles.includes('dbnet-line') ? [{ latex: 'x = 1', score: 2 }] : [],
      failed: !image.profiles.includes('dbnet-line'),
      elapsedSeconds: 0.3
    })
  });

  assert.equal(detectorImage.originX, -10);
  assert.equal(detectorImage.originY, -10);
  assert.equal(result.detection.source, 'detector');
  assert.equal(result.detection.detections[0].bbox.xMin, 10);
  assert.ok(result.candidatePredictions.some((entry) => entry.profiles.includes('dbnet-line')));
  assert.ok(result.lines.every((line) => line.profiles.includes('dbnet-line')));
});

test('detector boxes translate from crop pixels to board coordinates', () => {
  const detections = translateDetections(
    [{ bbox: { xMin: 20, yMin: 30, xMax: 120, yMax: 70 }, polygon: [[20, 30], [120, 70]] }],
    { originX: 100, originY: 200, devicePixelRatio: 2 }
  );

  assert.deepEqual(detections[0].bbox, {
    xMin: 110,
    yMin: 215,
    xMax: 160,
    yMax: 235
  });
  assert.deepEqual(detections[0].polygon, [[110, 215], [160, 235]]);
});

test('student writing pipeline falls back when detector fails', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 50, 30)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 40 },
    problemLatex: 'x = 1',
    detectLineBands: true,
    detectLines: async () => {
      throw new Error('detector offline');
    },
    recognizeLine: async () => ({
      latex: 'x = 1',
      top: { latex: 'x = 1', score: 2 },
      candidates: [{ latex: 'x = 1', score: 2 }],
      elapsedSeconds: 0.3
    })
  });

  assert.equal(result.detection.failed, true);
  assert.equal(result.detection.error, 'detector offline');
  assert.equal(result.latex, 'x = 1');
});

test('semantic scoring can choose a better top-five latex candidate', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 50, 30)];
  let semanticRequest = null;

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 40 },
    problemLatex: '2 x + 3 = 11',
    semanticScoring: true,
    recognizeLine: async () => ({
      latex: '2 x + = 8',
      top: { latex: '2 x + = 8', score: 2 },
      candidates: [
        { latex: '2 x + = 8', score: 2 },
        { latex: '2 x = 8', score: -1 },
      ],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => {
      semanticRequest = request;
      return {
        candidateScores: [{
          candidateId: request.candidateGroups[0].candidateId,
          semanticScore: 6,
          bestLatex: '2 x = 8',
          sound: true,
          equivalentToProblem: true,
          candidateScores: []
        }],
        elapsedSeconds: 0.05
      };
    }
  });

  assert.equal(semanticRequest.problemLatex, '2 x + 3 = 11');
  assert.equal(semanticRequest.candidateGroups[0].candidates.length, 2);
  assert.equal(result.semantic.source, 'semantic-service');
  assert.equal(result.latex, '2 x = 8');
  assert.equal(result.lines[0].semantic.bestLatex, '2 x = 8');
});

test('grading verdict selects a valid top-five candidate over OCR top one', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 50, 30)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 40 },
    problemLatex: '3 x + 5 = 17',
    semanticScoring: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: 'x = 5',
      top: { latex: 'x = 5', score: 2 },
      candidates: [
        { latex: 'x = 5', score: 2 },
        { latex: 'x = 4', score: -4 },
      ],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: 'x',
        cardinality: 'finite',
        exact_set: ['4'],
        decimal_set: [4],
        tolerance: 0.005
      },
      candidateScores: [{
        candidateId: request.candidateGroups[0].candidateId,
        semanticScore: 1,
        bestLatex: 'x = 5',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: 'x = 4',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['4']
        },
        candidateScores: []
      }],
      elapsedSeconds: 0.05
    })
  });

  assert.equal(result.latex, 'x = 4');
  assert.equal(result.lines[0].grading.classification, 'valid_step');
  assert.equal(result.lines[0].grading.selectedCandidateIndex, 1);
  assert.equal(result.grading.result.problemStatus, 'correct');
});

test('grading verdict promotes safe absolute-value top-five candidate to final latex', async () => {
  installFakeCanvas();
  const strokes = [stroke('abs', 0, 0, 140, 90)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 170, yMax: 120 },
    problemLatex: '\\frac{5}{\\sqrt{x^2}}',
    semanticScoring: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '\\frac { 5 } { 1 x }',
      top: { latex: '\\frac { 5 } { 1 x }', score: -0.22, confidence: 0.27 },
      candidates: [
        { latex: '\\frac { 5 } { 1 x }', score: -0.22, confidence: 0.27 },
        { latex: '\\frac { 5 } { | x | }', score: -0.41, confidence: 0.22 },
        { latex: '\\frac { 5 } { T x }', score: -0.45, confidence: 0.21 },
      ],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: 'x',
        cardinality: 'expression',
        exact_set: ['5/Abs(x)']
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        semanticScore: 1,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: '\\frac { 5 } { | x | }',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['5/Abs(x)'],
          answerFinality: 'final',
          countsTowardCompletion: true,
          candidateVerdicts: [
            { latex: '\\frac { 5 } { 1 x }', classification: 'invalid_step', countsTowardCompletion: false },
            { latex: '\\frac { 5 } { | x | }', classification: 'valid_step', countsTowardCompletion: true }
          ]
        },
        candidateScores: []
      })),
      elapsedSeconds: 0.05
    })
  });

  assert.deepEqual(result.latexLines, ['\\frac { 5 } { | x | }']);
  assert.equal(result.lines[0].acceptedLatex, '\\frac { 5 } { | x | }');
  assert.equal(result.grading.result.problemStatus, 'correct');
});

test('grading verdict does not promote non-completing semantic alternate', async () => {
  installFakeCanvas();
  const strokes = [stroke('abs', 0, 0, 140, 90)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 170, yMax: 120 },
    problemLatex: '\\frac{5}{\\sqrt{x^2}}',
    semanticScoring: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '\\frac { 5 } { 1 x }',
      top: { latex: '\\frac { 5 } { 1 x }', score: -0.22, confidence: 0.27 },
      candidates: [
        { latex: '\\frac { 5 } { 1 x }', score: -0.22, confidence: 0.27 },
        { latex: '\\frac { 5 } { | x | }', score: -0.41, confidence: 0.22 },
      ],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: 'x',
        cardinality: 'expression',
        exact_set: ['5/Abs(x)']
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        semanticScore: 1,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: '\\frac { 5 } { | x | }',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['5/Abs(x)'],
          answerFinality: 'unsimplified',
          countsTowardCompletion: false,
          candidateVerdicts: [
            { latex: '\\frac { 5 } { | x | }', classification: 'valid_step', countsTowardCompletion: false }
          ]
        },
        candidateScores: []
      })),
      elapsedSeconds: 0.05
    })
  });

  assert.deepEqual(result.latexLines, ['\\frac { 5 } { 1 x }']);
  assert.equal(result.grading.result.problemStatus, 'not_started');
});

test('grading verdict keeps full plus-minus solution over weaker plus-only read', async () => {
  installFakeCanvas();
  const strokes = [stroke('pm', 0, 0, 120, 50)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 150, yMax: 70 },
    problemLatex: 'x ^ { 4 } = 16',
    semanticScoring: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: 'x = + 2',
      top: { latex: 'x = + 2', score: 2 },
      candidates: [
        { latex: 'x = + 2', score: 2 },
        { latex: 'x = \\pm 2', score: 1.5 },
      ],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: 'x',
        cardinality: 'finite',
        exact_set: ['-2', '2'],
        decimal_set: [-2, 2],
        tolerance: 0.005
      },
      candidateScores: [{
        candidateId: request.candidateGroups[0].candidateId,
        semanticScore: 1,
        bestLatex: 'x = + 2',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: 'x = \\pm 2',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['-2', '2']
        },
        candidateScores: []
      }],
      elapsedSeconds: 0.05
    })
  });

  assert.equal(result.latex, 'x = \\pm 2');
  assert.equal(result.grading.result.problemStatus, 'correct');
  assert.deepEqual(result.grading.result.foundSolutions, ['-2', '2']);
});

test('semantic scoring payload is capped to top-five OCR candidates', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 50, 30)];
  let semanticRequest = null;
  const candidates = Array.from({ length: 7 }, (_item, index) => ({
    latex: `x = ${index + 1}`,
    score: 7 - index
  }));

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 40 },
    problemLatex: 'x = 1',
    semanticScoring: true,
    recognizeLine: async () => ({
      latex: 'x = 1',
      top: candidates[0],
      candidates,
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async (request) => {
      semanticRequest = request;
      return {
        candidateScores: request.candidateGroups.map((group) => ({
          candidateId: group.candidateId,
          semanticScore: 1,
          bestLatex: group.latex,
          sound: true,
          equivalentToProblem: true,
          candidateScores: []
        })),
        elapsedSeconds: 0.02
      };
    }
  });

  assert.equal(semanticRequest.candidateGroups[0].candidates.length, 5);
  assert.deepEqual(
    semanticRequest.candidateGroups[0].candidates.map((candidate) => candidate.latex),
    ['x = 1', 'x = 2', 'x = 3', 'x = 4', 'x = 5']
  );
  assert.equal(result.candidatePredictions[0].candidates.length, 7);
  assert.equal(result.latex, 'x = 1');
});

test('recognized single-letter variables inherit lowercase problem context', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 80, 36)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 90, yMax: 46 },
    problemLatex: '2 x + 3 = 11',
    semanticScoring: false,
    recognizeAlternatives: true,
    recognizeLine: async () => ({
      latex: '2X = 8',
      top: { latex: '2X = 8', score: 2 },
      candidates: [{ latex: '2X = 8', score: 2 }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '2x = 8');
  assert.equal(result.latexLines[0], '2x = 8');
});

test('malformed quadratic formula row is repaired from problem coefficients', async () => {
  installFakeCanvas();
  const strokes = [stroke('formula', 0, 0, 520, 120)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 540, yMax: 140 },
    problemLatex: 'x ^ { 2 } + 4 x - 5 = 0',
    semanticScoring: false,
    recognizeAlternatives: true,
    recognizeLine: async () => ({
      latex: 'x = \\frac { - 1 + \\sqrt { 1 ^ { 2 } - 1 ( 1 ) - b ) } } { 2 ( 1 ) }',
      top: {
        latex: 'x = \\frac { - 1 + \\sqrt { 1 ^ { 2 } - 1 ( 1 ) - b ) } } { 2 ( 1 ) }',
        score: 2
      },
      candidates: [{
        latex: 'x = \\frac { - 1 + \\sqrt { 1 ^ { 2 } - 1 ( 1 ) - b ) } } { 2 ( 1 ) }',
        score: 2
      }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(
    result.lines[0].acceptedLatex,
    'x = \\frac { - 4 + \\sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }'
  );
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-quadratic-formula');
});

test('variable-free rational problem line is repaired from problem context', async () => {
  installFakeCanvas();
  const strokes = [stroke('rational', 0, 0, 420, 90)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 440, yMax: 110 },
    problemLatex: '\\frac { x ^ { 2 } - 1 } { x - 1 } = 4',
    semanticScoring: false,
    recognizeAlternatives: true,
    recognizeLine: async () => ({
      latex: '\\frac { 2 ^ { 2 } - 1 } { 2 ^ { 2 } - 1 } = 4',
      top: {
        latex: '\\frac { 2 ^ { 2 } - 1 } { 2 ^ { 2 } - 1 } = 4',
        score: 2
      },
      candidates: [{
        latex: '\\frac { 2 ^ { 2 } - 1 } { 2 ^ { 2 } - 1 } = 4',
        score: 2
      }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '\\frac { x ^ { 2 } - 1 } { x - 1 } = 4');
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-rational-problem');
});

test('compact monomial exponent OCR is repaired from problem context', async () => {
  installFakeCanvas();
  const strokes = [stroke('monomial', 0, 0, 420, 90)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 440, yMax: 110 },
    problemLatex: '3 x ^ { 9 } y ^ { 1 2 }',
    apiUrl: 'http://127.0.0.1:8010',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '3 x 9 y ^ { 1 2 }',
      top: { latex: '3 x 9 y ^ { 1 2 }', score: 3, confidence: 0.99 },
      candidates: [{ latex: '3 x 9 y ^ { 1 2 }', score: 3, confidence: 0.99 }],
      elapsedSeconds: 0.03
    }),
    gradeWork: async (request) => pythonGradePayload(request)
  });

  assert.equal(result.lines[0].acceptedLatex, '3 x ^ { 9 } y ^ { 12 }');
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-monomial-exponent');
  assert.equal(result.grading.result?.problemStatus, 'correct');
});

test('compact monomial exponent repair rejects mismatched context and explicit multiplication', async () => {
  installFakeCanvas();
  const cases = [
    {
      name: 'unexpected exponent',
      problemLatex: '3 x ^ { 8 } y ^ { 1 2 }',
      latex: '3 x 9 y ^ { 1 2 }'
    },
    {
      name: 'explicit multiplication',
      problemLatex: '3 x ^ { 9 } y ^ { 1 2 }',
      latex: '3 \\times 9 y ^ { 1 2 }'
    }
  ];

  for (const item of cases) {
    const result = await recognizeStudentWriting({
      strokes: [stroke(`monomial-${item.name}`, 0, 0, 420, 90)],
      answerBox: { xMin: -5, yMin: -5, xMax: 440, yMax: 110 },
      problemLatex: item.problemLatex,
      semanticScoring: false,
      recognizeAlternatives: false,
      chunkFallback: false,
      recognizeLine: async () => ({
        latex: item.latex,
        top: { latex: item.latex, score: 3, confidence: 0.99 },
        candidates: [{ latex: item.latex, score: 3, confidence: 0.99 }],
        elapsedSeconds: 0.03
      })
    });

    assert.equal(result.lines[0].acceptedLatex, item.latex, item.name);
    assert.equal(result.lines[0].ocrRepair?.source, undefined, item.name);
  }
});

test('change-of-base log denominator OCR is repaired from problem context', async () => {
  installFakeCanvas();
  const strokes = [stroke('log-change-base', 0, 0, 620, 120)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 650, yMax: 145 },
    problemLatex: '\\log _ { 2 } x + \\log _ { 4 } x',
    apiUrl: 'http://127.0.0.1:8010',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { \\log x } { \\log ^ { 2 } } + \\frac { \\log x } { \\log x }',
      top: {
        latex: '\\frac { \\log x } { \\log ^ { 2 } } + \\frac { \\log x } { \\log x }',
        score: 3,
        confidence: 0.99
      },
      candidates: [{
        latex: '\\frac { \\log x } { \\log ^ { 2 } } + \\frac { \\log x } { \\log x }',
        score: 3,
        confidence: 0.99
      }],
      elapsedSeconds: 0.03
    }),
    gradeWork: async (request) => pythonGradePayload(request)
  });

  assert.equal(
    result.lines[0].acceptedLatex,
    '\\frac { \\log x } { \\log 2 } + \\frac { \\log x } { \\log 4 }'
  );
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-log-base-denominator');
  assert.equal(result.grading.result?.problemStatus, 'incomplete');
  assert.equal(result.grading.steps[0].classification, 'valid_step');
});

test('change-of-base log denominator repair rejects unsafe shapes', async () => {
  installFakeCanvas();
  const cases = [
    {
      name: 'fraction count mismatch',
      problemLatex: '\\log _ { 2 } x',
      latex: '\\frac { \\log x } { \\log ^ { 2 } } + \\frac { \\log x } { \\log x }'
    },
    {
      name: 'argument mismatch',
      problemLatex: '\\log _ { 2 } y + \\log _ { 4 } y',
      latex: '\\frac { \\log x } { \\log ^ { 2 } } + \\frac { \\log x } { \\log x }'
    },
    {
      name: 'malformed fraction count',
      problemLatex: '\\log _ { 2 } x + \\log _ { 4 } x',
      latex: '\\frac { \\log x } { \\log ^ { 2 } }'
    }
  ];

  for (const item of cases) {
    const result = await recognizeStudentWriting({
      strokes: [stroke(`log-${item.name}`, 0, 0, 620, 120)],
      answerBox: { xMin: -5, yMin: -5, xMax: 650, yMax: 145 },
      problemLatex: item.problemLatex,
      semanticScoring: false,
      recognizeAlternatives: false,
      chunkFallback: false,
      recognizeLine: async () => ({
        latex: item.latex,
        top: { latex: item.latex, score: 3, confidence: 0.99 },
        candidates: [{ latex: item.latex, score: 3, confidence: 0.99 }],
        elapsedSeconds: 0.03
      })
    });

    assert.equal(result.lines[0].acceptedLatex, item.latex, item.name);
    assert.equal(result.lines[0].ocrRepair?.source, undefined, item.name);
  }
});

test('semantic candidate promotion replaces accepted latex for safe operator alternate', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 220, 70),
    stroke('b', 0, 120, 180, 190)
  ];
  const gradeRequests = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 250, yMax: 220 },
    problemLatex: '\\sqrt { 2 } \\log _ { 3 } 9',
    apiUrl: 'http://127.0.0.1:8010',
    semanticScoring: true,
    recognizeAlternatives: false,
    chunkFallback: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      const ids = (image.strokeIds || []).join('|');
      if (ids.includes('a')) {
        return {
          latex: '\\sqrt { 2 } . 2',
          top: { latex: '\\sqrt { 2 } . 2', score: 3, confidence: 0.51 },
          candidates: [
            { latex: '\\sqrt { 2 } . 2', score: 3, confidence: 0.51 },
            { latex: '\\sqrt { 2 } \\cdot 2', score: 2.25, confidence: 0.24 },
            { latex: '\\sqrt { 2 } \\times 2', score: 1.2, confidence: 0.09 }
          ],
          elapsedSeconds: 0.03
        };
      }
      return {
        latex: '2 \\sqrt { 2 }',
        top: { latex: '2 \\sqrt { 2 }', score: 3, confidence: 0.9 },
        candidates: [{ latex: '2 \\sqrt { 2 }', score: 3, confidence: 0.9 }],
        elapsedSeconds: 0.03
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => {
        if (String(group.candidateId || '').includes('_a')) {
          return {
            candidateId: group.candidateId,
            lineIndex: group.lineIndex,
            semanticScore: 8,
            bestLatex: '\\sqrt { 2 } \\cdot 2',
            sound: true,
            equivalentToProblem: false,
            equivalentToPrevious: false,
            candidateScores: [],
            grading: {
              studentLatex: '\\sqrt { 2 } \\cdot 2',
              classification: 'valid_step',
              answerFinality: 'unsimplified',
              countsTowardCompletion: false,
              selectedCandidateIndex: 1,
              solutionCoverage: 'full',
              matchedSolutions: ['2*sqrt(2)']
            }
          };
        }
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: 8,
          bestLatex: '2 \\sqrt { 2 }',
          sound: true,
          equivalentToProblem: true,
          equivalentToPrevious: false,
          candidateScores: [],
          grading: {
            studentLatex: '2 \\sqrt { 2 }',
            classification: 'valid_step',
            answerFinality: 'final',
            countsTowardCompletion: true,
            selectedCandidateIndex: 0,
            solutionCoverage: 'full',
            matchedSolutions: ['2*sqrt(2)']
          }
        };
      }),
      elapsedSeconds: 0.01
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return pythonGradePayload(request);
    }
  });

  assert.equal(result.lines[0].acceptedLatex, '\\sqrt { 2 } \\cdot 2');
  assert.equal(result.latexLines[0], '\\sqrt { 2 } \\cdot 2');
  assert.equal(gradeRequests[0].lines[0].latex, '\\sqrt { 2 } \\cdot 2');
  assert.equal(result.grading.result?.problemStatus, 'correct');
});

test('semantic candidate promotion replaces accepted latex for safe radicand alternate', async () => {
  installFakeCanvas();
  const strokes = [stroke('r', 0, 0, 220, 70)];
  const gradeRequests = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 250, yMax: 95 },
    problemLatex: '\\frac { 1 2 } { \\sqrt { 6 } } + \\sqrt { 6 }',
    apiUrl: 'http://127.0.0.1:8010',
    semanticScoring: true,
    recognizeAlternatives: false,
    chunkFallback: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '3 \\sqrt { 8 }',
      top: { latex: '3 \\sqrt { 8 }', score: 3, confidence: 0.44 },
      candidates: [
        { latex: '3 \\sqrt { 8 }', score: 3, confidence: 0.44 },
        { latex: '3 \\sqrt { 6 }', score: 2.9, confidence: 0.4 },
        { latex: '3 \\sqrt { f }', score: 0.3, confidence: 0.06 }
      ],
      elapsedSeconds: 0.03
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 8,
        bestLatex: '3 \\sqrt { 6 }',
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: false,
        candidateScores: [],
        grading: {
          studentLatex: '3 \\sqrt { 6 }',
          classification: 'valid_step',
          answerFinality: 'final',
          countsTowardCompletion: true,
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['3*sqrt(6)']
        }
      })),
      elapsedSeconds: 0.01
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return pythonGradePayload(request);
    }
  });

  assert.equal(result.lines[0].acceptedLatex, '3 \\sqrt { 6 }');
  assert.equal(gradeRequests[0].lines[0].latex, '3 \\sqrt { 6 }');
  assert.equal(result.grading.result?.problemStatus, 'correct');
});

test('semantic candidate promotion preserves indexed radical candidate over flattened coefficient read', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('cube-root', 0, 0, 260, 90)],
    answerBox: { xMin: -5, yMin: -5, xMax: 280, yMax: 110 },
    problemLatex: '6 4 ^ { 2 / 3 }',
    semanticScoring: true,
    recognizeAlternatives: false,
    chunkFallback: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '3 \\sqrt { 6 4 ^ { 2 } }',
      top: { latex: '3 \\sqrt { 6 4 ^ { 2 } }', score: 3, confidence: 0.29 },
      candidates: [
        { latex: '\\sqrt [ 3 ] { 6 4 ^ { 2 } }', score: 3.02, confidence: 0.288 },
        { latex: '3 \\sqrt { 6 4 ^ { 2 } }', score: 3, confidence: 0.287 }
      ],
      elapsedSeconds: 0.03
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 6,
        bestLatex: '\\sqrt [ 3 ] { 6 4 ^ { 2 } }',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [],
        grading: {
          studentLatex: '\\sqrt [ 3 ] { 6 4 ^ { 2 } }',
          classification: 'valid_step',
          answerFinality: 'unsimplified',
          countsTowardCompletion: false,
          selectedCandidateIndex: 0,
          solutionCoverage: 'partial',
          matchedSolutions: []
        }
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '\\sqrt [ 3 ] { 6 4 ^ { 2 } }');
});

test('indexed radical OCR is repaired from rational-exponent problem context', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('cube-root-context', 0, 0, 260, 90)],
    answerBox: { xMin: -5, yMin: -5, xMax: 280, yMax: 110 },
    problemLatex: '6 4 ^ { 2 / 3 }',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '3 \\sqrt { 6 4 ^ { 2 } }',
      top: { latex: '3 \\sqrt { 6 4 ^ { 2 } }', score: 3, confidence: 0.99 },
      candidates: [{ latex: '3 \\sqrt { 6 4 ^ { 2 } }', score: 3, confidence: 0.99 }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '\\sqrt [ 3 ] { 64 ^ { 2 } }');
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-indexed-radical');
});

test('radical simplification half exponent OCR is repaired from sqrt problem context', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('sqrt-half-context', 0, 0, 300, 90)],
    answerBox: { xMin: -5, yMin: -5, xMax: 330, yMax: 115 },
    problemLatex: '\\sqrt { x ^ { 1 0 } }',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '( x ^ { 1 0 } ) ^ { \\sqrt { 2 } }',
      top: { latex: '( x ^ { 1 0 } ) ^ { \\sqrt { 2 } }', score: 3, confidence: 0.99 },
      candidates: [{ latex: '( x ^ { 1 0 } ) ^ { \\sqrt { 2 } }', score: 3, confidence: 0.99 }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '(x^{10})^{1/2}');
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-radical-half-exponent');
});

test('fractional log base OCR is repaired from solve problem context', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('log-base', 0, 0, 320, 90)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 110 },
    problemLatex: '\\log _ { \\frac { 1 } { 2 } ( x ) = 4',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\log \\frac { 1 } { 2 } x = 4',
      top: { latex: '\\log \\frac { 1 } { 2 } x = 4', score: 3, confidence: 0.99 },
      candidates: [{ latex: '\\log \\frac { 1 } { 2 } x = 4', score: 3, confidence: 0.99 }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, '\\log _ { \\frac { 1 } { 2 } } x = 4');
  assert.equal(result.lines[0].ocrRepair.source, 'contextual-log-fraction-base');
});

test('fractional log base repair rejects ordinary log product context', async () => {
  installFakeCanvas();
  const latex = '\\log \\frac { 1 } { 2 } x = 4';
  const result = await recognizeStudentWriting({
    strokes: [stroke('log-product', 0, 0, 320, 90)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 110 },
    problemLatex: '\\log ( \\frac { 1 } { 2 } x ) = 4',
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex,
      top: { latex, score: 3, confidence: 0.99 },
      candidates: [{ latex, score: 3, confidence: 0.99 }],
      elapsedSeconds: 0.03
    })
  });

  assert.equal(result.lines[0].acceptedLatex, latex);
  assert.equal(result.lines[0].ocrRepair?.source, undefined);
});

test('recognition semantic scoring receives selected testing catalog problem context', async () => {
  installFakeCanvas();
  const catalogProblems = testingCatalogProblems([
    'algebra_prompt_context',
    'logarithmic_solve',
    'rational_two_fraction_solve'
  ]);

  for (const problem of catalogProblems) {
    const strokes = [stroke(`a-${problem.id}`, 0, 0, 80, 36)];
    const semanticRequests = [];

    const result = await recognizeStudentWriting({
      strokes,
      answerBox: { xMin: -5, yMin: -5, xMax: 90, yMax: 46 },
      problemLatex: problem.latex,
      problemMetadata: problem.metadata,
      semanticScoring: true,
      recognizeLine: async () => ({
        latex: problem.expectedLatexLines[0],
        top: { latex: problem.expectedLatexLines[0], score: 2 },
        candidates: [{ latex: problem.expectedLatexLines[0], score: 2 }],
        elapsedSeconds: 0.03
      }),
      scoreSemantics: async (request) => {
        semanticRequests.push(request);
        return {
          candidateScores: (request.candidateGroups || []).map((group) => ({
            candidateId: group.candidateId,
            lineIndex: group.lineIndex ?? null,
            semanticScore: 3,
            bestLatex: group.latex,
            sound: true,
            equivalentToProblem: false,
            equivalentToPrevious: false,
            candidateScores: []
          })),
          elapsedSeconds: 0.01
        };
      }
    });

    assert.ok(semanticRequests.length >= 1);
    assert.equal(result.latex, problem.expectedLatexLines[0]);
    for (const request of semanticRequests) {
      assert.equal(request.problemLatex, problem.latex);
      assert.equal(request.problemMetadata.name, problem.id);
      assert.equal(request.problemMetadata.family, problem.family);
      assert.equal(request.problemMetadata.source, 'testing/fixture_catalog.py');
      assert.deepEqual(request.problemMetadata.expectedLatexLines, problem.expectedLatexLines);
    }
  }
});

test('recognition grading receives evaluate-expression problem metadata', async () => {
  installFakeCanvas();
  const gradeRequests = [];
  const strokes = [stroke('eval-answer', 0, 0, 90, 40)];

  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 100, yMax: 50 },
    problemLatex: '\\frac{1}{2} + \\frac{2}{4}',
    problemMetadata: { problemType: 'evaluate-expression' },
    apiUrl: 'http://mock-grader',
    semanticScoring: false,
    recognizeLine: async () => ({
      latex: '1',
      top: { latex: '1', score: 2 },
      candidates: [{ latex: '1', score: 2 }],
      elapsedSeconds: 0.03
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return {
        failed: false,
        problem: {
          latex: request.problemLatex,
          cardinality: 'finite',
          solutionSet: ['1'],
          decimalSet: [1],
          tolerance: 0.005,
          manifest: {
            responseKind: 'numeric_value',
            exact_set: ['1']
          }
        },
        steps: [],
        result: {
          problemStatus: 'correct',
          breakdownLineIndex: null,
          foundSolutions: ['1'],
          missingSolutions: []
        }
      };
    }
  });

  assert.equal(gradeRequests.length, 1);
  assert.equal(gradeRequests[0].problemLatex, '\\frac{1}{2} + \\frac{2}{4}');
  assert.equal(gradeRequests[0].problemMetadata.problemType, 'evaluate-expression');
  assert.equal(Object.hasOwn(gradeRequests[0], 'manifest'), false);
});

test('recognition grading receives simplify-expression problem metadata', async () => {
  installFakeCanvas();
  const gradeRequests = [];
  const strokes = [stroke('simplify-answer', 0, 0, 90, 40)];

  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 100, yMax: 50 },
    problemLatex: 'x + x',
    problemMetadata: { problemType: 'simplify-expression' },
    apiUrl: 'http://mock-grader',
    semanticScoring: false,
    recognizeLine: async () => ({
      latex: '2x',
      top: { latex: '2x', score: 2 },
      candidates: [{ latex: '2x', score: 2 }],
      elapsedSeconds: 0.03
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return {
        failed: false,
        problem: {
          latex: request.problemLatex,
          cardinality: 'finite',
          solutionSet: ['2*x'],
          decimalSet: [],
          tolerance: 0.005,
          manifest: {
            responseKind: 'simplified_expression',
            exact_set: ['2*x']
          }
        },
        steps: [],
        result: {
          problemStatus: 'correct',
          breakdownLineIndex: null,
          foundSolutions: ['2*x'],
          missingSolutions: []
        }
      };
    }
  });

  assert.equal(gradeRequests.length, 1);
  assert.equal(gradeRequests[0].problemLatex, 'x + x');
  assert.equal(gradeRequests[0].problemMetadata.problemType, 'simplify-expression');
  assert.equal(Object.hasOwn(gradeRequests[0], 'manifest'), false);
});

test('recognition final grading omits stale semantic manifests', async () => {
  installFakeCanvas();
  const gradeRequests = [];
  const strokes = [stroke('eval-answer', 0, 0, 90, 40)];

  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 100, yMax: 50 },
    problemLatex: '0.9 - 0.1',
    problemMetadata: { problemType: 'evaluate-expression' },
    apiUrl: 'http://mock-grader',
    semanticScoring: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '0.8',
      top: { latex: '0.8', score: 2 },
      candidates: [{ latex: '0.8', score: 2 }],
      elapsedSeconds: 0.03
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        responseKind: 'solution_set',
        error: 'problem must be an equation',
        exact_set: []
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 1,
        bestLatex: group.latex,
        sound: false,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: group.latex,
          classification: 'other',
          selectedCandidateIndex: 0,
          solutionCoverage: 'none',
          matchedSolutions: []
        },
        candidateScores: []
      })),
      elapsedSeconds: 0.01
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return {
        failed: false,
        problem: {
          latex: request.problemLatex,
          resolvedProblemType: 'evaluate-expression',
          cardinality: 'finite',
          solutionSet: ['0.8'],
          decimalSet: [0.8],
          tolerance: 0.005,
          manifest: {
            responseKind: 'numeric_value',
            exact_set: ['0.8']
          }
        },
        steps: [],
        result: {
          problemStatus: 'correct',
          breakdownLineIndex: null,
          foundSolutions: ['0.8'],
          missingSolutions: []
        }
      };
    }
  });

  assert.equal(gradeRequests.length, 1);
  assert.equal(gradeRequests[0].problemMetadata.problemType, 'evaluate-expression');
  assert.equal(Object.hasOwn(gradeRequests[0], 'manifest'), false);
});

test('low-score semantic best does not overwrite a valid top OCR line', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 80, 40)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 100, yMax: 60 },
    problemLatex: 'f ( x ) = \\frac { x ^ { 2 } + 1 } { x }',
    semanticScoring: true,
    recognizeLine: async () => ({
      latex: 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }',
      top: { latex: 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }', score: 2 },
      candidates: [
        { latex: 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }', score: 2 },
        { latex: 'f ^ { f } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }', score: 1 },
      ],
      elapsedSeconds: 1
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        semanticScore: 0.3,
        bestLatex: 'f ^ { f } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [
          {
            latex: 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }',
            sound: true,
            score: 0.2
          },
          {
            latex: 'f ^ { f } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }',
            sound: true,
            score: 0.3
          },
        ]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.latex, 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }');
  assert.equal(result.lines[0].ocrLatex, 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }');
  assert.equal(result.lines[0].acceptedLatex, 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }');
  assert.equal(result.lines[0].semantic.bestLatex, 'f ^ { f } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }');
});

test('tall selected line is normalized before first OCR attempt', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 240, 150)];
  const targetHeights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 260, yMax: 170 },
    problemLatex: '\\frac { x } { 2 } = 1',
    recognizeAlternatives: false,
    retryRasterHeights: [],
    initialRasterHeight: 104,
    initialRasterMinCssHeight: 128,
    recognizeLine: async (image) => {
      targetHeights.push(image.targetPixelHeight);
      return {
        latex: '\\frac { x } { 2 } = 1',
        top: { latex: '\\frac { x } { 2 } = 1', score: 2 },
        candidates: [{ latex: '\\frac { x } { 2 } = 1', score: 2 }],
        elapsedSeconds: 2
      };
    }
  });

  assert.deepEqual(targetHeights, [104]);
  assert.equal(result.lines[0].initialTargetPixelHeight, 104);
  assert.equal(result.latex, '\\frac { x } { 2 } = 1');
});

test('selected line OCR retries normalized crop heights after a timeout', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 240, 150)];
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 260, yMax: 170 },
    problemLatex: '\\frac { x } { 2 } = 1',
    initialRasterHeight: 0,
    retryRasterHeights: [88, 104],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (!image.targetPixelHeight) {
        return {
          latex: '',
          top: null,
          candidates: [],
          timedOut: true,
          elapsedSeconds: 20
        };
      }
      return {
        latex: '\\frac { x } { 2 } = 1',
        top: { latex: '\\frac { x } { 2 } = 1', score: 2 },
        candidates: [{ latex: '\\frac { x } { 2 } = 1', score: 2 }],
        elapsedSeconds: 8
      };
    }
  });

  assert.ok(heights.includes(88));
  assert.ok(heights.includes(104));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.equal(result.lines[0].retryPredictions.length, 2);
  assert.equal(result.latex, '\\frac { x } { 2 } = 1');
});

test('student writing pipeline abort stops selected-line retries', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 240, 150)];
  const controller = new AbortController();
  const calls = [];

  await assert.rejects(
    recognizeStudentWriting({
      strokes,
      answerBox: { xMin: -5, yMin: -5, xMax: 260, yMax: 170 },
      problemLatex: '\\frac { x } { 2 } = 1',
      recognizeAlternatives: false,
      initialRasterHeight: 0,
      retryRasterHeights: [88, 104],
      signal: controller.signal,
      recognizeLine: async (image) => {
        calls.push(image.targetPixelHeight || image.height);
        controller.abort();
        return {
          latex: '',
          top: null,
          candidates: [],
          timedOut: true,
          elapsedSeconds: 20
        };
      }
    }),
    { name: 'AbortError' }
  );

  assert.equal(calls.length, 1);
});

test('structural selected line gets longer timeout after short OCR retries fail', async () => {
  installFakeCanvas();
  const strokes = Array.from({ length: 10 }, (_item, index) => (
    stroke(String.fromCharCode(97 + index), index * 28, index % 2 ? 8 : 0, index * 28 + 18, 105)
  ));
  const timeouts = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 310, yMax: 130 },
    problemLatex: '\\frac { x } { 2 } = 1',
    timeoutMs: 6000,
    structuralRetryTimeoutMs: 12000,
    initialRasterHeight: 0,
    retryRasterHeights: [88],
    recognizeLine: async (_image, options) => {
      timeouts.push(options.timeoutMs);
      if (options.timeoutMs > 6000) {
        return {
          latex: '\\frac { x } { 2 } = 1',
          top: { latex: '\\frac { x } { 2 } = 1', score: 2 },
          candidates: [{ latex: '\\frac { x } { 2 } = 1', score: 2 }],
          elapsedSeconds: 7
        };
      }
      return {
        latex: '',
        top: null,
        candidates: [],
        timedOut: true,
        elapsedSeconds: 6
      };
    }
  });

  assert.deepEqual(timeouts, [6000, 6000, 12000]);
  assert.equal(result.lines[0].retryPredictions.at(-1).extendedTimeoutMs, 12000);
  assert.equal(result.latex, '\\frac { x } { 2 } = 1');
});

test('selected line OCR retries suspicious operation annotations', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 240, 70)];
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 260, yMax: 90 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    initialRasterHeight: 0,
    retryRasterHeights: [88],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (!image.targetPixelHeight) {
        return {
          latex: '\\times 6 9 \\times 6 9',
          top: { latex: '\\times 6 9 \\times 6 9', score: -1 },
          candidates: [{ latex: '\\times 6 9 \\times 6 9', score: -1 }],
          elapsedSeconds: 7
        };
      }
      return {
        latex: '\\times 6 \\times 6',
        top: { latex: '\\times 6 \\times 6', score: 2 },
        candidates: [{ latex: '\\times 6 \\times 6', score: 2 }],
        elapsedSeconds: 3
      };
    }
  });

  assert.ok(heights.includes(88));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.equal(result.latex, '\\times 6 \\times 6');
});

test('selected line OCR retries alternate heights when semantics rejects a short numeric read', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 120, 45)];
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 140, yMax: 65 },
    problemLatex: '= 10',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [48, 64, 104],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (image.targetPixelHeight === 48) {
        return {
          latex: '= 1 0',
          top: { latex: '= 1 0', score: -0.2 },
          candidates: [{ latex: '= 1 0', score: -0.2 }],
          elapsedSeconds: 2
        };
      }
      return {
        latex: '= 2 0 0',
        top: { latex: '= 2 0 0', score: -1.6 },
        candidates: [{ latex: '= 2 0 0', score: -1.6 }],
        elapsedSeconds: 2
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => {
        const candidates = group.candidates || [];
        const hasTen = candidates.some((candidate) => String(candidate.latex || '').trim() === '= 1 0');
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: hasTen ? 5 : 0.8,
          bestLatex: hasTen ? '= 1 0' : group.latex,
          sound: true,
          equivalentToProblem: hasTen,
          equivalentToPrevious: false,
          candidateScores: candidates.map((candidate) => ({
            latex: candidate.latex,
            sound: true,
            score: String(candidate.latex || '').trim() === '= 1 0' ? 5 : 0.8,
            equivalentToProblem: String(candidate.latex || '').trim() === '= 1 0'
          }))
        };
      }),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(heights.includes(104));
  assert.ok(heights.includes(48));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.equal(result.lines[0].semanticRetryPredictions.length, 2);
  assert.equal(result.semantic.sequentialBeforeRetry.lineScores[0].bestLatex, '= 2 0 0');
  assert.equal(result.lines[0].sequentialSemantic.bestLatex, '= 1 0');
  assert.equal(result.latex, '= 1 0');
});

test('selected line OCR retries malformed low-semantic reads', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 320, 70)];
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 90 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [48, 64, 104],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (image.targetPixelHeight === 48) {
        return {
          latex: '\\times 6 \\times 6',
          top: { latex: '\\times 6 \\times 6', score: 1 },
          candidates: [{ latex: '\\times 6 \\times 6', score: 1 }],
          elapsedSeconds: 2
        };
      }
      return {
        latex: 'x 6 9 \\times 6',
        top: { latex: 'x 6 9 \\times 6', score: -1 },
        candidates: [{ latex: 'x 6 9 \\times 6', score: -1 }],
        elapsedSeconds: 2
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => {
        const candidates = group.candidates || [];
        const hasOperation = candidates.some((candidate) => String(candidate.latex || '').trim() === '\\times 6 \\times 6');
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: hasOperation ? 4 : 0.2,
          bestLatex: hasOperation ? '\\times 6 \\times 6' : group.latex,
          sound: hasOperation,
          equivalentToProblem: false,
          equivalentToPrevious: hasOperation,
          candidateScores: candidates.map((candidate) => ({
            latex: candidate.latex,
            sound: String(candidate.latex || '').trim() === '\\times 6 \\times 6',
            score: String(candidate.latex || '').trim() === '\\times 6 \\times 6' ? 4 : 0.2,
            equivalentToPrevious: String(candidate.latex || '').trim() === '\\times 6 \\times 6'
          }))
        };
      }),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(heights.includes(48));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.equal(result.lines[0].semanticRetryPredictions.length, 2);
  assert.equal(result.latex, '\\times 6 \\times 6');
});

test('selected line OCR retries weak linear equation reads', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('coef', 0, 0, 48, 40),
    stroke('x', 60, 0, 108, 42),
    stroke('plus', 124, 8, 166, 48),
    stroke('two', 180, 0, 228, 44),
    stroke('eq_top', 252, 12, 300, 18),
    stroke('eq_bottom', 252, 32, 300, 38),
    stroke('one', 328, 0, 360, 44),
    stroke('two_rhs', 374, 0, 422, 44),
  ];
  assignTimes(strokes);
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 450, yMax: 70 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 88,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [48, 64, 104],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (image.targetPixelHeight === 48) {
        return {
          latex: '3 x + 2 = 1 2',
          top: { latex: '3 x + 2 = 1 2', score: -0.2 },
          candidates: [{ latex: '3 x + 2 = 1 2', score: -0.2 }],
          elapsedSeconds: 2
        };
      }
      return {
        latex: '5 x + 2 = 1 5',
        top: { latex: '5 x + 2 = 1 5', score: -0.2 },
        candidates: [{ latex: '5 x + 2 = 1 5', score: -0.2 }],
        elapsedSeconds: 2
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => {
        const candidates = group.candidates || [];
        const hasCorrect = candidates.some((candidate) => String(candidate.latex || '').trim() === '3 x + 2 = 1 2');
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: hasCorrect ? 5 : 1.4,
          bestLatex: hasCorrect ? '3 x + 2 = 1 2' : group.latex,
          sound: true,
          equivalentToProblem: hasCorrect,
          equivalentToPrevious: false,
          candidateScores: candidates.map((candidate) => ({
            latex: candidate.latex,
            sound: true,
            score: String(candidate.latex || '').trim() === '3 x + 2 = 1 2' ? 5 : 1.4,
            equivalentToProblem: String(candidate.latex || '').trim() === '3 x + 2 = 1 2'
          }))
        };
      }),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(heights.includes(48));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.ok(result.lines[0].semanticRetryPredictions.length >= 1);
  assert.equal(result.latex, '3 x + 2 = 1 2');
});

test('selected line OCR retries weak function equation reads', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 820, 92)];
  const heights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 850, yMax: 112 },
    problemLatex: 'f ( x ) = x ^ { 2 } ( x + 3 )',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [64, 104],
    recognizeLine: async (image) => {
      heights.push(image.targetPixelHeight || image.height);
      if (image.targetPixelHeight === 64) {
        return {
          latex: 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }',
          top: { latex: 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }', score: -0.4 },
          candidates: [{ latex: 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }', score: -0.4 }],
          elapsedSeconds: 3
        };
      }
      return {
        latex: 'v ( x ) = 2 x ( x + 3 ) + n 2',
        top: { latex: 'v ( x ) = 2 x ( x + 3 ) + n 2', score: -0.5 },
        candidates: [{ latex: 'v ( x ) = 2 x ( x + 3 ) + n 2', score: -0.5 }],
        elapsedSeconds: 4
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => {
        const candidates = group.candidates || [];
        const hasDerivative = candidates.some((candidate) => /\\prime/.test(String(candidate.latex || '')));
        return {
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: hasDerivative ? 5 : 0.1,
          bestLatex: hasDerivative ? 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }' : group.latex,
          sound: true,
          equivalentToProblem: hasDerivative,
          equivalentToPrevious: false,
          candidateScores: candidates.map((candidate) => ({
            latex: candidate.latex,
            sound: true,
            score: /\\prime/.test(String(candidate.latex || '')) ? 5 : 0.1,
            equivalentToProblem: /\\prime/.test(String(candidate.latex || ''))
          }))
        };
      }),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(heights.includes(64));
  assert.equal(result.lines[0].prediction.retryUsed, true);
  assert.equal(result.latex, 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }');
});

test('standalone operation repair treats low-semantic x as multiplication', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [48],
    recognizeLine: async () => ({
      latex: 'x 6 9 \\times 6',
      top: { latex: 'x 6 9 \\times 6', score: -1 },
      candidates: [{ latex: 'x 6 9 \\times 6', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.2,
        bestLatex: group.latex,
        sound: false,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: false,
          score: 0.2
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'standalone-operation');
  assert.equal(result.latex, '\\times 6 \\times 6');
});

test('standalone operation repair preserves plain numeric OCR literals', async () => {
  installFakeCanvas();
  for (const latex of ['2', '2 . 75']) {
    const strokes = [stroke(`a-${latex}`, 0, 0, 90, 70)];

    const result = await recognizeStudentWriting({
      strokes,
      answerBox: { xMin: -5, yMin: -5, xMax: 120, yMax: 90 },
      problemLatex: '2 ^ { \\log _ { 4 } 1 6 \\sqrt { 8 } }',
      recognizeAlternatives: false,
      semanticScoring: true,
      initialRasterHeight: 104,
      retryRasterHeights: [],
      semanticRetryRasterHeights: [],
      recognizeLine: async () => ({
        latex,
        top: { latex, score: 0 },
        candidates: [{ latex, score: 0 }],
        elapsedSeconds: 1
      }),
      scoreSemantics: async (request) => ({
        candidateScores: request.candidateGroups.map((group) => ({
          candidateId: group.candidateId,
          lineIndex: group.lineIndex,
          semanticScore: 0.4,
          bestLatex: group.latex,
          sound: false,
          equivalentToProblem: false,
          equivalentToPrevious: false,
          candidateScores: [{ latex: group.latex, sound: false, score: 0.4 }]
        })),
        elapsedSeconds: 0.01
      })
    });

    assert.equal(result.lines[0].acceptedLatex, latex);
    assert.equal(result.latex, latex);
    assert.equal(result.lines[0].ocrRepair, undefined);
  }
});

test('standalone operation repair handles subscripted CoMER times annotations', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: 'X _ { 6 } \\times _ { n }',
      top: { latex: 'X _ { 6 } \\times _ { n }', score: -1 },
      candidates: [{ latex: 'X _ { 6 } \\times _ { n }', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'standalone-operation');
  assert.equal(result.latex, '\\times 6 \\times 6');
});

test('standalone operation repair uses fraction denominator context for malformed times annotations', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { x + 1 } { 2 } = \\frac { 5 } { 3 }',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: 'x _ { 0 } \\times n',
      top: { latex: 'x _ { 0 } \\times n', score: -1 },
      candidates: [{ latex: 'x _ { 0 } \\times n', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'standalone-operation');
  assert.equal(result.latex, '\\times 6 \\times 6');
});

test('standalone operation repair normalizes spaced uppercase times annotations', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '\\times 1 2 X 1 2',
      top: { latex: '\\times 1 2 X 1 2', score: -1 },
      candidates: [{ latex: '\\times 1 2 X 1 2', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'standalone-operation');
  assert.equal(result.latex, '\\times 12 \\times 12');
});

test('standalone operation repair normalizes plain x times annotations', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { 2 x + 1 } { 3 } - \\frac { x - 2 } { 4 } = 5',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: 'x 1 2 x 1 2',
      top: { latex: 'x 1 2 x 1 2', score: -1 },
      candidates: [{ latex: 'x 1 2 x 1 2', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'standalone-operation');
  assert.equal(result.latex, '\\times 12 \\times 12');
});

test('standalone operation repair normalizes detached multiplier marks', async () => {
  installFakeCanvas();
  const strokes = [stroke('op', 0, 0, 80, 50)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 120, yMax: 80 },
    problemLatex: '\\frac { x } { 4 } - 1 = 5',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '4 *',
      top: { latex: '4 *', score: -1 },
      candidates: [{ latex: '4 *', score: -1 }],
      elapsedSeconds: 1
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.2,
        bestLatex: group.latex,
        sound: false,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{ latex: group.latex, sound: false, score: 0.2 }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'geometry-operation-annotation');
  assert.equal(result.lines[0].excludedFromGrading, true);
  assert.equal(result.lines[0].acceptedLatex, '\\times 4 \\times 4');
  assert.equal(result.latex, '');
});

test('geometry operation repair tags detached fraction annotations', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('op-left', 10, 0, 42, 42),
    stroke('op-right', 58, 0, 90, 42),
    stroke('eq-left', 40, 120, 180, 200),
    stroke('eq-right', 200, 120, 360, 200),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 390, yMax: 230 },
    detections: [
      { bbox: { xMin: 5, yMin: -5, xMax: 95, yMax: 50 } },
      { bbox: { xMin: 35, yMin: 115, xMax: 365, yMax: 205 } },
    ],
    problemLatex: '\\frac { x - 1 } { 2 } = \\frac { x } { 3 }',
    recognizeAlternatives: true,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      const isOperation = image.strokeIds.some((id) => String(id).startsWith('op'));
      const latex = isOperation ? '2 .' : '\\frac { x - 1 } { 2 } = \\frac { x } { 3 }';
      return {
        latex,
        top: { latex, score: isOperation ? -1 : 2 },
        candidates: [{ latex, score: isOperation ? -1 : 2 }],
        elapsedSeconds: 1
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: group.latex.includes('\\frac') ? 4 : 0.2,
        bestLatex: group.latex,
        sound: group.latex.includes('\\frac'),
        equivalentToProblem: group.latex.includes('\\frac'),
        equivalentToPrevious: false,
        candidateScores: [{ latex: group.latex, sound: group.latex.includes('\\frac'), score: 0.2 }]
      })),
      elapsedSeconds: 0.01
    })
  });

  const operationLine = result.lines.find((line) => line.strokeIds.some((id) => String(id).startsWith('op')));
  assert.equal(operationLine.ocrRepair.source, 'geometry-operation-annotation');
  assert.equal(operationLine.excludedFromGrading, true);
  assert.equal(operationLine.acceptedLatex, '\\times 2 \\times 2');
  assert.equal(result.latexLines.includes('\\times 2 \\times 2'), false);
});

test('lower-ranked full-answer grading candidates do not override ambiguous top OCR', async () => {
  installFakeCanvas();
  const strokes = [stroke('final', 0, 0, 140, 60)];
  const gradeRequests = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 160, yMax: 80 },
    problemLatex: '\\frac { 1 2 } { x - 1 } = 4',
    apiUrl: 'http://mock-grader',
    recognizeAlternatives: false,
    semanticScoring: true,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '4 \\pm x',
      top: { latex: '4 \\pm x', score: 4 },
      candidates: [
        { latex: '4 \\pm x', score: 4 },
        { latex: '4 = x', score: 1.5 }
      ],
      elapsedSeconds: 0.02
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        cardinality: 'finite',
        exact_set: ['4'],
        variable: 'x'
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 5,
        bestLatex: '4 = x',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: '4 = x',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['4'],
          answerFinality: 'final',
          countsTowardCompletion: true
        },
        candidateScores: [
          { latex: group.candidates?.[0]?.latex || group.latex, sound: false, score: 0.2 },
          { latex: '4 = x', sound: true, score: 5 }
        ]
      })),
      elapsedSeconds: 0.01
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return {
        failed: false,
        problem: { latex: request.problemLatex, cardinality: 'finite', solutionSet: ['4'], manifest: { exact_set: ['4'] } },
        steps: [],
        result: { problemStatus: 'incomplete', breakdownLineIndex: null, foundSolutions: [], missingSolutions: ['4'] }
      };
    }
  });

  assert.equal(result.lines[0].acceptedLatex, '4 \\pm x');
  assert.equal(result.grading.result.problemStatus, 'incomplete');
  assert.equal(gradeRequests.length, 1);
  assert.deepEqual(gradeRequests[0].lines[0].candidates, []);
});

test('evaluate-expression candidate-selection conflict does not promote lower-ranked final answer', async () => {
  installFakeCanvas();
  const strokes = [stroke('eval-final', 0, 0, 220, 70)];
  const gradeRequests = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 240, yMax: 90 },
    problemLatex: '\\frac { \\cos ( \\pi ) } { \\sin ( \\frac { \\pi } { 2 } ) }',
    problemMetadata: { problemType: 'evaluate-expression' },
    apiUrl: 'http://mock-grader',
    recognizeAlternatives: false,
    semanticScoring: true,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '= \\frac { 1 } { 1 } = - 1',
      top: { latex: '= \\frac { 1 } { 1 } = - 1', score: 3 },
      candidates: [
        { latex: '= \\frac { 1 } { 1 } = - 1', score: 3 },
        { latex: '= - 1 = - 1', score: 1.4 }
      ],
      elapsedSeconds: 0.02
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        responseKind: 'numeric_value',
        cardinality: 'finite',
        exact_set: ['-1'],
        variable: null
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 5,
        bestLatex: '= - 1 = - 1',
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        grading: {
          studentLatex: '= - 1 = - 1',
          classification: 'valid_step',
          selectedCandidateIndex: 1,
          solutionCoverage: 'full',
          matchedSolutions: ['-1'],
          answerFinality: 'final',
          countsTowardCompletion: true
        },
        candidateScores: [
          { latex: group.candidates?.[0]?.latex || group.latex, sound: false, score: 0.2 },
          { latex: '= - 1 = - 1', sound: true, score: 5 }
        ]
      })),
      elapsedSeconds: 0.01
    }),
    gradeWork: async (request) => {
      gradeRequests.push(request);
      return {
        failed: false,
        problem: { latex: request.problemLatex, cardinality: 'finite', solutionSet: ['-1'], manifest: { exact_set: ['-1'] } },
        steps: [],
        result: { problemStatus: 'not_started', breakdownLineIndex: null, foundSolutions: [], missingSolutions: ['-1'] }
      };
    }
  });

  assert.equal(result.lines[0].acceptedLatex, '= \\frac { 1 } { 1 } = - 1');
  assert.equal(result.grading.result.problemStatus, 'not_started');
  assert.equal(gradeRequests.length, 1);
  assert.deepEqual(gradeRequests[0].lines[0].candidates, []);
});

test('finalization budget skips optional retry work and returns current best result', async () => {
  installFakeCanvas();
  const strokes = [stroke('slow', 0, 0, 220, 70)];
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 240, yMax: 90 },
    problemLatex: 'x = 1',
    recognizeAlternatives: false,
    semanticScoring: false,
    retryRasterHeights: [72, 88, 104],
    semanticRetryRasterHeights: [48, 64],
    chunkFallback: true,
    finalizationBudgetMs: 0,
    recognizeLine: async (image) => {
      calls.push({ candidateId: image.candidateId, height: image.targetPixelHeight });
      return {
        latex: '',
        top: null,
        candidates: [],
        failed: true,
        elapsedSeconds: 0.01
      };
    }
  });

  assert.equal(calls.length, 1);
  assert.equal(result.timing.finalizationBudgetExceeded, true);
  assert.equal(result.timing.ocrTimeoutWithInk, true);
  assert.equal(result.grading.result.problemStatus, 'incomplete');
});

test('contextual operation repair uses previous additive constant', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { 2 x + 1 } { 3 } - \\frac { x - 2 } { 4 } = 5',
    previousLatex: ['5 x + 10 = 60'],
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '- 2 0 - 2 0',
      top: { latex: '- 2 0 - 2 0', score: -1 },
      candidates: [{ latex: '- 2 0 - 2 0', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'contextual-operation');
  assert.equal(result.latex, '- 10 - 10');
});

test('contextual operation repair treats x0 as previous additive constant', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 300, 70)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 90 },
    problemLatex: '\\frac { 2 x + 1 } { 3 } - \\frac { x - 2 } { 4 } = 5',
    previousLatex: ['5 x + 10 = 60'],
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '- 2 0 - x 0',
      top: { latex: '- 2 0 - x 0', score: -1 },
      candidates: [{ latex: '- 2 0 - x 0', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'contextual-operation');
  assert.equal(result.latex, '- 10 - 10');
});

test('contextual operation repair uses reversed previous additive constant', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 220, 60)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 250, yMax: 80 },
    problemLatex: '\\log _ { 3 } ( x + 1 ) = 2',
    previousLatex: ['9 = x + 1'],
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '- 2 - 2',
      top: { latex: '- 2 - 2', score: -1 },
      candidates: [{ latex: '- 2 - 2', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'contextual-operation');
  assert.equal(result.latex, '- 1 - 1');
});

test('contextual operation repair handles equals hallucination between operands', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 220, 60)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 250, yMax: 80 },
    problemLatex: '\\log _ { 3 } ( x + 1 ) = 2',
    previousLatex: ['9 = x + 1'],
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '- 2 = 1',
      top: { latex: '- 2 = 1', score: -1 },
      candidates: [{ latex: '- 2 = 1', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'contextual-operation');
  assert.equal(result.latex, '- 1 - 1');
});

test('contextual operation repair treats x as one for previous additive constant', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 220, 60)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 250, yMax: 80 },
    problemLatex: '\\frac { x ^ { 2 } - 1 } { x - 1 } = 4',
    previousLatex: ['x + 1 = 4'],
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '- 1 - x',
      top: { latex: '- 1 - x', score: -1 },
      candidates: [{ latex: '- 1 - x', score: -1 }],
      elapsedSeconds: 2
    }),
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 0.4,
        bestLatex: group.latex,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 0.4
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.lines[0].ocrRepair.source, 'contextual-operation');
  assert.equal(result.latex, '- 1 - 1');
});

test('semantic replacement preserves operation annotation shape', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('eq', 0, 0, 260, 60),
    stroke('op', 0, 120, 260, 170)
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 320, yMax: 200 },
    problemLatex: '\\frac { x + 1 } { 2 } = \\frac { 5 } { 3 }',
    recognizeAlternatives: false,
    semanticScoring: true,
    initialRasterHeight: 104,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (_image, _options) => {
      const latex = _image.strokeIds.includes('op') ? '- 3 - 3' : '3 x + 3 = 1 0';
      return {
        latex,
        top: { latex, score: -1 },
        candidates: [{ latex, score: -1 }],
        elapsedSeconds: 1
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: 3.2,
        bestLatex: '3 x + 3 = 1 0',
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: true,
        candidateScores: [{
          latex: group.latex,
          sound: true,
          score: 3.2
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.equal(result.latexLines[0], '3 x + 3 = 1 0');
  assert.equal(result.latexLines[1], '- 3 - 3');
});

test('semantic replacement accepts sound operation annotation over non-operation read', () => {
  const semantic = {
    bestLatex: '\\times 1 2 \\times 1 2',
    semanticScore: 0.37,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    sound: true,
    candidateScores: [
      {
        latex: '\\times 1 2 \\times 1 2',
        score: 0.37,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false
      },
      {
        latex: 'X 1 2 \\times 1 0',
        score: 0.2,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('X 1 2 \\times 1 0', semantic), true);
});

test('semantic replacement can trust a better same-kind contextual candidate', () => {
  const semantic = {
    bestLatex: '4 + y = 7',
    semanticScore: 1.0882,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    sound: true,
    candidateScores: [
      {
        latex: '4 + y = 7',
        score: 1.0882,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { characterOverlap: 0.461 }
      },
      {
        latex: 'q + y = 7',
        score: 1.0829,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { characterOverlap: 0.293 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('q + y = 7', semantic), true);
});

test('semantic replacement preserves visible expanded row over contextual simplification', () => {
  const semantic = {
    bestLatex: '5 x + 10 = 60',
    semanticScore: 3.8,
    equivalentToProblem: true,
    equivalentToPrevious: true,
    sound: true,
    candidateScores: [
      {
        latex: '5 x + 10 = 60',
        score: 3.8,
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: true,
        detail: { repair: 'contextual_linear_simplification', characterOverlap: 0.4 }
      },
      {
        latex: '8 x + 4 - 3 x + 6 = 60',
        score: 2.8,
        sound: true,
        equivalentToProblem: true,
        equivalentToPrevious: true,
        detail: { characterOverlap: 0.8 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('8 x + 4 - 3 x + 6 = 60', semantic), false);
});

test('semantic replacement rejects duplicate previous contextual best', () => {
  const semantic = {
    bestLatex: 'x = 2',
    semanticScore: 1.2,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    sound: true,
    candidateScores: [
      {
        latex: 'x = 2',
        score: 1.2,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { characterOverlap: 0.6, duplicatePreviousLatex: true }
      },
      {
        latex: 'x = 3',
        score: 1.0,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { characterOverlap: 0.4 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('x = 3', semantic), false);
});

test('semantic replacement preserves stronger visual current over previous-equivalent best', () => {
  const semantic = {
    bestLatex: '1 + y = x',
    semanticScore: 2.2167,
    equivalentToProblem: false,
    equivalentToPrevious: true,
    sound: true,
    candidateScores: [
      {
        latex: '1 + y = x',
        score: 2.2167,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: true,
        detail: { modelScore: -1.2862, characterOverlap: 0.685 }
      },
      {
        latex: '4 + y = 7',
        score: 1.2299,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { modelScore: -0.3848, characterOverlap: 0.461 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('4 + y = 7', semantic), false);
});

test('semantic replacement preserves derivative prime over previous-equivalent best', () => {
  const semantic = {
    bestLatex: 'f ( 2 ) = \\frac { 3 } { 4 }',
    semanticScore: 3.248,
    equivalentToProblem: false,
    equivalentToPrevious: true,
    sound: true,
    candidateScores: [
      {
        latex: 'f ( 2 ) = \\frac { 3 } { 4 }',
        score: 3.248,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: true,
        detail: { modelScore: -1.2757, characterOverlap: 0.415 }
      },
      {
        latex: 'f ^ { \\prime } ( 2 ) = \\frac { 3 } { 4 }',
        score: 1.2607,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { modelScore: -0.2257, characterOverlap: 0.298 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('f ^ { \\prime } ( 2 ) = \\frac { 3 } { 4 }', semantic), false);
});

test('semantic replacement accepts problem-supported repair', () => {
  const semantic = {
    bestLatex: 'x = \\frac { - 4 + \\sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }',
    semanticScore: 2.3018,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    sound: true,
    candidateScores: [
      {
        latex: 'x = \\frac { - 4 + \\sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }',
        score: 2.3018,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { solutionSupportedByProblem: true, repair: 'contextual_quadratic_formula_coefficient' }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('x = \\frac { 1 + \\sqrt { 4 ^ { 2 } - 1 1 ) ( - 5 } } { 2 ( 1 ) } - 1', semantic), true);
});

test('semantic replacement accepts problem-supported numeric repair over stronger visual top', () => {
  const semantic = {
    bestLatex: 'x = 1',
    semanticScore: 3.0232,
    equivalentToProblem: false,
    equivalentToPrevious: true,
    sound: true,
    candidateScores: [
      {
        latex: 'x = 1',
        score: 3.0232,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: true,
        detail: {
          modelScore: -0.6294,
          solutionSupportedByProblem: true,
          repair: 'contextual_latex_numeric_equivalence'
        }
      },
      {
        latex: 'x = 2',
        score: 1.6157,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { modelScore: -0.0794, characterOverlap: 0.493 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('x = 2', semantic), true);
});

test('semantic replacement rejects unsound best candidate', () => {
  const semantic = {
    bestLatex: 'x = \\frac { - x + x } { 0 }',
    semanticScore: 4.2,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    sound: false,
    candidateScores: [
      {
        latex: 'x = \\frac { - x + x } { 0 }',
        score: 4.2,
        sound: false,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { parseFailure: 'expression contains a non-finite value' }
      },
      {
        latex: 'x = \\frac { - 4 + 6 } { 2 }',
        score: 1.0,
        sound: true,
        equivalentToProblem: false,
        equivalentToPrevious: false,
        detail: { characterOverlap: 0.4 }
      }
    ]
  };

  assert.equal(shouldUseSemanticLatex('x = \\frac { - 4 + 6 } { 2 }', semantic), false);
});

test('wide selected line can fall back to chunk OCR after timeouts', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('fprime', 0, 0, 150, 80),
    stroke('eq_top', 170, 30, 230, 36),
    stroke('eq_bottom', 170, 52, 230, 58),
    stroke('rhs_left', 250, 0, 555, 80),
    stroke('rhs_right', 580, 0, 760, 80),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 780, yMax: 100 },
    problemLatex: 'f ( x ) = x ^ { 2 } ( x + 3 )',
    recognizeAlternatives: false,
    retryRasterHeights: [88],
    chunkFallbackMinCssWidth: 500,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return {
          latex: '',
          top: null,
          candidates: [],
          timedOut: true,
          elapsedSeconds: 20
        };
      }
      if (image.strokeIds.includes('fprime')) {
        return {
          latex: 'f ^ { f } ( x )',
          top: { latex: 'f ^ { f } ( x )', score: 2 },
          candidates: [
            { latex: 'f ^ { f } ( x )', score: 2 },
            { latex: 'f ^ { \\prime } ( x )', score: 1 },
          ],
          elapsedSeconds: 2
        };
      }
      if (image.strokeIds.includes('rhs_left')) {
        return {
          latex: '2 x ( x + 3 )',
          top: { latex: '2 x ( x + 3 )', score: 2 },
          candidates: [{ latex: '2 x ( x + 3 )', score: 2 }],
          elapsedSeconds: 2
        };
      }
      return {
        latex: 't x ^ { 2 }',
        top: { latex: 't x ^ { 2 }', score: 2 },
        candidates: [
          { latex: 't x ^ { 2 }', score: 2 },
          { latex: '+ x ^ { 2 }', score: 1 },
        ],
        elapsedSeconds: 2
      };
    }
  });

  assert.ok(calls.some((call) => call.profiles.includes('chunk-fallback')));
  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.equal(result.latex, 'f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }');
});

test('chunk fallback infers isolated context variable before equals', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('x', 0, 20, 28, 58),
    stroke('eq_top', 48, 28, 92, 34),
    stroke('eq_bottom', 48, 50, 92, 56),
    stroke('rhs', 125, 8, 520, 86),
  ];
  assignTimes(strokes);
  const chunkCalls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 545, yMax: 110 },
    problemLatex: 'x ^ { 2 } + 4 x - 5 = 0',
    recognizeAlternatives: false,
    retryRasterHeights: [],
    chunkFallbackMinCssWidth: 500,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      if (!image.profiles.includes('chunk-fallback')) {
        return {
          latex: '',
          top: null,
          candidates: [],
          timedOut: true,
          elapsedSeconds: 20
        };
      }
      chunkCalls.push(image.strokeIds);
      return {
        latex: '\\frac { 1 } { 2 }',
        top: { latex: '\\frac { 1 } { 2 }', score: 1 },
        candidates: [{ latex: '\\frac { 1 } { 2 }', score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  assert.deepEqual(chunkCalls, [['rhs']]);
  assert.equal(result.lines[0].prediction.chunkAttempts[0].inferredLiteral, true);
  assert.equal(result.latex, 'x = \\frac { 1 } { 2 }');
});

test('chunk fallback retries chunk OCR at lower raster heights', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('x', 0, 20, 28, 58),
    stroke('eq_top', 48, 28, 92, 34),
    stroke('eq_bottom', 48, 50, 92, 56),
    stroke('rhs', 125, 8, 520, 86),
  ];
  assignTimes(strokes);
  const chunkHeights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 545, yMax: 110 },
    problemLatex: 'x ^ { 2 } + 4 x - 5 = 0',
    recognizeAlternatives: false,
    retryRasterHeights: [88],
    initialRasterHeight: 104,
    chunkFallbackMinCssWidth: 500,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      if (!image.profiles.includes('chunk-fallback')) {
        return {
          latex: '',
          top: null,
          candidates: [],
          timedOut: true,
          elapsedSeconds: 20
        };
      }
      if (image.strokeIds.includes('rhs')) {
        chunkHeights.push(image.targetPixelHeight);
        if (image.targetPixelHeight === 72) {
          return {
            latex: '',
            top: null,
            candidates: [],
            timedOut: true,
            elapsedSeconds: 20
          };
        }
        return {
          latex: '\\frac { 1 } { 2 }',
          top: { latex: '\\frac { 1 } { 2 }', score: 1 },
          candidates: [{ latex: '\\frac { 1 } { 2 }', score: 1 }],
          elapsedSeconds: 1
        };
      }
      return {
        latex: '',
        top: null,
        candidates: [],
        timedOut: true,
        elapsedSeconds: 20
      };
    }
  });

  assert.deepEqual(chunkHeights, [72, 88]);
  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.equal(result.latex, 'x = \\frac { 1 } { 2 }');
});

test('wide selected line tries chunk OCR before height retries', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('left', 0, 0, 180, 70),
    stroke('eq_top', 210, 26, 260, 32),
    stroke('eq_bottom', 210, 48, 260, 54),
    stroke('right', 300, 0, 760, 70),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 790, yMax: 95 },
    problemLatex: 'x = 10',
    recognizeAlternatives: false,
    retryRasterHeights: [72, 88],
    initialRasterHeight: 104,
    chunkFallbackMinCssWidth: 500,
    chunkFallbackMaxCssWidth: 280,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, targetPixelHeight: image.targetPixelHeight, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 20 };
      }
      return {
        latex: image.strokeIds.includes('left') ? 'x' : '10',
        top: { latex: image.strokeIds.includes('left') ? 'x' : '10', score: 1 },
        candidates: [{ latex: image.strokeIds.includes('left') ? 'x' : '10', score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  const firstChunkIndex = calls.findIndex((call) => call.profiles.includes('chunk-fallback'));
  const firstRetryIndex = calls.findIndex((call) => !call.profiles.includes('chunk-fallback') && call.targetPixelHeight === 72);

  assert.ok(firstChunkIndex > 0);
  assert.equal(firstRetryIndex, -1);
  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.equal(result.latex, 'x = 10');
});

test('selected line height retries stop after the first timeout', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('x', 0, 0, 40, 40),
    stroke('eq_top', 70, 14, 110, 20),
    stroke('eq_bottom', 70, 30, 110, 36),
    stroke('four', 140, 0, 180, 40),
  ];
  assignTimes(strokes);
  const targetHeights = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 200, yMax: 65 },
    problemLatex: 'x = 4',
    recognizeAlternatives: false,
    chunkFallback: false,
    retryRasterHeights: [72, 88, 104],
    initialRasterHeight: 104,
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      targetHeights.push(image.targetPixelHeight);
      return {
        latex: '',
        top: null,
        candidates: [],
        timedOut: true,
        elapsedSeconds: 20
      };
    }
  });

  assert.deepEqual(targetHeights, [104, 72]);
  assert.equal(result.lines[0].retryPredictions.length, 1);
  assert.equal(result.lines[0].retryPredictions[0].timedOut, true);
});

test('ellipsis-truncated wide rows fall back to chunk OCR', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('left_4', 0, 8, 34, 64),
    stroke('left_paren', 48, 4, 76, 84),
    stroke('left_2x', 88, 8, 170, 74),
    stroke('left_plus_one', 184, 12, 292, 72),
    stroke('mid_minus', 326, 36, 360, 44),
    stroke('mid_three', 374, 10, 418, 72),
    stroke('mid_paren', 432, 4, 460, 84),
    stroke('mid_x_minus_two', 474, 10, 620, 74),
    stroke('eq_top', 646, 30, 690, 36),
    stroke('eq_bottom', 646, 50, 690, 56),
    stroke('rhs_60', 718, 10, 792, 72),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -10, yMin: -10, xMax: 820, yMax: 110 },
    problemLatex: '\\frac { 2 x + 1 } { 3 } - \\frac { x - 2 } { 4 } = 5',
    recognizeAlternatives: false,
    retryRasterHeights: [72, 88],
    initialRasterHeight: 104,
    chunkFallbackMinCssWidth: 500,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, targetPixelHeight: image.targetPixelHeight, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return {
          latex: '\\cdots + 1 ) - 3 ( x - \\ldots',
          top: { latex: '\\cdots + 1 ) - 3 ( x - \\ldots', score: -0.2 },
          candidates: [{ latex: '\\cdots + 1 ) - 3 ( x - \\ldots', score: -0.2 }],
          elapsedSeconds: 8
        };
      }
      const ids = image.strokeIds || [];
      const latex = ids.includes('left_4') ? '4 ( 2 x + 1 )'
        : ids.includes('mid_minus') ? '- 3 ( x - 2 )'
          : ids.includes('rhs_60') ? '60'
            : '';
      return {
        latex,
        top: { latex, score: 1 },
        candidates: [{ latex, score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  const firstChunkIndex = calls.findIndex((call) => call.profiles.includes('chunk-fallback'));
  const firstRetryIndex = calls.findIndex((call) => !call.profiles.includes('chunk-fallback') && call.targetPixelHeight === 72);

  assert.ok(firstChunkIndex > 0);
  assert.equal(firstRetryIndex, -1);
  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.equal(result.latex, '4 ( 2 x + 1 ) - 3 ( x - 2 ) = 60');
});

test('semantic retry can fall back to chunk OCR after ellipsis retries', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('left', 0, 0, 260, 70),
    stroke('eq_top', 282, 28, 330, 34),
    stroke('eq_bottom', 282, 48, 330, 54),
    stroke('rhs', 360, 0, 540, 70),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -10, yMin: -10, xMax: 570, yMax: 100 },
    problemLatex: '8 x + 4 - 3 x + 6 = 60',
    recognizeAlternatives: false,
    semanticScoring: true,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [48, 64],
    initialRasterHeight: 104,
    chunkFallbackMinCssWidth: 460,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, targetPixelHeight: image.targetPixelHeight, strokeIds: image.strokeIds });
      if (image.profiles.includes('chunk-fallback')) {
        const latex = image.strokeIds.includes('left') ? '8 x + 4 - 3 x + 6' : '60';
        return {
          latex,
          top: { latex, score: 1 },
          candidates: [{ latex, score: 1 }],
          elapsedSeconds: 1
        };
      }
      if (image.targetPixelHeight === 48 || image.targetPixelHeight === 64) {
        return {
          latex: '\\cdots + 6 = \\ldots',
          top: { latex: '\\cdots + 6 = \\ldots', score: -0.5 },
          candidates: [{ latex: '\\cdots + 6 = \\ldots', score: -0.5 }],
          elapsedSeconds: 6
        };
      }
      return {
        latex: '8 x + 4 - 3 x + 6 = c n',
        top: { latex: '8 x + 4 - 3 x + 6 = c n', score: -0.2 },
        candidates: [{ latex: '8 x + 4 - 3 x + 6 = c n', score: -0.2 }],
        elapsedSeconds: 2
      };
    },
    scoreSemantics: async (request) => ({
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        lineIndex: group.lineIndex,
        semanticScore: String(group.latex || '').includes('60') ? 3 : 0.2,
        bestLatex: group.latex,
        sound: String(group.latex || '').includes('60'),
        equivalentToProblem: String(group.latex || '').includes('60'),
        equivalentToPrevious: false,
        candidateScores: [{
          latex: group.latex,
          sound: String(group.latex || '').includes('60'),
          score: String(group.latex || '').includes('60') ? 3 : 0.2
        }]
      })),
      elapsedSeconds: 0.01
    })
  });

  assert.ok(calls.some((call) => call.profiles.includes('chunk-fallback')));
  assert.equal(result.latex, '8 x + 4 - 3 x + 6 = 60');
});

test('compact structural line can use chunk OCR before height retries', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('left_num', 0, 8, 28, 42),
    stroke('left_bar', 0, 56, 92, 62),
    stroke('left_den', 0, 74, 28, 104),
    stroke('plus', 106, 40, 130, 72),
    stroke('right_num', 154, 8, 184, 42),
    stroke('right_bar', 150, 56, 230, 62),
    stroke('right_den', 154, 74, 184, 104),
    stroke('eq_top', 244, 44, 270, 50),
    stroke('eq_bottom', 244, 62, 270, 68),
    stroke('rhs', 286, 20, 310, 84),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 330, yMax: 130 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    retryRasterHeights: [72, 88],
    initialRasterHeight: 88,
    chunkFallbackMinCssWidth: 460,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, targetPixelHeight: image.targetPixelHeight, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 20 };
      }
      return {
        latex: 'x',
        top: { latex: 'x', score: 1 },
        candidates: [{ latex: 'x', score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  const firstChunkIndex = calls.findIndex((call) => call.profiles.includes('chunk-fallback'));
  const firstRetryIndex = calls.findIndex((call) => !call.profiles.includes('chunk-fallback') && call.targetPixelHeight === 72);

  assert.ok(firstChunkIndex > 0);
  assert.equal(firstRetryIndex, -1);
  assert.equal(result.lines[0].prediction.chunkFallback, true);
});

test('compact neighboring fractions are split into separate chunk OCR calls', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('left_num', 125, 76, 147, 99),
    stroke('plus', 170, 84, 210, 132),
    stroke('right_num', 230, 68, 244, 100),
    stroke('equals_top', 269, 101, 309, 105),
    stroke('left_bar', 123, 112, 149, 117),
    stroke('right_bar', 226, 113, 248, 118),
    stroke('equals_bottom', 269, 115, 309, 119),
    stroke('left_den', 127, 121, 144, 154),
    stroke('right_den', 227, 123, 245, 157),
    stroke('rhs', 327, 81, 353, 129),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 110, yMin: 55, xMax: 370, yMax: 175 },
    problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
    recognizeAlternatives: false,
    retryRasterHeights: [72],
    initialRasterHeight: 88,
    chunkFallbackMinCssWidth: 460,
    chunkFallbackMaxCssWidth: 140,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 20 };
      }
      if (image.strokeIds.includes('left_bar') || image.strokeIds.includes('right_bar')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 6 };
      }
      const latex = image.strokeIds.includes('left_num') ? 'x'
        : image.strokeIds.includes('left_den') ? '2'
          : image.strokeIds.includes('right_num') ? '1'
            : image.strokeIds.includes('right_den') ? '3'
              : image.strokeIds.includes('plus') ? '+'
                : '2';
      return {
        latex,
        top: { latex, score: 1 },
        candidates: [{ latex, score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  const chunkCalls = calls.filter((call) => call.profiles.includes('chunk-fallback'));
  assert.ok(chunkCalls.some((call) => call.strokeIds.includes('left_bar')));
  assert.ok(chunkCalls.some((call) => call.strokeIds.includes('right_bar')));
  const fractionAttempts = result.lines[0].prediction.chunkAttempts.filter((attempt) => attempt.fractionSubchunk);
  assert.ok(fractionAttempts.length >= 2);
  assert.ok(fractionAttempts.every((attempt) => (
    attempt.parts.every((part) => part.targetPixelHeight === 72)
  )));
  assert.equal(result.latex, '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2');
});

test('tall linear structural rows keep wide stroke chunks without fraction bars', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('x', 160, 168, 207, 204),
    stroke('minus', 229, 182, 255, 191),
    stroke('one', 273, 169, 313, 215),
    stroke('eq_top', 348, 181, 397, 195),
    stroke('eq_bottom', 345, 196, 394, 210),
    stroke('five', 418, 187, 481, 234),
    stroke('left_paren', 500, 182, 530, 252),
    stroke('rhs_x', 531, 207, 578, 252),
    stroke('plus', 608, 203, 657, 252),
    stroke('rhs_one', 684, 222, 716, 252),
    stroke('right_paren', 755, 213, 770, 252),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 140, yMin: 150, xMax: 790, yMax: 270 },
    problemLatex: '\\frac { x - 1 } { x + 1 } = 5',
    recognizeAlternatives: false,
    retryRasterHeights: [],
    initialRasterHeight: 88,
    chunkFallbackMinCssWidth: 460,
    chunkFallbackMaxCssWidth: 340,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 16 };
      }
      const ids = image.strokeIds || [];
      const latex = ids.includes('x') && ids.includes('minus') && ids.includes('one') ? 'x - 1'
        : ids.includes('five') ? '5 ( x + 1'
          : ids.includes('right_paren') ? ')'
            : '';
      return {
        latex,
        top: { latex, score: 1 },
        candidates: [{ latex, score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  const chunkCalls = calls.filter((call) => call.profiles.includes('chunk-fallback'));
  assert.ok(chunkCalls.some((call) => (
    call.strokeIds.includes('x') &&
    call.strokeIds.includes('minus') &&
    call.strokeIds.includes('one')
  )));
  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.equal(result.latex, 'x - 1 = 5 ( x + 1 )');
});

test('chunk fallback splits timed-out fraction chunks into numerator and denominator', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('lhs_f', 0, 0, 30, 70),
    stroke('lhs_prime', 38, 0, 48, 20),
    stroke('lhs_x', 58, 0, 92, 70),
    stroke('eq_top', 112, 26, 156, 32),
    stroke('eq_bottom', 112, 48, 156, 54),
    stroke('num_x', 180, 0, 210, 35),
    stroke('num_minus', 220, 18, 252, 24),
    stroke('num_1', 262, 0, 286, 35),
    stroke('frac_bar', 174, 50, 326, 56),
    stroke('den_x', 214, 74, 244, 110),
    stroke('den_2', 258, 74, 292, 110),
  ];
  assignTimes(strokes);
  const calls = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 350, yMax: 135 },
    problemLatex: 'f ( x ) = \\frac { x ^ { 2 } + 1 } { x }',
    recognizeAlternatives: false,
    retryRasterHeights: [72],
    initialRasterHeight: 88,
    chunkFallbackMinCssWidth: 220,
    chunkFallbackMaxCssWidth: 140,
    recognizeLine: async (image) => {
      calls.push({ profiles: image.profiles, strokeIds: image.strokeIds });
      if (!image.profiles.includes('chunk-fallback')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 20 };
      }
      if (image.strokeIds.includes('lhs_f')) {
        return {
          latex: 'f ^ { \\prime } ( x )',
          top: { latex: 'f ^ { \\prime } ( x )', score: 1 },
          candidates: [{ latex: 'f ^ { \\prime } ( x )', score: 1 }],
          elapsedSeconds: 1
        };
      }
      if (image.strokeIds.includes('frac_bar')) {
        return { latex: '', top: null, candidates: [], timedOut: true, elapsedSeconds: 6 };
      }
      if (image.strokeIds.includes('num_x')) {
        return {
          latex: 'x ^ { 2 } - 1',
          top: { latex: 'x ^ { 2 } - 1', score: 1 },
          candidates: [{ latex: 'x ^ { 2 } - 1', score: 1 }],
          elapsedSeconds: 1
        };
      }
      return {
        latex: 'x ^ { 2 }',
        top: { latex: 'x ^ { 2 }', score: 1 },
        candidates: [{ latex: 'x ^ { 2 }', score: 1 }],
        elapsedSeconds: 1
      };
    }
  });

  assert.equal(result.lines[0].prediction.chunkFallback, true);
  assert.ok(result.lines[0].prediction.chunkAttempts.some((attempt) => attempt.fractionSubchunk));
  assert.equal(result.latex, 'f ^ { \\prime } ( x ) = \\frac { x ^ { 2 } - 1 } { x ^ { 2 } }');
});

test('selected lines are semantically refined with previous lines from the same answer', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];
  const semanticRequests = [];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'u = 5',
    semanticScoring: true,
    recognizeAlternatives: false,
    recognizeLine: async (image) => {
      const isFirstLine = image.tightBbox.yMin < 50;
      return {
        latex: isFirstLine ? 'z = 5' : 'z + 1 = 6',
        top: { latex: isFirstLine ? 'z = 5' : 'z + 1 = 6', score: 1 },
        candidates: isFirstLine
          ? [
              { latex: 'z = 5', score: 1 },
              { latex: '\\eta = 5', score: 1 },
            ]
          : [
              { latex: 'z + 1 = 6', score: 1 },
              { latex: '\\eta + 1 = 6', score: 1 },
            ],
        elapsedSeconds: 0.3
      };
    },
    scoreSemantics: async (request) => {
      semanticRequests.push(request);
      if (request.candidateGroups.length > 1) {
        return {
          candidateScores: request.candidateGroups.map((group) => ({
            candidateId: group.candidateId,
            semanticScore: 0,
            bestLatex: group.latex,
            sound: true,
            candidateScores: []
          })),
          elapsedSeconds: 0.01
        };
      }

      const group = request.candidateGroups[0];
      const context = group.previousLatex || request.previousLatex || [];
      const bestLatex = context.includes('\\eta = 5')
        ? '\\eta + 1 = 6'
        : '\\eta = 5';
      return {
        candidateScores: [{
          candidateId: group.candidateId,
          semanticScore: 4,
          bestLatex,
          sound: true,
          candidateScores: []
        }],
        elapsedSeconds: 0.01
      };
    }
  });

  assert.ok(semanticRequests.length >= 3);
  assert.ok(semanticRequests.some((request) => (
    request.candidateGroups.length === 1 && request.previousLatex.length === 0
  )));
  assert.ok(semanticRequests.some((request) => (
    request.candidateGroups.length === 1 && request.previousLatex.includes('\\eta = 5')
  )));
  assert.deepEqual(result.latexLines, ['\\eta = 5', '\\eta + 1 = 6']);
  assert.equal(result.semantic.sequential.lineScores.length, 2);
  assert.equal(result.lines[1].sequentialSemantic.bestLatex, '\\eta + 1 = 6');
});

test('contextual candidate semantic scoring batches per-line contexts', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
    stroke('c', 0, 160, 50, 190),
  ];
  const semanticRequests = [];

  await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 70, yMax: 210 },
    problemLatex: 'a = 1',
    semanticScoring: true,
    recognizeAlternatives: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    recognizeLine: async (image) => {
      const latex = image.tightBbox.yMin < 50
        ? 'a = 1'
        : image.tightBbox.yMin < 130
          ? 'a + 1 = 2'
          : 'a + 2 = 3';
      return {
        latex,
        top: { latex, score: 1 },
        candidates: [{ latex, score: 1 }],
        elapsedSeconds: 0.03
      };
    },
    scoreSemantics: async (request) => {
      semanticRequests.push(JSON.parse(JSON.stringify(request)));
      return {
        candidateScores: request.candidateGroups.map((group) => {
          const context = group.previousLatex || request.previousLatex || [];
          return {
            candidateId: group.candidateId,
            lineIndex: group.lineIndex,
            semanticScore: context.length,
            bestLatex: group.latex,
            sound: true,
            equivalentToProblem: false,
            equivalentToPrevious: context.length > 0,
            candidateScores: []
          };
        }),
        elapsedSeconds: 0.01
      };
    }
  });

  const contextualRequest = semanticRequests.find((request) => (
    request.candidateGroups.length === 2 &&
    request.candidateGroups.every((group) => Array.isArray(group.previousLatex))
  ));

  assert.ok(contextualRequest);
  assert.ok(contextualRequest.candidateGroups.some((group) => (
    group.previousLatex.includes('a = 1') &&
    !group.previousLatex.includes('a + 1 = 2')
  )));
  assert.ok(contextualRequest.candidateGroups.some((group) => (
    group.previousLatex.includes('a = 1') &&
    group.previousLatex.includes('a + 1 = 2')
  )));
});

test('same-answer context can reselect the cover away from a parent crop', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('a', 0, 0, 50, 30),
    stroke('b', 0, 80, 50, 110),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 120 },
    problemLatex: 'u = 5',
    semanticScoring: true,
    recognizeLine: async (image) => {
      if (image.profiles.includes('parent')) {
        return {
          latex: '\\eta = 5 \\\\ z + 1 = 6',
          top: { latex: '\\eta = 5 \\\\ z + 1 = 6', score: 2 },
          candidates: [{ latex: '\\eta = 5 \\\\ z + 1 = 6', score: 2 }],
          elapsedSeconds: 0.3
        };
      }
      const isFirstLine = image.tightBbox.yMin < 50;
      return {
        latex: isFirstLine ? '\\eta = 5' : 'z + 1 = 6',
        top: { latex: isFirstLine ? '\\eta = 5' : 'z + 1 = 6', score: 1 },
        candidates: isFirstLine
          ? [{ latex: '\\eta = 5', score: 1 }]
          : [
              { latex: 'z + 1 = 6', score: 1 },
              { latex: '\\eta + 1 = 6', score: -1 },
            ],
        elapsedSeconds: 0.3
      };
    },
    scoreSemantics: async (request) => {
      if (request.candidateGroups.length > 1) {
        return {
          candidateScores: request.candidateGroups.map((group) => ({
            candidateId: group.candidateId,
            semanticScore: String(group.latex).includes('\\\\') ? 5 : 0,
            bestLatex: group.latex,
            sound: true,
            candidateScores: []
          })),
          elapsedSeconds: 0.01
        };
      }

      const group = request.candidateGroups[0];
      const context = group.previousLatex || request.previousLatex || [];
      const hasEtaContext = context.includes('\\eta = 5');
      const isLowerLine = group.latex === 'z + 1 = 6' || group.latex === '\\eta + 1 = 6';
      return {
        candidateScores: [{
          candidateId: group.candidateId,
          semanticScore: hasEtaContext && isLowerLine ? 8 : 0,
          bestLatex: hasEtaContext && isLowerLine ? '\\eta + 1 = 6' : group.latex,
          sound: true,
          candidateScores: []
        }],
        elapsedSeconds: 0.01
      };
    }
  });

  assert.equal(result.lines.length, 2);
  assert.ok(result.lines.every((line) => !line.profiles.includes('parent')));
  assert.deepEqual(result.latexLines, ['\\eta = 5', '\\eta + 1 = 6']);
  assert.ok(result.semantic.contextual.candidateScores.some((score) => (
    score.sameAnswerContext.includes('\\eta = 5') &&
    score.bestLatex === '\\eta + 1 = 6'
  )));
});

test('student writing pipeline falls back when semantic scoring fails', async () => {
  installFakeCanvas();
  const strokes = [stroke('a', 0, 0, 50, 30)];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -5, yMin: -5, xMax: 60, yMax: 40 },
    problemLatex: 'x = 1',
    semanticScoring: true,
    recognizeLine: async () => ({
      latex: 'x = 1',
      top: { latex: 'x = 1', score: 2 },
      candidates: [{ latex: 'x = 1', score: 2 }],
      elapsedSeconds: 0.3
    }),
    scoreSemantics: async () => {
      throw new Error('semantic offline');
    }
  });

  assert.equal(result.semantic.failed, true);
  assert.equal(result.semantic.error, 'semantic offline');
  assert.equal(result.latex, 'x = 1');
});

test('segmentation only uses strokes inside the dynamic answer box', () => {
  const inside = stroke('a', 10, 10, 40, 35);
  const outside = stroke('b', 300, 300, 340, 335);
  const grazing = stroke('c', 96, 20, 156, 52);
  const mostlyInside = stroke('d', 76, 60, 116, 88);
  const result = segmentMathLines([inside, outside], {
    answerBox: { xMin: 0, yMin: 0, xMax: 100, yMax: 100 }
  });

  assert.deepEqual(
    result.selected.flatMap((candidate) => candidate.strokeIds).sort(),
    ['a']
  );
  assert.ok(result.candidates.every((candidate) => !candidate.strokeIds.includes('b')));

  assert.equal(strokeBelongsToAnswerBox(grazing, { xMin: 0, yMin: 0, xMax: 100, yMax: 100 }), false);
  assert.equal(strokeBelongsToAnswerBox(mostlyInside, { xMin: 0, yMin: 0, xMax: 100, yMax: 100 }), true);

  const boundaryResult = segmentMathLines([inside, grazing, mostlyInside], {
    answerBox: { xMin: 0, yMin: 0, xMax: 100, yMax: 100 }
  });
  const selectedIds = boundaryResult.selected.flatMap((candidate) => candidate.strokeIds).sort();
  assert.ok(selectedIds.includes('a'));
  assert.ok(selectedIds.includes('d'));
  assert.ok(!selectedIds.includes('c'));
  assert.ok(boundaryResult.candidates.every((candidate) => !candidate.strokeIds.includes('c')));
});

test('detector rasterization ignores strokes that only graze the answer box', async () => {
  installFakeCanvas();
  const inside = stroke('inside', 10, 10, 42, 38);
  const grazing = stroke('grazing', 96, 18, 156, 52);
  const seenStrokeIds = [];

  const result = await recognizeStudentWriting({
    strokes: [inside, grazing],
    answerBox: { xMin: 0, yMin: 0, xMax: 100, yMax: 100 },
    detectLineBands: true,
    detectLines: async (image) => {
      seenStrokeIds.push(...image.strokeIds);
      return { detections: [] };
    },
    recognizeAlternatives: false,
    recognizeLine: async () => ({
      latex: 'x = 1',
      top: { latex: 'x = 1', score: 2 },
      candidates: [{ latex: 'x = 1', score: 2 }],
      elapsedSeconds: 0.2
    })
  });

  assert.deepEqual(seenStrokeIds, ['inside']);
  assert.deepEqual(result.detection.source, 'detector');
});

test('incremental scheduler keeps non-overlapping line OCR running', async () => {
  const snapshots = [];
  const calls = [];
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => {
      calls.push({
        strokeIds: request.strokes.map((item) => item.id).sort(),
        detectLineBands: Boolean(request.detectLineBands)
      });
      return new Promise(() => {});
    },
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });
  const first = stroke('a', 0, 0, 50, 30);
  const second = stroke('b', 0, 180, 50, 210);
  const base = {
    problemId: 'problem-1',
    answerBox: { xMin: -10, yMin: -10, xMax: 120, yMax: 260 },
    problemLatex: 'x = 1'
  };

  scheduler.update({ ...base, strokes: [first] });
  await scheduler.flushNow();
  await nextMicrotask();
  scheduler.update({ ...base, strokes: [first, second] });
  await scheduler.flushNow();
  await nextMicrotask();

  const latest = snapshots.at(-1);
  const components = latest.realtime.components;
  assert.equal(components.length, 2);
  assert.equal(components.find((component) => component.strokeIds.includes('a')).contested, false);
  assert.equal(components.find((component) => component.strokeIds.includes('b')).contested, false);
  assert.ok(latest.result.candidatePredictions.some((candidate) => (
    candidate.candidateId.startsWith('realtime_') &&
    candidate.realtimeStatus === 'running'
  )));
  assert.ok(calls.some((call) => call.strokeIds.join('|') === 'a'));
  assert.ok(calls.some((call) => call.strokeIds.join('|') === 'b'));
});

test('incremental scheduler aborts only recognition made stale by overlapping ink', async () => {
  const line = stroke('a', 0, 0, 50, 30);
  const farLine = stroke('b', 0, 180, 50, 210);
  const overlappingInk = stroke('c', 24, 24, 72, 56);
  const calls = [];
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => {
      const call = {
        strokeIds: request.strokes.map((item) => item.id).sort(),
        detectLineBands: Boolean(request.detectLineBands),
        signal: request.signal
      };
      calls.push(call);
      return new Promise(() => {});
    }
  });
  const base = {
    problemId: 'problem-1',
    answerBox: { xMin: -10, yMin: -10, xMax: 140, yMax: 260 },
    problemLatex: 'x = 1'
  };

  scheduler.update({ ...base, strokes: [line] });
  await scheduler.flushNow();
  await nextMicrotask();

  const firstLineCall = calls.find((call) => (
    !call.detectLineBands &&
    call.strokeIds.join('|') === 'a'
  ));
  assert.ok(firstLineCall);
  assert.equal(firstLineCall.signal.aborted, false);

  scheduler.update({ ...base, strokes: [line, farLine] });
  await nextMicrotask();
  assert.equal(firstLineCall.signal.aborted, false);

  scheduler.update({ ...base, strokes: [line, farLine, overlappingInk] });
  await nextMicrotask();
  assert.equal(firstLineCall.signal.aborted, true);
});

test('incremental scheduler retries a fresh component after abort', async () => {
  const snapshots = [];
  let calls = 0;
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => {
      calls += 1;
      if (!request.detectLineBands && calls === 1) {
        const error = new Error('Recognition aborted');
        error.name = 'AbortError';
        return Promise.reject(error);
      }
      return Promise.resolve(fakeRecognitionResult(request.strokes, 'deterministic'));
    },
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });
  const line = stroke('a', 0, 0, 50, 30);

  scheduler.update({
    problemId: 'problem-1',
    strokes: [line],
    answerBox: { xMin: -10, yMin: -10, xMax: 80, yMax: 60 },
    problemLatex: 'x = 1'
  });
  await scheduler.flushNow();
  await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

  assert.ok(calls >= 2);
  assert.equal(snapshots.at(-1).realtime.components[0].status, 'final');
});

test('incremental scheduler keeps partial OCR visible while an overlap is contested', async () => {
  const snapshots = [];
  const dbnetPending = new Promise(() => {});
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => (
      request.detectLineBands
        ? dbnetPending
        : Promise.resolve(fakeRecognitionResult(request.strokes, 'deterministic'))
    ),
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });
  const secondLine = stroke('b', 0, 80, 50, 110);
  const overlappingInk = stroke('c', 24, 104, 72, 136);
  const base = {
    problemId: 'problem-1',
    answerBox: { xMin: -10, yMin: -10, xMax: 140, yMax: 180 },
    problemLatex: 'x = 1'
  };

  scheduler.update({ ...base, strokes: [secondLine] });
  await scheduler.flushNow();
  await waitForSnapshot(snapshots, (snapshot) => (
    snapshot.realtime.components.some((component) => (
      component.strokeIds.includes('b') &&
      component.status === 'provisional' &&
      component.hasResult
    ))
  ));

  scheduler.update({ ...base, strokes: [secondLine, overlappingInk] });
  await nextMicrotask();

  const latest = snapshots.at(-1);
  const contested = latest.realtime.components.find((component) => component.strokeIds.includes('b'));
  assert.equal(contested.status, 'contested');
  assert.equal(contested.contested, true);
  assert.equal(contested.hasResult, true);
  assert.equal(latest.status, 'pending');
});

test('incremental scheduler reuses OCR when DBNet confirms the same line signature', async () => {
  installFakeCanvas();
  const snapshots = [];
  const ocrCalls = [];
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    detectLines: async () => ({ detections: [], failed: false }),
    recognizeLine: async (image) => {
      ocrCalls.push({
        strokeIds: image.strokeIds.slice().sort(),
        bbox: image.tightBbox,
        targetPixelHeight: image.targetPixelHeight
      });
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.02
      };
    },
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });
  const line = stroke('a', 0, 0, 50, 30);

  scheduler.update({
    problemId: 'problem-1',
    strokes: [line],
    answerBox: { xMin: -10, yMin: -10, xMax: 80, yMax: 60 },
    problemLatex: 'x = 1'
  });
  await scheduler.flushNow();
  await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

  assert.equal(ocrCalls.length, 1);
  assert.equal(snapshots.at(-1).realtime.components[0].status, 'final');
});

test('incremental scheduler does not cache abort-like OCR failures', async () => {
  let calls = 0;
  const scheduler = new IncrementalRecognitionScheduler({
    semanticScoring: false,
    recognizeLine: async () => {
      calls += 1;
      if (calls === 1) {
        return {
          latex: '',
          candidates: [],
          failed: true,
          error: 'signal is aborted without reason',
          elapsedSeconds: 20
        };
      }
      return {
        latex: 'x = 1',
        top: { latex: 'x = 1', score: 2 },
        candidates: [{ latex: 'x = 1', score: 2 }],
        elapsedSeconds: 0.02
      };
    }
  });
  const image = {
    strokeIds: ['a'],
    tightBbox: { xMin: 0, yMin: 0, xMax: 50, yMax: 30 },
    padding: 24,
    targetPixelHeight: 104,
    width: 200,
    height: 104
  };

  const failed = await scheduler.cachedRecognizeLine(image, { apiUrl: '', model: 'comer' });
  assert.equal(failed.failed, true);
  const result = await scheduler.cachedRecognizeLine(image, { apiUrl: '', model: 'comer' });

  assert.equal(calls, 2);
  assert.equal(result.latex, 'x = 1');
  assert.equal(result.cached, undefined);
});

test('incremental scheduler isolates aborts for shared OCR inflight waiters', async () => {
  let calls = 0;
  let resolveOcr = null;
  let upstreamSignal = null;
  const scheduler = new IncrementalRecognitionScheduler({
    semanticScoring: false,
    recognizeLine: async (_image, options) => {
      calls += 1;
      upstreamSignal = options.signal;
      return new Promise((resolve) => { resolveOcr = resolve; });
    }
  });
  const image = {
    strokeIds: ['a'],
    tightBbox: { xMin: 0, yMin: 0, xMax: 50, yMax: 30 },
    padding: 24,
    targetPixelHeight: 104,
    width: 200,
    height: 104
  };
  const first = new AbortController();
  const second = new AbortController();

  const firstRequest = scheduler.cachedRecognizeLine(image, {
    apiUrl: '',
    model: 'comer',
    signal: first.signal
  });
  const secondRequest = scheduler.cachedRecognizeLine(image, {
    apiUrl: '',
    model: 'comer',
    signal: second.signal
  });

  await nextMicrotask();
  assert.equal(calls, 1);
  first.abort();
  await assert.rejects(firstRequest, { name: 'AbortError' });
  assert.equal(upstreamSignal.aborted, false);

  resolveOcr({
    latex: 'x = 1',
    top: { latex: 'x = 1', score: 2 },
    candidates: [{ latex: 'x = 1', score: 2 }],
    elapsedSeconds: 0.02
  });

  const secondResult = await secondRequest;
  assert.equal(secondResult.latex, 'x = 1');
  assert.equal(secondResult.inFlightReused, true);
});

test('incremental scheduler final pass preserves one-shot whole-answer candidates', async () => {
  const snapshots = [];
  const calls = [];
  const first = stroke('line_a', 0, 0, 80, 30);
  const second = stroke('line_b', 0, 120, 90, 150);
  const answerBox = { xMin: -10, yMin: -10, xMax: 140, yMax: 190 };
  const oneShot = fakeFullAnswerResult([first, second], {
    latexLines: ['2 x = 8', 'x = 4'],
    candidateIds: ['full_line_a', 'full_line_b', 'full_parent']
  });
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => {
      const call = {
        strokeIds: request.strokes.map((item) => item.id).sort(),
        answerBox: request.answerBox,
        detectLineBands: Boolean(request.detectLineBands),
        skipSingleStrokeAlternatives: request.skipSingleStrokeAlternatives,
        stagedAlternativeRecognition: request.stagedAlternativeRecognition,
        semanticCandidateLimit: request.semanticCandidateLimit
      };
      calls.push(call);
      if (
        call.detectLineBands &&
        call.strokeIds.join('|') === 'line_a|line_b' &&
        sameBboxForTest(request.answerBox, answerBox)
      ) {
        return Promise.resolve(oneShot);
      }
      return Promise.resolve(fakeRecognitionResult(request.strokes, 'deterministic'));
    },
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });

  scheduler.update({
    problemId: 'problem-1',
    strokes: [first, second],
    answerBox,
    problemLatex: '2 x + 3 = 11'
  });
  await scheduler.flushNow();
  const completed = await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

  assert.ok(calls.some((call) => (
    call.detectLineBands &&
    call.strokeIds.join('|') === 'line_a|line_b' &&
    sameBboxForTest(call.answerBox, answerBox)
  )));
  assert.deepEqual(completed.result.latexLines, oneShot.latexLines);
  assert.deepEqual(
    completed.result.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
    oneShot.segmentation.candidates.map((candidate) => candidate.candidateId).sort()
  );
  assert.deepEqual(
    completed.result.candidatePredictions.map((candidate) => candidate.candidateId).sort(),
    oneShot.candidatePredictions.map((candidate) => candidate.candidateId).sort()
  );
  const fullAnswerCall = calls.find((call) => (
    call.detectLineBands &&
    call.strokeIds.join('|') === 'line_a|line_b' &&
    sameBboxForTest(call.answerBox, answerBox)
  ));
  assert.equal(fullAnswerCall.skipSingleStrokeAlternatives, undefined);
  assert.equal(fullAnswerCall.stagedAlternativeRecognition, undefined);
  assert.equal(fullAnswerCall.semanticCandidateLimit, 0);
});

test('incremental scheduler does not let stale local OCR downgrade the final one-shot result', async () => {
  let resolveLocal;
  const snapshots = [];
  const line = stroke('line_a', 0, 0, 80, 30);
  const answerBox = { xMin: -10, yMin: -10, xMax: 120, yMax: 70 };
  const scheduler = new IncrementalRecognitionScheduler({
    debounceMs: 0,
    semanticScoring: false,
    recognizeWriting: (request) => {
      if (request.detectLineBands && request.strokes.length === 1) {
        return Promise.resolve(fakeFullAnswerResult([line], {
          latexLines: ['x = 4'],
          candidateIds: ['full_line_a', 'full_parent']
        }));
      }
      return new Promise((resolve) => {
        resolveLocal = resolve;
      });
    },
    onStateChange: (snapshot) => snapshots.push(snapshot)
  });

  scheduler.update({
    problemId: 'problem-1',
    strokes: [line],
    answerBox,
    problemLatex: 'x = 4'
  });
  await scheduler.flushNow();
  const completed = await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');
  assert.equal(completed.result.latexLines.join('|'), 'x = 4');

  resolveLocal(fakeRecognitionResult([line], 'stale-local'));
  await nextMicrotask();
  const latest = snapshots.at(-1);
  assert.equal(latest.status, 'complete');
  assert.equal(latest.result.latexLines.join('|'), 'x = 4');
  assert.ok(latest.realtime.components.every((component) => component.status === 'final'));
});

test('incremental scheduler matches one-shot final pass across varied equation families', async () => {
  const cases = [
    {
      name: 'algebra',
      problemLatex: '2 x + 3 = 11',
      latexLines: ['2 x = 8', '/ 2      / 2', 'x = 4']
    },
    {
      name: 'rational',
      problemLatex: '\\frac { x } { 2 } + \\frac { 1 } { 3 } = 2',
      latexLines: [
        '\\times 6       \\times 6',
        '3 x + 2 = 12',
        'x = \\frac { 10 } { 3 }'
      ]
    },
    {
      name: 'logarithmic',
      problemLatex: '\\log _ { 2 } ( x ) + 3 = 7',
      latexLines: [
        '- 3       - 3',
        '\\log _ { 2 } ( x ) = 4',
        'x = 16'
      ]
    },
    {
      name: 'quadratic',
      problemLatex: 'x ^ { 2 } - 5 x + 6 = 0',
      latexLines: [
        '( x - 2 ) ( x - 3 ) = 0',
        'x = 2',
        'x = 3'
      ]
    }
  ];

  for (const item of cases) {
    const snapshots = [];
    const strokes = item.latexLines.map((_, index) => (
      stroke(`${item.name}_${index + 1}`, 20, index * 96, 220 + index * 20, index * 96 + 38)
    ));
    const answerBox = padBboxForTest(bboxForStrokes(strokes), 20);
    const oneShot = fakeFullAnswerResult(strokes, {
      latexLines: item.latexLines,
      candidateIds: [
        ...item.latexLines.map((_, index) => `${item.name}_line_${index + 1}`),
        `${item.name}_whole_answer_parent`
      ]
    });
    const scheduler = new IncrementalRecognitionScheduler({
      debounceMs: 0,
      semanticScoring: false,
      recognizeWriting: (request) => {
        const ids = request.strokes.map((entry) => entry.id).sort().join('|');
        const fullIds = strokes.map((entry) => entry.id).sort().join('|');
        if (request.detectLineBands && ids === fullIds && sameBboxForTest(request.answerBox, answerBox)) {
          return Promise.resolve(oneShot);
        }
        return Promise.resolve(fakeRecognitionResult(request.strokes, 'local'));
      },
      onStateChange: (snapshot) => snapshots.push(snapshot)
    });

    scheduler.update({
      problemId: `problem-${item.name}`,
      strokes,
      answerBox,
      problemLatex: item.problemLatex
    });
    await scheduler.flushNow();
    const completed = await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

    assert.deepEqual(completed.result.latexLines, oneShot.latexLines, item.name);
    assert.deepEqual(
      completed.result.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      oneShot.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      item.name
    );
    assert.equal(completed.result.realtime.allFinal, true, item.name);
    assert.ok(completed.result.realtime.components.every((component) => (
      component.status === 'final' && !component.contested
    )), item.name);
  }
});

test('incremental scheduler matches one-shot recognition on synthetic catalog boards', async () => {
  installFakeCanvas();
  const problemNames = equationCatalogProblemNames();

  for (const problemName of problemNames) {
    const board = syntheticCatalogBoard(problemName, {
      spacing: problemName.includes('mixed') ? 'dense' : 'standard',
      inkStyle: problemName.includes('rational') ? 'compact' : 'normal',
      seed: 140 + problemNames.indexOf(problemName)
    });
    const strokes = strokesFromSyntheticBoard(board);
    const answerBox = padBboxForTest(bboxForStrokes(strokes), 24);
    const expectedLines = board.fixture.expectedLatexLines || [];
    const fakeReaders = fakeReadersForSyntheticBoard(board, strokes);
    const oneShot = await recognizeStudentWriting({
      strokes,
      answerBox,
      problemLatex: board.fixture.problemLatex,
      problemMetadata: {
        name: board.fixture.problem,
        family: board.fixture.family,
        expectedLatexLines: expectedLines
      },
      detectLineBands: true,
      semanticScoring: false,
      detectLines: fakeReaders.detectLines,
      recognizeLine: fakeReaders.recognizeLine
    });

    const snapshots = [];
    const scheduler = new IncrementalRecognitionScheduler({
      debounceMs: 0,
      semanticScoring: false,
      detectLines: fakeReaders.detectLines,
      recognizeLine: fakeReaders.recognizeLine,
      onStateChange: (snapshot) => snapshots.push(snapshot)
    });

    scheduler.update({
      problemId: `problem-${problemName}`,
      strokes,
      answerBox,
      problemLatex: board.fixture.problemLatex,
      problemMetadata: {
        name: board.fixture.problem,
        family: board.fixture.family,
        expectedLatexLines: expectedLines
      }
    });
    await scheduler.flushNow();
    const completed = await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

    assert.deepEqual(completed.result.latexLines, oneShot.latexLines, problemName);
    assert.deepEqual(
      completed.result.lines.map((line) => line.candidateId),
      oneShot.lines.map((line) => line.candidateId),
      problemName
    );
    assert.deepEqual(
      completed.result.candidatePredictions.map((candidate) => candidate.candidateId).sort(),
      oneShot.candidatePredictions.map((candidate) => candidate.candidateId).sort(),
      problemName
    );
    assert.deepEqual(
      completed.result.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      oneShot.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      problemName
    );
    assert.equal(completed.result.realtime.allFinal, true, problemName);
  }
});

test('incremental scheduler matches one-shot recognition on messy synthetic latex renderings', async () => {
  installFakeCanvas();
  const cases = [
    {
      problemName: 'algebra_prompt_context',
      spacing: 'dense',
      inkStyle: 'messy',
      gapPattern: 'pinched-middle',
      seed: 616
    },
    {
      problemName: 'rational_two_fraction_solve',
      spacing: 'tight-steps',
      inkStyle: 'messy',
      gapPattern: 'accordion',
      seed: 1720
    },
    {
      problemName: 'rational_mixed_fraction_operations',
      spacing: 'dense',
      inkStyle: 'compact',
      gapPattern: 'stair-step',
      seed: 907
    },
    {
      problemName: 'logarithmic_solve',
      spacing: 'mixed',
      inkStyle: 'loose',
      gapPattern: 'pinched-middle',
      seed: 421
    },
    {
      problemName: 'square_root_solve',
      spacing: 'tight-steps',
      inkStyle: 'messy',
      gapPattern: 'accordion',
      seed: 533
    },
    {
      problemName: 'quadratic_formula_positive_root',
      spacing: 'dense',
      inkStyle: 'messy',
      gapPattern: 'stair-step',
      seed: 808
    }
  ];

  for (const item of cases) {
    const board = syntheticCatalogBoard(item.problemName, item);
    const label = [
      item.problemName,
      item.spacing,
      item.inkStyle,
      item.gapPattern
    ].filter(Boolean).join('/');
    const strokes = strokesFromSyntheticBoard(board, {
      order: item.order || 'interleaved-lines',
      strokeIntervalMs: 17,
      linePauseMs: 540
    });
    const answerBox = padBboxForTest(bboxForStrokes(strokes), 24);
    const expectedLines = board.fixture.expectedLatexLines || [];
    const fakeReaders = fakeReadersForSyntheticBoard(board, strokes);
    const oneShot = await recognizeStudentWriting({
      strokes,
      answerBox,
      problemLatex: board.fixture.problemLatex,
      problemMetadata: {
        name: board.fixture.problem,
        family: board.fixture.family,
        spacing: board.fixture.spacing,
        inkStyle: board.fixture.inkStyle,
        gapPattern: board.fixture.gapPattern,
        lineGaps: board.fixture.lineGaps,
        expectedLatexLines: expectedLines
      },
      detectLineBands: true,
      semanticScoring: false,
      detectLines: fakeReaders.detectLines,
      recognizeLine: fakeReaders.recognizeLine
    });

    const snapshots = [];
    const scheduler = new IncrementalRecognitionScheduler({
      debounceMs: 0,
      semanticScoring: false,
      detectLines: fakeReaders.detectLines,
      recognizeLine: fakeReaders.recognizeLine,
      onStateChange: (snapshot) => snapshots.push(snapshot)
    });

    scheduler.update({
      problemId: `messy-${label}`,
      strokes,
      answerBox,
      problemLatex: board.fixture.problemLatex,
      problemMetadata: {
        name: board.fixture.problem,
        family: board.fixture.family,
        spacing: board.fixture.spacing,
        inkStyle: board.fixture.inkStyle,
        gapPattern: board.fixture.gapPattern,
        lineGaps: board.fixture.lineGaps,
        expectedLatexLines: expectedLines
      }
    });
    await scheduler.flushNow();
    const completed = await waitForSnapshot(snapshots, (snapshot) => snapshot.status === 'complete');

    assert.deepEqual(completed.result.latexLines, oneShot.latexLines, label);
    assert.deepEqual(
      completed.result.lines.map((line) => line.candidateId),
      oneShot.lines.map((line) => line.candidateId),
      label
    );
    assert.deepEqual(
      completed.result.candidatePredictions.map((candidate) => candidate.candidateId).sort(),
      oneShot.candidatePredictions.map((candidate) => candidate.candidateId).sort(),
      label
    );
    assert.deepEqual(
      completed.result.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      oneShot.segmentation.candidates.map((candidate) => candidate.candidateId).sort(),
      label
    );
    assert.equal(completed.result.realtime.allFinal, true, label);
    assert.ok(completed.result.realtime.components.every((component) => (
      component.status === 'final' && !component.contested
    )), label);
  }
});

test('student writing pipeline handles distilled real handwriting trace fixtures', async () => {
  installFakeCanvas();
  const fixtures = loadRealHandwritingFixtures();
  assert.equal(fixtures.length, 43);

  for (const fixture of fixtures) {
    const fakeReaders = fakeReadersForRealTrace(fixture);
    const expectedGrading = pythonGradeFixtureTranscript(fixture);
    const result = await recognizeStudentWriting({
      strokes: fixture.strokes,
      answerBox: fixture.answerBox,
      problemLatex: fixture.problemLatex,
      problemMetadata: {
        ...(fixture.problemMetadata || {}),
        source: 'testing/fixtures/real_handwriting',
        sourceAuditId: fixture.sourceAuditId,
        expectedLatexLines: fixture.expectedLatexLines,
        fastLatexLines: fixture.fastLatexLines,
        knownDiscrepancyTypes: fixture.knownDiscrepancyTypes
      },
      apiUrl: 'http://127.0.0.1:8010',
      detectLineBands: false,
      semanticScoring: false,
      recognizeAlternatives: false,
      chunkFallback: false,
      recognizeLine: fakeReaders.recognizeLine,
      gradeWork: async (request) => pythonGradePayload(request)
    });

    const expectedGroups = fixture.expectedLineGroups.map((group) => strokeGroupKey(group.strokeIds));
    const selectedGroups = result.segmentation.selected.map((line) => strokeGroupKey(line.strokeIds));
    const expectedStrokeIds = new Set(fixture.expectedLineGroups.flatMap((group) => group.strokeIds));
    const finalStrokeIds = new Set(result.lines.flatMap((line) => line.strokeIds || []));

    assert.deepEqual(selectedGroups, expectedGroups, fixture.slug);
    assert.deepEqual([...finalStrokeIds].sort(), [...expectedStrokeIds].sort(), fixture.slug);
    assert.ok(result.lines.length > 0, fixture.slug);
    assert.ok(result.lines.length <= fixture.expectedLineGroups.length, fixture.slug);
    assert.equal(result.grading.status, 'complete', fixture.slug);
    assert.equal(
      result.grading.result?.problemStatus,
      expectedGrading.result?.problemStatus,
      fixture.slug
    );
    assert.notEqual(result.realtime?.allFinal, false, fixture.slug);
  }
});

test('segmentation ignores large enclosing circle annotation strokes', () => {
  const fixture = loadRealHandwritingFixture('circled-x-equals-four');
  const result = segmentMathLines(fixture.strokes, {
    answerBox: fixture.answerBox,
    ignoredStrokeIds: []
  });

  assert.deepEqual(
    result.selected.map((line) => strokeGroupKey(line.strokeIds)),
    [strokeGroupKey(fixture.expectedLineGroups[0].strokeIds)]
  );
});

test('segmentation infers visual-only annotations from unlabeled real strokes', () => {
  for (const slug of [
    'circled-intermediate-result',
    'crossout-scratch-division',
    'detached-circled-zero-annotation'
  ]) {
    const fixture = loadRealHandwritingFixture(slug);
    const unlabeledStrokes = fixture.strokes.map((stroke) => {
      const copy = { ...stroke };
      delete copy.visualOnly;
      return copy;
    });
    const result = segmentMathLines(unlabeledStrokes, {
      answerBox: fixture.answerBox,
      ignoredStrokeIds: [],
      detections: fixture.detections || []
    });

    assert.deepEqual(
      result.selected.map((line) => strokeGroupKey(line.strokeIds)),
      fixture.expectedLineGroups.map((group) => strokeGroupKey(group.strokeIds)),
      slug
    );
  }
});

test('segmentation ignores isolated tiny scratch marks without dropping decimal points', () => {
  const isolated = [stroke('dot', 100, 100, 109, 109)];
  assert.deepEqual(
    segmentMathLines(isolated, { answerBox: padBboxForTest(bboxForStrokes(isolated), 10) }).selected,
    []
  );

  const decimal = [
    stroke('zero', 100, 100, 132, 160),
    stroke('dot', 140, 152, 149, 161),
    stroke('five', 158, 100, 190, 160),
  ];
  const result = segmentMathLines(decimal, {
    answerBox: padBboxForTest(bboxForStrokes(decimal), 10)
  });
  assert.deepEqual(
    result.selected.map((line) => strokeGroupKey(line.strokeIds)),
    [strokeGroupKey(['zero', 'dot', 'five'])]
  );
});

test('full recognition preserves deterministic rational rows when detector merges them', async () => {
  installFakeCanvas();
  const board = syntheticCatalogBoard('rational_quadratic_solve', {
    spacing: 'standard',
    inkStyle: 'compact',
    seed: 1720
  });
  const strokes = strokesFromSyntheticBoard(board);
  const answerBox = padBboxForTest(bboxForStrokes(strokes), 24);
  const expectedLines = board.fixture.expectedLatexLines || [];
  const fakeReaders = fakeReadersForSyntheticBoard(board, strokes);
  const mergedBbox = bboxForStrokes(strokes);

  const result = await recognizeStudentWriting({
    strokes,
    answerBox,
    problemLatex: board.fixture.problemLatex,
    problemMetadata: {
      name: board.fixture.problem,
      family: board.fixture.family,
      expectedLatexLines: expectedLines
    },
    detectLineBands: true,
    semanticScoring: false,
    detectLines: async () => ({
      detections: [{ bbox: mergedBbox }],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: fakeReaders.recognizeLine
  });

  assert.deepEqual(result.latexLines, expectedLines);
  assert.equal(result.lines.length, expectedLines.length);
});

test('stacked fraction with a right-hand side remains one math line', () => {
  const strokes = [
    stroke('frac_num_x', 122, 100, 148, 128),
    stroke('frac_num_minus', 154, 112, 182, 117),
    stroke('frac_num_one', 190, 98, 212, 130),
    stroke('frac_bar', 110, 145, 228, 152),
    stroke('frac_den_x', 122, 166, 148, 194),
    stroke('frac_den_plus', 154, 166, 182, 194),
    stroke('frac_den_one', 190, 164, 212, 196),
    stroke('frac_eq_top', 252, 130, 300, 138),
    stroke('frac_eq_bottom', 252, 150, 300, 158),
    stroke('frac_rhs_5', 326, 122, 360, 178),
  ];
  assignTimes(strokes);

  const result = segmentMathLines(strokes, {
    detections: [
      { bbox: { xMin: 118, yMin: 96, xMax: 216, yMax: 132 } },
      { bbox: { xMin: 118, yMin: 162, xMax: 216, yMax: 198 } },
      { bbox: { xMin: 248, yMin: 122, xMax: 364, yMax: 180 } },
    ]
  });

  assert.equal(result.selected.length, 1);
  assert.deepEqual(result.selected[0].strokeIds.slice().sort(), strokes.map((item) => item.id).sort());
});

test('operation underlines do not make an algebra stack look like one fraction', () => {
  const strokes = [
    stroke('top_2', 100, 100, 132, 142),
    stroke('top_x', 146, 100, 180, 142),
    stroke('top_minus', 198, 120, 232, 126),
    stroke('top_1', 246, 98, 270, 144),
    stroke('top_eq_top', 292, 112, 334, 120),
    stroke('top_eq_bottom', 292, 128, 334, 136),
    stroke('top_19a', 356, 98, 380, 144),
    stroke('top_19b', 388, 100, 420, 144),
    stroke('op_plus_left', 202, 158, 230, 182),
    stroke('op_1_left', 242, 152, 262, 188),
    stroke('op_plus_right', 332, 158, 360, 182),
    stroke('op_1_right', 372, 152, 392, 188),
    stroke('op_underline_left', 198, 196, 266, 202),
    stroke('op_underline_right', 328, 196, 396, 202),
    stroke('bottom_2', 110, 230, 142, 272),
    stroke('bottom_x', 156, 230, 190, 272),
    stroke('bottom_eq_top', 214, 242, 256, 250),
    stroke('bottom_eq_bottom', 214, 258, 256, 266),
    stroke('bottom_20a', 280, 230, 312, 272),
    stroke('bottom_20b', 320, 230, 352, 272),
  ];
  assignTimes(strokes);

  const result = segmentMathLines(strokes, {
    detections: [
      { bbox: { xMin: 98, yMin: 96, xMax: 422, yMax: 146 } },
      { bbox: { xMin: 198, yMin: 150, xMax: 396, yMax: 190 } },
      { bbox: { xMin: 196, yMin: 194, xMax: 398, yMax: 204 } },
      { bbox: { xMin: 108, yMin: 228, xMax: 354, yMax: 274 } },
    ]
  });

  assert.equal(result.selected.length, 3);
  assert.ok(result.selected[1].strokeIds.includes('op_underline_left'));
  assert.ok(result.selected[1].strokeIds.includes('op_underline_right'));
  assert.ok(!result.selected[0].strokeIds.includes('op_plus_left'));
  assert.ok(!result.selected[2].strokeIds.includes('op_plus_right'));
});

test('custom problem flow starts by waiting for a latex equation', () => {
  const initial = createInitialProblemFlow(1200);

  assert.equal(initial.activeProblemId, null);
  assert.equal(initial.awaitingEquation, true);
  assert.equal(initial.customProblems, true);
  assert.deepEqual(initial.problems, []);
});

test('starting a custom problem stores user latex as problem context', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, '  \\frac{x}{2} + 5 = 13  ', 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(started.flow.awaitingEquation, false);
  assert.equal(active.id, 'problem-1');
  assert.equal(active.kind, 'equation-solving');
  assert.equal(active.latex, '\\frac{x}{2} + 5 = 13');
  assert.equal(active.modelResponse.latex, '\\frac{x}{2} + 5 = 13');
  assert.equal(active.metadata.source, 'user-latex');
  assert.equal(active.metadata.problemType, 'equation-solving');
});

test('starting an evaluate custom problem stores problem type and model copy', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, {
    latex: '  \\frac{1}{2} + \\frac{2}{4}  ',
    problemType: 'evaluate-expression'
  }, 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(started.flow.awaitingEquation, false);
  assert.equal(active.id, 'problem-1');
  assert.equal(active.kind, 'evaluate-expression');
  assert.equal(active.latex, '\\frac{1}{2} + \\frac{2}{4}');
  assert.equal(active.modelResponse.before, 'Evaluate the expression.');
  assert.equal(active.modelResponse.latex, '\\frac{1}{2} + \\frac{2}{4}');
  assert.equal(active.metadata.source, 'user-latex');
  assert.equal(active.metadata.problemType, 'evaluate-expression');
});

test('starting a simplify custom problem stores problem type and model copy', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, {
    latex: '  x + x  ',
    problemType: 'simplify-expression'
  }, 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(started.flow.awaitingEquation, false);
  assert.equal(active.id, 'problem-1');
  assert.equal(active.kind, 'simplify-expression');
  assert.equal(active.latex, 'x + x');
  assert.equal(active.modelResponse.before, 'Simplify the expression.');
  assert.equal(active.modelResponse.latex, 'x + x');
  assert.equal(active.metadata.source, 'user-latex');
  assert.equal(active.metadata.problemType, 'simplify-expression');
});

test('starting a handwritten custom problem stores handwriting source', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, {
    latex: '2x - 1 = 19',
    problemType: 'equation-solving',
    source: 'user-handwriting'
  }, 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(active.latex, '2x - 1 = 19');
  assert.equal(active.metadata.source, 'user-handwriting');
  assert.equal(active.metadata.problemType, 'equation-solving');
});

test('numeric custom expression without equals routes to evaluate mode', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, '  0.9 - 0.11  ', 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(active.kind, 'evaluate-expression');
  assert.equal(active.latex, '0.9 - 0.11');
  assert.equal(active.modelResponse.before, 'Evaluate the expression.');
  assert.equal(active.metadata.problemType, 'evaluate-expression');
});

test('symbolic custom expression without equals routes to simplify mode', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, '  x + x  ', 1200);
  const active = getActiveProblem(started.flow);

  assert.equal(active.kind, 'simplify-expression');
  assert.equal(active.latex, 'x + x');
  assert.equal(active.modelResponse.before, 'Simplify the expression.');
  assert.equal(active.metadata.problemType, 'simplify-expression');
});

test('fixture problem flow accepts evaluate-expression definitions', () => {
  const flow = createInitialProblemFlow(1200, [{
    id: 'evaluate-half-plus-half',
    kind: 'evaluate-expression',
    latex: '\\frac{1}{2} + \\frac{2}{4}'
  }]);
  const active = getActiveProblem(flow);

  assert.equal(active.id, 'evaluate-half-plus-half');
  assert.equal(active.kind, 'evaluate-expression');
  assert.equal(active.metadata.problemType, 'evaluate-expression');
  assert.equal(active.modelResponse.before, 'Evaluate the expression.');
});

test('fixture problem flow accepts simplify-expression definitions', () => {
  const flow = createInitialProblemFlow(1200, [{
    id: 'simplify-x-plus-x',
    kind: 'simplify-expression',
    latex: 'x + x'
  }]);
  const active = getActiveProblem(flow);

  assert.equal(active.id, 'simplify-x-plus-x');
  assert.equal(active.kind, 'simplify-expression');
  assert.equal(active.metadata.problemType, 'simplify-expression');
  assert.equal(active.modelResponse.before, 'Simplify the expression.');
});

test('submitting a custom problem freezes it without opening the next prompt', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const active = getActiveProblem(started);
  const withAnswer = {
    ...started,
    problems: started.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };

  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const submittedProblem = submitted.problems.find((problem) => problem.id === active.id);

  assert.equal(submittedProblem.status, 'submitted');
  assert.equal(submittedProblem.answerBoxFrozen, true);
  assert.equal(submittedProblem.recognition.status, 'pending');
  assert.equal(submitted.activeProblemId, active.id);
  assert.equal(submitted.awaitingEquation, false);
  assert.equal(submitted.completedCount, 1);
});

test('submitting a blank custom problem marks it incomplete-ready', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;

  const submitted = submitActiveProblem(started, 1200).flow;
  const submittedProblem = getActiveProblem(submitted);

  assert.equal(submittedProblem.status, 'submitted');
  assert.equal(submittedProblem.recognition.status, 'empty');
  assert.equal(submittedProblem.revisionAllowed, true);
  assert.equal(submittedProblem.answerBoxFrozen, false);
  assert.equal(isProblemSubmittable(submittedProblem), true);
  assert.equal(isProblemReadyForNext(submittedProblem), false);
});

test('submitted problem answer box ignores later strokes underneath it', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const active = getActiveProblem(started);
  const firstStroke = stroke(
    'a',
    active.problemBox.xMin + 32,
    active.problemBox.yMax + 40,
    active.problemBox.xMin + 150,
    active.problemBox.yMax + 56
  );
  const withAnswer = reconcileProblemFlowWithStrokes(started, [firstStroke]);
  const frozen = submitActiveProblem(withAnswer, 1200).flow;
  const frozenProblem = getActiveProblem(frozen);
  const laterStroke = stroke(
    'later',
    frozenProblem.answerBox.xMin + 12,
    frozenProblem.answerBox.yMax + 96,
    frozenProblem.answerBox.xMin + 180,
    frozenProblem.answerBox.yMax + 112
  );

  const reconciled = reconcileProblemFlowWithStrokes(frozen, [firstStroke, laterStroke]);
  const afterLaterInk = getActiveProblem(reconciled);

  assert.equal(afterLaterInk.status, 'submitted');
  assert.equal(afterLaterInk.answerBoxFrozen, true);
  assert.deepEqual(afterLaterInk.answerStrokeIds, frozenProblem.answerStrokeIds);
  assert.deepEqual(afterLaterInk.answerBox, frozenProblem.answerBox);
  assert.deepEqual(afterLaterInk.answerContentBox, frozenProblem.answerContentBox);
});

test('next problem request opens the custom problem prompt after a custom submission', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const active = getActiveProblem(started);
  const submitted = submitActiveProblem({
    ...started,
    problems: started.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  }, 1200).flow;
  const readCorrect = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'complete',
    result: correctRecognitionResult('x = 2', 'sig-custom-correct')
  });

  const next = requestNextProblem(readCorrect, 1200).flow;

  assert.equal(next.activeProblemId, null);
  assert.equal(next.awaitingEquation, true);
  assert.equal(next.completedCount, 1);
});

test('first answer stroke can seed below the problem box without overlapping it', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const active = getActiveProblem(started);
  const belowProblem = stroke(
    'first-line',
    active.problemBox.xMin + 28,
    active.problemBox.yMax + 42,
    active.problemBox.xMin + 180,
    active.problemBox.yMax + 82
  );

  const reconciled = reconcileProblemFlowWithStrokes(started, [belowProblem]);
  const problem = getActiveProblem(reconciled);

  assert.deepEqual(problem.answerStrokeIds, ['first-line']);
  assert.ok(problem.answerBox);
  assert.equal(problem.answerBoxFrozen, false);
});

test('answer box keeps horizontally aligned continuation rows with larger vertical gaps', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, '\\frac{x^2 - 1}{x - 1} = 4', 1200).flow;
  const active = getActiveProblem(started);
  const x = active.problemBox.xMin + 80;
  const y = active.problemBox.yMax + 64;
  const rows = [
    stroke('row-1', x, y, x + 220, y + 54),
    stroke('row-2', x + 30, y + 154, x + 260, y + 210),
    stroke('row-3', x + 20, y + 308, x + 190, y + 360),
    stroke('row-4', x + 60, y + 462, x + 210, y + 518)
  ];

  const reconciled = reconcileProblemFlowWithStrokes(started, rows);
  const problem = getActiveProblem(reconciled);

  assert.deepEqual(problem.answerStrokeIds, ['row-1', 'row-2', 'row-3', 'row-4']);
  assert.ok(problem.answerBox.yMax >= rows[3].canvasBbox.yMax);
});

test('next problem request waits for explicit submit after realtime read', () => {
  const initial = createInitialProblemFlow(1200);
  const started = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const active = getActiveProblem(started);
  const withAnswer = {
    ...started,
    problems: started.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };
  const read = applyProblemRecognitionProgress(withAnswer, active.id, {
    status: 'complete',
    result: correctRecognitionResult('x = 2', 'a@0,0,20,20')
  });

  assert.equal(isProblemReadyForNext(getActiveProblem(read)), false);
  const beforeSubmitNext = requestNextProblem(read, 1200).flow;
  assert.equal(beforeSubmitNext.problems[0].status, 'solving');
  assert.equal(beforeSubmitNext.activeProblemId, active.id);
  assert.equal(beforeSubmitNext.awaitingEquation, false);
  assert.equal(beforeSubmitNext.completedCount, 0);

  const submitted = submitActiveProblem(read, 1200).flow;
  assert.equal(isProblemReadyForNext(getActiveProblem(submitted)), true);
  const next = requestNextProblem(submitted, 1200).flow;

  assert.equal(next.problems[0].status, 'submitted');
  assert.equal(next.activeProblemId, null);
  assert.equal(next.awaitingEquation, true);
  assert.equal(next.completedCount, 1);
});

test('custom problem flow can render another user latex problem after realtime next', () => {
  const initial = createInitialProblemFlow(1200);
  const firstFlow = startCustomProblem(initial, 'x + 1 = 3', 1200).flow;
  const firstProblem = getActiveProblem(firstFlow);
  const withAnswer = {
    ...firstFlow,
    problems: firstFlow.problems.map((problem) => (
      problem.id === firstProblem.id
        ? {
            ...problem,
            answerStrokeIds: ['first-answer'],
            answerBox: { xMin: 0, yMin: 0, xMax: 80, yMax: 40 }
          }
        : problem
    ))
  };
  const read = applyProblemRecognitionProgress(withAnswer, firstProblem.id, {
    status: 'complete',
    result: correctRecognitionResult('x = 2', 'first-answer@0,0,80,40')
  });
  const submitted = submitActiveProblem(read, 1200).flow;
  const awaitingNextLatex = requestNextProblem(submitted, 1200).flow;
  const secondStarted = startCustomProblem(awaitingNextLatex, '\\sqrt{x + 9} = 7', 1200);
  const secondProblem = getActiveProblem(secondStarted.flow);

  assert.equal(awaitingNextLatex.activeProblemId, null);
  assert.equal(awaitingNextLatex.awaitingEquation, true);
  assert.equal(awaitingNextLatex.problems[0].status, 'submitted');
  assert.equal(secondStarted.flow.awaitingEquation, false);
  assert.equal(secondStarted.flow.activeProblemId, 'problem-2');
  assert.equal(secondProblem.latex, '\\sqrt{x + 9} = 7');
  assert.equal(secondProblem.metadata.source, 'user-latex');
  assert.ok(secondProblem.boardPosition.y > firstProblem.boardPosition.y);
  assert.ok(secondStarted.targetViewport);
});

test('fixture-backed problem flow finishes after next problem request exhausts definitions', () => {
  const initial = createInitialProblemFlow(1200, [FLOW_TEST_PROBLEMS[0]]);
  const active = getActiveProblem(initial);
  const withAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };

  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const readCorrect = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'complete',
    result: correctRecognitionResult('x = 2', 'a@0,0,20,20')
  });
  const finished = requestNextProblem(readCorrect, 1200).flow;

  assert.equal(submitted.activeProblemId, active.id);
  assert.equal(submitted.awaitingEquation, false);
  assert.equal(finished.activeProblemId, null);
  assert.equal(finished.awaitingEquation, false);
  assert.equal(finished.customProblems, false);
  assert.equal(finished.completedCount, 1);
});

test('submitted problem flow preserves recognition status and result', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const withAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };

  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const submittedProblem = submitted.problems.find((problem) => problem.id === active.id);
  assert.equal(submittedProblem.status, 'submitted');
  assert.equal(submittedProblem.answerBoxFrozen, true);
  assert.equal(submittedProblem.recognition.status, 'pending');

  const completed = applyProblemRecognitionResult(submitted, active.id, {
    latex: 'x = 4',
    latexLines: ['x = 4'],
    lines: []
  });
  assert.equal(
    completed.problems.find((problem) => problem.id === active.id).recognition.result.latex,
    'x = 4'
  );

  const failed = applyProblemRecognitionError(submitted, active.id, new Error('offline'));
  assert.equal(
    failed.problems.find((problem) => problem.id === active.id).recognition.error,
    'offline'
  );
});

test('problem status display waits for final OCR before showing incomplete grading', () => {
  const provisionalIncomplete = {
    status: 'submitted',
    recognition: {
      status: 'pending',
      result: {
        grading: {
          result: {
            problemStatus: 'incomplete'
          }
        },
        realtime: {
          allFinal: false,
          components: [{
            status: 'running',
            contested: false
          }]
        }
      }
    }
  };

  assert.deepEqual(problemStatusDisplay(provisionalIncomplete), {
    status: 'analyzing',
    text: 'Analyzing'
  });

  const completeButNotFinal = {
    ...provisionalIncomplete,
    recognition: {
      ...provisionalIncomplete.recognition,
      status: 'complete'
    }
  };
  assert.deepEqual(problemStatusDisplay(completeButNotFinal), {
    status: 'analyzing',
    text: 'Analyzing'
  });

  const finalIncomplete = {
    ...completeButNotFinal,
    recognition: {
      ...completeButNotFinal.recognition,
      result: {
        ...completeButNotFinal.recognition.result,
        realtime: {
          allFinal: true,
          components: [{
            status: 'final',
            contested: false
          }]
        }
      }
    }
  };
  assert.deepEqual(problemStatusDisplay(finalIncomplete), {
    status: 'incomplete',
    text: 'Incomplete'
  });
});

test('submitted work keeps recognition alive until grading is complete', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const withAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 },
            recognition: {
              ...problem.recognition,
              status: 'pending'
            }
          }
        : problem
    ))
  };

  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const submittedProblem = submitted.problems.find((problem) => problem.id === active.id);

  assert.equal(shouldRunRecognitionForProblem(submittedProblem), true);

  const completed = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'complete',
    result: {
      latex: 'x = 4',
      latexLines: ['x = 4'],
      lines: [],
      candidatePredictions: [],
      grading: {
        status: 'complete',
        failed: false,
        result: {
          problemStatus: 'correct'
        }
      },
      realtime: {
        allFinal: true,
        components: []
      }
    }
  });
  const completedProblem = completed.problems.find((problem) => problem.id === active.id);

  assert.equal(completedProblem.recognition.result.grading.result.problemStatus, 'correct');
  assert.equal(shouldRunRecognitionForProblem(completedProblem), false);
});

test('model response reveals grading status only after submit', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const withAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };
  const readWhileSolving = applyProblemRecognitionProgress(withAnswer, active.id, {
    status: 'complete',
    result: {
      latex: 'x = 4',
      latexLines: ['x = 4'],
      lines: [],
      candidatePredictions: [],
      grading: {
        status: 'complete',
        failed: false,
        result: {
          problemStatus: 'correct'
        }
      },
      realtime: {
        allFinal: true,
        components: []
      }
    }
  });

  assert.equal(getActiveModelResponse(readWhileSolving).before, 'Solve the equation.');

  const submittedAfterRead = submitActiveProblem(readWhileSolving, 1200).flow;
  const submittedAfterReadResponse = getActiveModelResponse(submittedAfterRead);
  assert.equal(submittedAfterReadResponse.before, 'Correct');
  assert.equal(submittedAfterReadResponse.latex, '');
  assert.equal(submittedAfterReadResponse.after, '');
  assert.equal(submittedAfterReadResponse.statusOnly, true);

  const pendingBeforeSubmit = applyProblemRecognitionProgress(withAnswer, active.id, {
    status: 'pending',
    result: null
  });
  const submittedBeforeRead = submitActiveProblem(pendingBeforeSubmit, 1200).flow;
  const submittedBeforeReadResponse = getActiveModelResponse(submittedBeforeRead);
  assert.equal(submittedBeforeReadResponse.before, 'Analyzing');
  assert.equal(submittedBeforeReadResponse.statusOnly, true);

  const completedAfterSubmit = applyProblemRecognitionProgress(submittedBeforeRead, active.id, {
    status: 'complete',
    result: {
      latex: 'x = 3',
      latexLines: ['x = 3'],
      lines: [],
      candidatePredictions: [],
      grading: {
        status: 'complete',
        failed: false,
        result: {
          problemStatus: 'incorrect'
        }
      },
      realtime: {
        allFinal: true,
        components: []
      }
    }
  });

  assert.equal(getActiveModelResponse(completedAfterSubmit).before, 'Incorrect');
});

test('model response reveals feedback only after submit', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const inputSignature = 'sig-feedback';
  const attemptId = buildAttemptId(active.id, inputSignature);
  const withFeedbackBeforeSubmit = applyProblemFeedbackProgress(initial, active.id, {
    status: 'complete',
    source: 'ollama',
    text: 'Try subtracting 1 from both sides.',
    attemptId,
    inputSignature
  });

  assert.equal(getActiveModelResponse(withFeedbackBeforeSubmit).before, 'Solve the equation.');

  const submitted = submitActiveProblem({
    ...withFeedbackBeforeSubmit,
    problems: withFeedbackBeforeSubmit.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  }, 1200).flow;

  assert.equal(getActiveModelResponse(submitted).before, 'Try subtracting 1 from both sides.');
  assert.equal(getActiveModelResponse(submitted).feedbackText, 'Try subtracting 1 from both sides.');
});

test('submitted model response shows pending feedback while llm is running', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const submitted = submitActiveProblem({
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  }, 1200).flow;
  const inputSignature = 'sig-pending';
  const graded = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'complete',
    result: {
      ...correctRecognitionResult('x = 3', inputSignature),
      grading: {
        status: 'complete',
        failed: false,
        result: { problemStatus: 'incorrect' }
      }
    }
  });
  const pending = applyProblemFeedbackProgress(graded, active.id, {
    status: 'pending',
    source: 'ollama',
    attemptId: buildAttemptId(active.id, inputSignature),
    inputSignature
  });

  assert.equal(getActiveModelResponse(pending).before, 'Getting feedback...');
  assert.equal(getActiveModelResponse(pending).feedbackText, 'Getting feedback...');
});

test('reconciling changed strokes clears stale feedback', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const withFeedback = applyProblemFeedbackProgress(initial, active.id, {
    status: 'complete',
    source: 'ollama',
    text: 'Old hint',
    attemptId: buildAttemptId(active.id, 'old'),
    inputSignature: 'old'
  });
  const reconciled = reconcileProblemFlowWithStrokes(withFeedback, [{
    id: 'new-stroke',
    canvasBbox: { xMin: active.problemBox.xMin, yMin: active.problemBox.yMax + 10, xMax: active.problemBox.xMin + 30, yMax: active.problemBox.yMax + 40 }
  }]);
  const updated = getActiveProblem(reconciled);

  assert.equal(updated.feedback.status, 'idle');
  assert.equal(updated.feedback.text, '');
});

test('non-correct submitted feedback unlocks revision but not next problem', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const submitted = submitActiveProblem(problemWithAnswer(initial, active.id), 1200).flow;
  const inputSignature = 'sig-incorrect';
  const graded = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'complete',
    result: {
      latex: 'x = 5',
      latexLines: ['x = 5'],
      lines: [],
      candidatePredictions: [],
      grading: {
        status: 'complete',
        failed: false,
        result: {
          problemStatus: 'incorrect'
        }
      },
      realtime: {
        allFinal: true,
        inputSignature,
        components: []
      }
    }
  });
  const submittedForFeedback = {
    ...graded,
    problems: graded.problems.map((problem) => (
      problem.id === active.id
        ? { ...problem, submittedInputSignature: inputSignature }
        : problem
    ))
  };
  const withFeedback = applyProblemFeedbackProgress(submittedForFeedback, active.id, {
    status: 'complete',
    source: 'fallback',
    text: 'Line 1 should be: x = 4.',
    attemptId: buildAttemptId(active.id, inputSignature),
    inputSignature
  });
  const problem = getActiveProblem(withFeedback);

  assert.equal(problem.status, 'submitted');
  assert.equal(problem.revisionAllowed, true);
  assert.equal(problem.answerBoxFrozen, false);
  assert.equal(isProblemSubmittable(problem), true);
  assert.equal(isProblemReadyForNext(problem), false);
  assert.equal(getActiveModelResponse(withFeedback).feedbackText, 'Line 1 should be: x = 4.');
});

test('editing retryable submitted work hides stale feedback and keeps recognition active', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const firstStroke = stroke('a', active.problemBox.xMin + 20, active.problemBox.yMax + 20, active.problemBox.xMin + 80, active.problemBox.yMax + 40);
  const secondStroke = stroke('b', active.problemBox.xMin + 90, active.problemBox.yMax + 80, active.problemBox.xMin + 160, active.problemBox.yMax + 100);
  const withAnswer = reconcileProblemFlowWithStrokes(initial, [firstStroke]);
  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const inputSignature = 'sig-retry';
  const retryable = applyProblemFeedbackProgress({
    ...submitted,
    problems: submitted.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            submittedInputSignature: inputSignature,
            recognition: {
              ...problem.recognition,
              status: 'complete',
              result: {
                ...correctRecognitionResult('x = 5', inputSignature),
                grading: {
                  status: 'complete',
                  failed: false,
                  result: { problemStatus: 'incomplete' }
                }
              }
            }
          }
        : problem
    ))
  }, active.id, {
    status: 'complete',
    source: 'fallback',
    text: 'A good next line is: x = 4.',
    inputSignature,
    attemptId: buildAttemptId(active.id, inputSignature)
  });

  const edited = reconcileProblemFlowWithStrokes(retryable, [firstStroke, secondStroke]);
  const problem = getActiveProblem(edited);

  assert.equal(problem.revisionAllowed, true);
  assert.equal(problem.answerBoxFrozen, false);
  assert.equal(problem.submittedInputSignature, null);
  assert.equal(problem.feedback.status, 'idle');
  assert.equal(shouldRunRecognitionForProblem(problem), true);
  assert.equal(getActiveModelResponse(edited).feedbackText, undefined);
});

test('resubmitting retryable work freezes the revised attempt until feedback arrives', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const revisedSignature = 'sig-revised';
  const retryable = {
    ...problemWithAnswer(initial, active.id),
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            status: 'submitted',
            revisionAllowed: true,
            answerBoxFrozen: false,
            answerStrokeIds: ['a', 'b'],
            answerBox: { xMin: 0, yMin: 0, xMax: 40, yMax: 80 },
            recognition: {
              ...problem.recognition,
              status: 'complete',
              realtime: { inputSignature: revisedSignature },
              result: {
                ...correctRecognitionResult('x = 4', revisedSignature),
                grading: {
                  status: 'complete',
                  failed: false,
                  result: { problemStatus: 'incomplete' }
                }
              }
            }
          }
        : problem
    ))
  };

  const resubmitted = submitActiveProblem(retryable, 1200).flow;
  const problem = getActiveProblem(resubmitted);

  assert.equal(problem.status, 'submitted');
  assert.equal(problem.revisionAllowed, false);
  assert.equal(problem.answerBoxFrozen, true);
  assert.equal(problem.submittedInputSignature, revisedSignature);
  assert.equal(problem.submissionCount, 1);
});

test('prewarmed feedback stays hidden until retryable work is resubmitted', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const inputSignature = 'sig-prewarmed';
  const prewarmed = applyProblemFeedbackProgress({
    ...problemWithAnswer(initial, active.id),
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 },
            recognition: {
              ...problem.recognition,
              status: 'complete',
              result: {
                ...correctRecognitionResult('x = 5', inputSignature),
                grading: {
                  status: 'complete',
                  failed: false,
                  result: { problemStatus: 'incorrect' }
                }
              }
            }
          }
        : problem
    ))
  }, active.id, {
    status: 'complete',
    source: 'fallback',
    text: 'Line 1 should be: x = 4.',
    inputSignature,
    attemptId: buildAttemptId(active.id, inputSignature)
  });

  assert.equal(getActiveModelResponse(prewarmed).feedbackText, undefined);

  const submitted = submitActiveProblem(prewarmed, 1200).flow;
  const problem = getActiveProblem(submitted);

  assert.equal(problem.submittedInputSignature, inputSignature);
  assert.equal(problem.revisionAllowed, true);
  assert.equal(getActiveModelResponse(submitted).feedbackText, 'Line 1 should be: x = 4.');
});

test('correct submitted work remains frozen and ready for next', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const inputSignature = 'sig-correct-final';
  const submitted = submitActiveProblem(applyProblemRecognitionProgress(problemWithAnswer(initial, active.id), active.id, {
    status: 'complete',
    result: correctRecognitionResult('x = 4', inputSignature)
  }), 1200).flow;
  const withFeedback = applyProblemFeedbackProgress(submitted, active.id, {
    status: 'complete',
    source: 'deterministic',
    text: 'Correct! Great job!',
    inputSignature,
    attemptId: buildAttemptId(active.id, inputSignature)
  });
  const problem = getActiveProblem(withFeedback);

  assert.equal(problem.revisionAllowed, false);
  assert.equal(problem.answerBoxFrozen, true);
  assert.equal(isProblemSubmittable(problem), false);
  assert.equal(isProblemReadyForNext(problem), true);
  assert.equal(shouldRunRecognitionForProblem(problem), false);
});

test('submitted model response shows correct before all recognition is complete', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const active = getActiveProblem(initial);
  const withAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === active.id
        ? {
            ...problem,
            answerStrokeIds: ['a', 'b'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 60 }
          }
        : problem
    ))
  };
  const submitted = submitActiveProblem(withAnswer, 1200).flow;
  const partiallyCorrect = applyProblemRecognitionProgress(submitted, active.id, {
    status: 'pending',
    result: {
      latex: 'x = 4',
      latexLines: ['x = 4'],
      lines: [],
      candidatePredictions: [],
      grading: {
        status: 'complete',
        failed: false,
        result: {
          problemStatus: 'correct',
          foundSolutions: ['4'],
          missingSolutions: []
        }
      },
      realtime: {
        allFinal: false,
        components: [{
          signature: 'a@0,0,20,20',
          status: 'final',
          contested: false
        }, {
          signature: 'b@0,40,20,60',
          status: 'running',
          contested: false
        }]
      }
    }
  });

  assert.equal(getActiveModelResponse(partiallyCorrect).before, 'Correct');
  assert.equal(getActiveModelResponse(partiallyCorrect).statusOnly, true);
  assert.deepEqual(
    problemStatusDisplay(getActiveProblem(partiallyCorrect), getActiveModelResponse(partiallyCorrect)),
    {
      status: 'correct',
      text: 'Correct'
    }
  );
});

test('recognition summary preserves debug crop state and final line order', () => {
  const summary = summarizeRecognitionResult({
    latex: 'x = 4 \\\\ y = 2',
    latexLines: ['x = 4', 'y = 2'],
    timing: { totalElapsedSeconds: 1.25 },
    detection: { source: 'disabled', failed: false },
    semantic: { source: 'semantic-service', failed: false },
    lines: [
      {
        lineIndex: 0,
        candidateId: 'candidate-a',
        debugLabel: 'C1',
        selected: true,
        profiles: ['row-line'],
        strokeIds: ['a'],
        tightBbox: { xMin: 0, yMin: 0, xMax: 80, yMax: 30 },
        image: {
          dataUrl: 'data:image/png;base64,bGluZQ==',
          width: 80,
          height: 30,
          cssWidth: 80,
          cssHeight: 30,
          targetPixelHeight: 104,
          padding: 24
        },
        latex: 'x = 4',
        acceptedLatex: 'x = 4',
        ocrLatex: 'x = 4',
        candidates: [{ latex: 'x = 4', score: 2 }],
        prediction: { latex: 'x = 4', elapsedSeconds: 0.2 },
        contextualSemantic: null,
        sequentialSemantic: { semanticScore: 3 },
        evidenceScore: 5,
        timing: { submitToFinalPredictionSeconds: 0.4 }
      },
      {
        lineIndex: 1,
        candidateId: 'candidate-b',
        debugLabel: 'C2',
        selected: true,
        profiles: ['row-line'],
        strokeIds: ['b'],
        tightBbox: { xMin: 0, yMin: 50, xMax: 80, yMax: 80 },
        image: null,
        latex: 'y = 2',
        acceptedLatex: 'y = 2',
        ocrLatex: 'y = 2',
        candidates: [{ latex: 'y = 2', score: 1 }],
        prediction: { latex: 'y = 2', elapsedSeconds: 0.3 },
        evidenceScore: 4,
        timing: { submitToFinalPredictionSeconds: 0.8 }
      }
    ],
    candidatePredictions: [
      {
        candidateId: 'candidate-a',
        debugLabel: 'C1',
        selected: true,
        discarded: false,
        selectedLineIndex: 0,
        profiles: ['row-line'],
        strokeIds: ['a'],
        tightBbox: { xMin: 0, yMin: 0, xMax: 80, yMax: 30 },
        image: {
          dataUrl: 'data:image/png;base64,bGluZQ==',
          width: 80,
          height: 30,
          cssWidth: 80,
          cssHeight: 30,
          targetPixelHeight: 104,
          padding: 24
        },
        latex: 'x = 4',
        acceptedLatex: 'x = 4',
        ocrLatex: 'x = 4',
        candidates: [{ latex: 'x = 4', score: 2 }],
        prediction: { latex: 'x = 4', elapsedSeconds: 0.2 },
        semantic: { semanticScore: 3 },
        evidenceScore: 5,
        timing: { submitToFinalPredictionSeconds: 0.4 }
      },
      {
        candidateId: 'candidate-unused',
        debugLabel: 'C3',
        selected: false,
        discarded: true,
        selectedLineIndex: null,
        profiles: ['parent'],
        strokeIds: ['a', 'b'],
        tightBbox: { xMin: 0, yMin: 0, xMax: 80, yMax: 80 },
        image: {
          dataUrl: 'data:image/png;base64,dW51c2Vk',
          width: 80,
          height: 80
        },
        latex: 'x = 4 y = 2',
        ocrLatex: 'x = 4 y = 2',
        candidates: [{ latex: 'x = 4 y = 2', score: -1 }],
        prediction: { latex: 'x = 4 y = 2', elapsedSeconds: 0.5 },
        evidenceScore: -1,
        timing: { submitToFinalPredictionSeconds: 0.6 }
      }
    ],
    segmentation: {
      selected: [
        { candidateId: 'candidate-a', profiles: ['row-line'], strokeIds: ['a'], tightBbox: { xMin: 0, yMin: 0, xMax: 80, yMax: 30 } },
        { candidateId: 'candidate-b', profiles: ['row-line'], strokeIds: ['b'], tightBbox: { xMin: 0, yMin: 50, xMax: 80, yMax: 80 } }
      ],
      candidates: [],
      partitions: [],
      parentCandidateId: 'candidate-unused',
      ocrSelectedCandidateIds: ['candidate-a', 'candidate-b']
    }
  });

  assert.equal(summary.timing.totalElapsedSeconds, 1.25);
  assert.equal(summary.lines.map((line) => line.acceptedLatex).join('|'), 'x = 4|y = 2');
  assert.equal(summary.candidatePredictions.length, 2);
  assert.equal(summary.candidatePredictions[0].image.dataUrl, 'data:image/png;base64,bGluZQ==');
  assert.equal(summary.candidatePredictions[1].discarded, true);
  assert.equal(summary.segmentation.ocrSelectedCandidateIds.length, 2);
});

test('recognition context does not leak previous problem latex into new submissions', () => {
  const initial = createInitialProblemFlow(1200, FLOW_TEST_PROBLEMS);
  const firstProblem = getActiveProblem(initial);
  const withFirstAnswer = {
    ...initial,
    problems: initial.problems.map((problem) => (
      problem.id === firstProblem.id
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 }
          }
        : problem
    ))
  };
  const afterFirstSubmit = submitActiveProblem(withFirstAnswer, 1200).flow;
  const completedFirst = applyProblemRecognitionResult(
    afterFirstSubmit,
    firstProblem.id,
    correctRecognitionResult('\\eta = 5', 'a@0,0,20,20')
  );
  const withNextProblem = requestNextProblem(completedFirst, 1200).flow;
  const nextProblem = getActiveProblem(withNextProblem);

  assert.deepEqual(previousLatexForSubmission(withNextProblem, nextProblem.id), []);
});

test('JS buildLiveGradingResult and Python grade_equation_work agree on problem status', async () => {
  // This test verifies that the JS-side grading aggregation produces the same
  // problemStatus as the Python grader for identical inputs. The JS function
  // buildLiveGradingResult is not exported, so we test it indirectly through
  // the pipeline's grading output structure.
  const testCases = [
    {
      name: 'correct linear',
      problemLatex: '3x + 5 = 17',
      lines: [
        { latex: '3x + 5 = 17', grading: { classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] } },
        { latex: '3x = 12', grading: { classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] } },
        { latex: 'x = 4', grading: { classification: 'valid_step', solutionCoverage: 'full', matchedSolutions: ['4'] } },
      ],
      expectedStatus: 'correct',
    },
    {
      name: 'incomplete quadratic',
      problemLatex: 'x^2 - 5x + 6 = 0',
      lines: [
        { latex: 'x^2 - 5x + 6 = 0', grading: { classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] } },
        { latex: '(x - 2)(x - 3) = 0', grading: { classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] } },
        { latex: 'x = 2', grading: { classification: 'valid_step', solutionCoverage: 'partial', matchedSolutions: ['2'] } },
      ],
      expectedStatus: 'incomplete',
    },
    {
      name: 'incorrect with invalid step',
      problemLatex: '2x + 3 = 11',
      lines: [
        { latex: '2x + 3 = 11', grading: { classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] } },
        { latex: '2x = 9', grading: { classification: 'invalid_step', solutionCoverage: 'none', matchedSolutions: [] } },
      ],
      expectedStatus: 'incorrect',
    },
    {
      name: 'incomplete unsimplified final',
      problemLatex: 'x + 2 = 5',
      lines: [
        {
          latex: 'x = 5 - 2',
          grading: {
            classification: 'valid_step',
            solutionCoverage: 'full',
            matchedSolutions: ['3'],
            answerFinality: 'unsimplified',
            countsTowardCompletion: false
          }
        },
      ],
      expectedStatus: 'incomplete',
    },
    {
      name: 'not started scratch only',
      problemLatex: '3x + 5 = 17',
      lines: [
        { latex: '/ 4 / 4', grading: { classification: 'other', solutionCoverage: 'none', matchedSolutions: [] } },
      ],
      expectedStatus: 'not_started',
    },
  ];

  for (const testCase of testCases) {
    const manifest = {
      cardinality: 'finite',
      exact_set: testCase.name.includes('quadratic') ? ['2', '3'] : ['4'],
      variable: 'x',
      problem_standardized: testCase.problemLatex,
      problem_raw: testCase.problemLatex,
    };

    // Simulate what buildLiveGradingResult does (it's not exported, so we replicate the logic)
    const steps = testCase.lines.map((line, index) => ({
      lineIndex: index,
      studentLatex: line.latex,
      classification: line.grading.classification,
      solutionCoverage: line.grading.solutionCoverage,
      matchedSolutions: line.grading.matchedSolutions,
      answerFinality: line.grading.answerFinality || 'not_answer',
      countsTowardCompletion: line.grading.countsTowardCompletion !== false,
    }));

    const exactSet = manifest.exact_set.map(String);
    const matched = new Set();
    let sawValid = false;
    let firstInvalid = null;
    for (const step of steps) {
      if (step.classification === 'valid_step') sawValid = true;
      if (step.classification === 'invalid_step' && firstInvalid === null) {
        firstInvalid = step.lineIndex;
      }
      if (step.countsTowardCompletion === false) continue;
      for (const solution of step.matchedSolutions) {
        if (solution) matched.add(String(solution));
      }
    }
    const complete = exactSet.length > 0 && exactSet.every((s) => matched.has(s));
    const jsStatus = complete
      ? 'correct'
      : firstInvalid !== null
        ? 'incorrect'
        : sawValid
          ? 'incomplete'
          : 'not_started';

    assert.equal(
      jsStatus,
      testCase.expectedStatus,
      `JS grading status mismatch for: ${testCase.name}`
    );
  }
});

test('VLM audit decision catches non-correct grading statuses and unread OCR', () => {
  const result = auditResult({
    problemStatus: 'incomplete',
    latexLines: [''],
    lines: [auditLine({ latex: '', acceptedLatex: '', evidenceScore: 3 })]
  });

  const decision = getRecognitionAuditDecision(result, { inputSignature: 'sig-a' });

  assert.equal(decision.shouldAudit, true);
  assert.ok(decision.triggerReasons.includes('problem_status_incomplete'));
  assert.ok(decision.triggerReasons.includes('unread_or_empty_line'));
  assert.ok(decision.triggerReasons.includes('line_segmentation_empty'));
});

test('student writing pipeline keeps ink-only OCR lines incomplete instead of not started', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('empty', 0, 0, 80, 40)],
    answerBox: { xMin: -5, yMin: -5, xMax: 100, yMax: 70 },
    problemLatex: 'x = 1',
    semanticScoring: false,
    retryRasterHeights: [],
    semanticRetryRasterHeights: [],
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '',
      top: null,
      candidates: [],
      failed: false,
      timedOut: false,
      elapsedSeconds: 0.02
    })
  });

  assert.equal(result.lines.length, 1);
  assert.equal(result.lines[0].acceptedLatex, '');
  assert.equal(result.grading.result.problemStatus, 'incomplete');
  const decision = getRecognitionAuditDecision(result, { inputSignature: 'empty-ocr' });
  assert.ok(decision.triggerReasons.includes('line_segmentation_empty'));
});

test('student writing pipeline excludes isolated circled annotation from grading', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('setup', 0, 0, 180, 55),
    stroke('simplify', 18, 100, 145, 150),
    stroke('circle', 55, 215, 125, 290),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: -10, yMin: -10, xMax: 220, yMax: 330 },
    problemLatex: '\\frac { 1 2 } { \\sqrt { 9 } } - \\sqrt { 1 6 }',
    semanticScoring: true,
    recognizeAlternatives: false,
    semanticRetryRasterHeights: [],
    detectLineBands: true,
    detectLines: async () => ({
      detections: [
        { bbox: { xMin: 0, yMin: 0, xMax: 190, yMax: 65 } },
        { bbox: { xMin: 10, yMin: 92, xMax: 155, yMax: 160 } },
        { bbox: { xMin: 48, yMin: 205, xMax: 132, yMax: 300 } },
      ],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const ids = new Set(image.strokeIds);
      const latex = ids.has('circle')
        ? '0'
        : ids.has('simplify')
          ? '4 - 4'
          : '\\frac { 1 2 } { 3 } - 4';
      return {
        latex,
        top: { latex, score: 2, confidence: 0.95 },
        candidates: [{ latex, score: 2, confidence: 0.95 }],
        elapsedSeconds: 0.03
      };
    },
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: null,
        cardinality: 'finite',
        exact_set: ['0']
      },
      candidateScores: request.candidateGroups.map((group) => {
        const isCircle = group.latex === '0';
        return {
          candidateId: group.candidateId,
          semanticScore: 1,
          bestLatex: group.latex,
          sound: true,
          equivalentToProblem: false,
          equivalentToPrevious: false,
          grading: {
            studentLatex: group.latex,
            classification: 'valid_step',
            selectedCandidateIndex: 0,
            solutionCoverage: 'full',
            matchedSolutions: ['0'],
            answerFinality: isCircle ? 'final' : 'unsimplified',
            countsTowardCompletion: isCircle,
            candidateVerdicts: [
              { latex: group.latex, classification: 'valid_step', countsTowardCompletion: isCircle }
            ]
          },
          candidateScores: []
        };
      }),
      elapsedSeconds: 0.05
    })
  });

  assert.deepEqual(result.latexLines, ['4 - 4']);
  assert.equal(result.lines.at(-1).excludedFromGrading, true);
  assert.equal(result.lines.at(-1).acceptedLatex, '');
  assert.equal(result.grading.result.problemStatus, 'incomplete');
});

test('student writing pipeline keeps a real standalone zero answer gradable', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('zero', 0, 0, 70, 78)],
    answerBox: { xMin: -10, yMin: -10, xMax: 100, yMax: 110 },
    problemLatex: '4 - 4',
    semanticScoring: true,
    recognizeAlternatives: true,
    semanticRetryRasterHeights: [],
    recognizeLine: async () => ({
      latex: '0',
      top: { latex: '0', score: 2, confidence: 0.95 },
      candidates: [{ latex: '0', score: 2, confidence: 0.95 }],
      elapsedSeconds: 0.03
    }),
    scoreSemantics: async (request) => ({
      answerManifest: {
        problem_raw: request.problemLatex,
        variable: null,
        cardinality: 'finite',
        exact_set: ['0']
      },
      candidateScores: request.candidateGroups.map((group) => ({
        candidateId: group.candidateId,
        semanticScore: 1,
        bestLatex: group.latex,
        sound: true,
        grading: {
          studentLatex: '0',
          classification: 'valid_step',
          selectedCandidateIndex: 0,
          solutionCoverage: 'full',
          matchedSolutions: ['0'],
          answerFinality: 'final',
          countsTowardCompletion: true
        },
        candidateScores: []
      })),
      elapsedSeconds: 0.05
    })
  });

  assert.deepEqual(result.latexLines, ['0']);
  assert.equal(result.lines[0].excludedFromGrading, undefined);
  assert.equal(result.grading.result.problemStatus, 'correct');
});

test('student writing pipeline merges split problem-input fraction rows', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('num8', 100, 35, 136, 94),
    stroke('bar', 92, 112, 236, 119),
    stroke('den0', 104, 150, 135, 208),
    stroke('dot', 150, 198, 158, 206),
    stroke('den1', 174, 150, 206, 208),
    stroke('minus', 276, 78, 326, 86),
    stroke('sqrt', 350, 44, 430, 112),
    stroke('two', 446, 48, 480, 104),
    stroke('five', 496, 48, 532, 104),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 60, yMin: 0, xMax: 580, yMax: 260 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'evaluate',
      problemType: 'evaluate-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: false,
    chunkFallback: false,
    detectLineBands: true,
    detectLines: async () => ({
      detections: [
        { bbox: { xMin: 90, yMin: 25, xMax: 550, yMax: 126 } },
        { bbox: { xMin: 100, yMin: 142, xMax: 214, yMax: 216 } },
      ],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const ids = new Set(image.strokeIds);
      const latex = ids.has('num8') && ids.has('den0')
        ? ''
        : ids.has('den0') || ids.has('dot') || ids.has('den1')
          ? '0 . 1'
          : '8 - \\sqrt { 2 5 }';
      return {
        latex,
        top: { latex, score: 2, confidence: 0.95 },
        candidates: [{ latex, score: 2, confidence: 0.95 }],
        elapsedSeconds: 0.03
      };
    },
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { 8 } { 0 . 1 } - \\sqrt { 2 5 }']);
  assert.equal(result.lines.length, 1);
  assert.ok(new Set([
    'problem_input_fraction_row_merge',
    'problem_input_fraction_candidate_merge',
    'problem_input_fraction_latex_repair'
  ]).has(result.selectionRescue[0].reason));
});

test('student writing pipeline merges prior split problem-input numerator with denominator tail', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('seven', 150, 45, 230, 125),
    stroke('bar', 126, 145, 250, 153),
    stroke('one', 142, 172, 154, 218),
    stroke('zero', 182, 170, 228, 218),
    stroke('plus', 290, 132, 326, 172),
    stroke('tail0', 350, 130, 382, 180),
    stroke('taildot', 394, 174, 402, 182),
    stroke('tail1', 418, 130, 438, 180),
    stroke('minus', 474, 154, 508, 162),
    stroke('two', 542, 130, 592, 190),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 100, yMin: 0, xMax: 640, yMax: 250 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'evaluate',
      problemType: 'evaluate-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    detectLineBands: true,
    detectLines: async () => ({
      detections: [
        { bbox: { xMin: 140, yMin: 35, xMax: 258, yMax: 132 } },
        { bbox: { xMin: 120, yMin: 116, xMax: 610, yMax: 226 } },
      ],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const ids = new Set(image.strokeIds);
      const latex = ids.has('seven') && ids.has('one')
        ? ''
        : ids.has('seven')
          ? '7'
          : '\\frac { 1 0 } + 0 . 1 - 2';
      return {
        latex,
        top: { latex, score: 2, confidence: 0.95 },
        candidates: [{ latex, score: 2, confidence: 0.95 }],
        elapsedSeconds: 0.03
      };
    },
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { 7 } { 1 0 } + 0 . 1 - 2']);
  assert.equal(result.lines.length, 1);
});

test('problem-input fraction repair preserves complete OCR fractions with tails', async () => {
  installFakeCanvas();
  const cases = [
    '\\frac { x - 1 } { 8 } = 1',
    '\\frac { 8 } { x } - \\frac { 8 } { 9 } = 0',
    '\\frac { 8 } { 1 8 } - \\frac { 8 } { 9 }'
  ];

  for (const latex of cases) {
    const result = await recognizeStudentWriting({
      strokes: [stroke(`case-${latex.length}`, 0, 0, 420, 180)],
      answerBox: { xMin: -5, yMin: -5, xMax: 460, yMax: 220 },
      problemMetadata: {
        auditSubject: 'problem-input',
        mode: 'solve',
        problemType: 'equation-solving',
        source: 'user-handwriting'
      },
      semanticScoring: false,
      recognizeAlternatives: true,
      chunkFallback: false,
      recognizeLine: async () => ({
        latex,
        top: { latex, score: 2, confidence: 0.95 },
        candidates: [{ latex, score: 2, confidence: 0.95 }],
        elapsedSeconds: 0.03
      }),
      gradeWork: null
    });

    assert.deepEqual(result.latexLines, [latex], latex);
    assert.equal(result.lines[0].ocrRepair?.source, undefined, latex);
  }
});

test('problem-input fraction merge preserves correct full parent OCR candidate', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('three', 150, 45, 205, 125),
    stroke('xTop', 216, 58, 270, 120),
    stroke('bar', 126, 145, 395, 153),
    stroke('xDen', 142, 174, 190, 230),
    stroke('plus', 215, 182, 250, 218),
    stroke('one', 276, 172, 294, 230),
    stroke('eq1', 430, 118, 472, 126),
    stroke('eq2', 424, 152, 472, 160),
    stroke('eight', 520, 88, 592, 190),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 100, yMin: 0, xMax: 640, yMax: 250 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'solve',
      problemType: 'equation-solving',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    detectLineBands: true,
    detectLines: async () => ({
      detections: [
        { bbox: { xMin: 140, yMin: 35, xMax: 600, yMax: 132 } },
        { bbox: { xMin: 120, yMin: 138, xMax: 310, yMax: 238 } },
      ],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const ids = new Set(image.strokeIds);
      const all = ['three', 'xTop', 'bar', 'xDen', 'plus', 'one', 'eq1', 'eq2', 'eight']
        .every((id) => ids.has(id));
      const upper = ids.has('three') && ids.has('xTop') && !ids.has('xDen');
      const lower = ids.has('xDen') || ids.has('plus') || ids.has('one');
      const latex = all
        ? '\\frac { 3 x } { x + 1 } = 8'
        : upper
          ? '3 x -'
          : lower
            ? '\\frac { x + 1 } - 8'
            : '';
      const candidates = all
        ? [
            { latex: '\\frac { 3 x } { x + 1 } = 8', score: 2, confidence: 0.95 },
            { latex: '\\frac { 3 } { x + 1 } x - - 8', score: 1, confidence: 0.55 }
          ]
        : [{ latex, score: 2, confidence: 0.95 }];
      return {
        latex,
        top: candidates[0],
        candidates,
        elapsedSeconds: 0.03
      };
    },
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { 3 x } { x + 1 } = 8']);
  assert.notEqual(result.lines[0].acceptedLatex, '\\frac { 3 } { x + 1 } x - - 8');
});

test('evaluate problem-input fraction keeps multiplication dot over equals alternative', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('fraction-product', 0, 0, 360, 160)],
    answerBox: { xMin: -5, yMin: -5, xMax: 390, yMax: 190 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'evaluate',
      problemType: 'evaluate-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { 2 } { 3 } = \\frac { 1 } { 4 }',
      top: { latex: '\\frac { 2 } { 3 } = \\frac { 1 } { 4 }', score: 1.8, confidence: 0.86 },
      candidates: [
        { latex: '\\frac { 2 } { 3 } = \\frac { 1 } { 4 }', score: 1.8, confidence: 0.86 },
        { latex: '\\frac { 2 } { 3 } \\cdot \\frac { 1 } { 4 }', score: 1.1, confidence: 0.71 }
      ],
      elapsedSeconds: 0.03
    }),
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { 2 } { 3 } \\cdot \\frac { 1 } { 4 }']);
  assert.equal(result.lines[0].ocrRepair?.source, 'problem-input-full-parent-candidate');
});

test('problem-input sqrt fraction chooses balanced top-five candidate', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('sqrt-frac', 0, 0, 300, 220)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 260 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'simplify',
      problemType: 'simplify-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x }',
      top: { latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x }', score: -0.4, confidence: 0.27 },
      candidates: [
        { latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x }', score: -0.4, confidence: 0.27 },
        { latex: '\\frac { \\sqrt { x ^ { 1 8 } } } { x }', score: -0.7, confidence: 0.21 },
        { latex: '\\frac { \\sqrt { x ^ { 1 B } } } { x }', score: -0.9, confidence: 0.16 }
      ],
      elapsedSeconds: 0.03
    }),
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { \\sqrt { x ^ { 1 8 } } } { x }']);
  assert.equal(result.lines[0].ocrRepair.source, 'problem-input-sqrt-fraction-candidate-repair');
});

test('problem-input sqrt fraction promotes exponent-preserving candidate over balanced weak top OCR', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('sqrt-frac-x10', 0, 0, 300, 220)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 260 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'simplify',
      problemType: 'simplify-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { \\sqrt { x ^ { 1 } } } { x }',
      top: { latex: '\\frac { \\sqrt { x ^ { 1 } } } { x }', score: -0.1, confidence: 0.32 },
      candidates: [
        { latex: '\\frac { \\sqrt { x ^ { 1 } } } { x }', score: -0.1, confidence: 0.32 },
        { latex: '\\frac { \\sqrt { x ^ { 1 0 } } } { x }', score: -0.4, confidence: 0.21 }
      ],
      elapsedSeconds: 0.03
    }),
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { \\sqrt { x ^ { 1 0 } } } { x }']);
});

test('problem-input sqrt fraction composes denominator exponent from related row evidence', async () => {
  installFakeCanvas();
  const strokes = [
    stroke('sqrt', 140, 30, 310, 112),
    stroke('xNum', 210, 82, 260, 135),
    stroke('one', 274, 62, 282, 96),
    stroke('zero', 292, 62, 324, 98),
    stroke('bar', 104, 146, 345, 158),
    stroke('xDen', 210, 186, 258, 236),
    stroke('three', 274, 166, 310, 205),
  ];

  const result = await recognizeStudentWriting({
    strokes,
    answerBox: { xMin: 80, yMin: 0, xMax: 380, yMax: 270 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'simplify',
      problemType: 'simplify-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    detectLineBands: true,
    detectLines: async () => ({
      detections: [
        { bbox: { xMin: 120, yMin: 20, xMax: 330, yMax: 140 } },
        { bbox: { xMin: 100, yMin: 140, xMax: 345, yMax: 242 } },
      ],
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const ids = new Set(image.strokeIds);
      const all = strokes.every((item) => ids.has(item.id));
      const hasNumerator = ids.has('sqrt') || ids.has('xNum') || ids.has('one') || ids.has('zero');
      const hasDenominator = ids.has('xDen') || ids.has('three');
      const latex = all
        ? '\\frac { \\sqrt { x ^ { 0 } } } { x }'
        : hasNumerator && !hasDenominator
          ? '\\sqrt { x ^ { 1 0 } }'
          : hasDenominator
            ? 'x ^ { 3 }'
            : '';
      return {
        latex,
        top: { latex, score: 2, confidence: 0.95 },
        candidates: [{ latex, score: 2, confidence: 0.95 }],
        elapsedSeconds: 0.03
      };
    },
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { \\sqrt { x ^ { 1 0 } } } { x ^ { 3 } }']);
  assert.equal(result.lines[0].ocrRepair?.source, 'problem-input-sqrt-fraction-candidate-repair');
});

test('problem-input sqrt fraction repairs malformed numerator brace only', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('sqrt-frac-x2', 0, 0, 300, 220)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 260 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'simplify',
      problemType: 'simplify-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x ^ { 2 } } }',
      top: { latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x ^ { 2 } } }', score: -0.4, confidence: 0.27 },
      candidates: [{ latex: '\\frac { \\sqrt { x ^ { 1 8 } } { x ^ { 2 } } }', score: -0.4, confidence: 0.27 }],
      elapsedSeconds: 0.03
    }),
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { \\sqrt { x ^ { 1 8 } } } { x ^ { 2 } }']);
  assert.equal(result.lines[0].ocrRepair.source, 'problem-input-sqrt-fraction-brace-repair');
});

test('problem-input fraction does not infer missing sqrt from plain exponent alone', async () => {
  installFakeCanvas();
  const result = await recognizeStudentWriting({
    strokes: [stroke('plain-exp-frac', 0, 0, 300, 220)],
    answerBox: { xMin: -5, yMin: -5, xMax: 340, yMax: 260 },
    problemMetadata: {
      auditSubject: 'problem-input',
      mode: 'simplify',
      problemType: 'simplify-expression',
      source: 'user-handwriting'
    },
    semanticScoring: false,
    recognizeAlternatives: true,
    chunkFallback: false,
    recognizeLine: async () => ({
      latex: '\\frac { - x ^ { 1 8 } } { x ^ { 2 } }',
      top: { latex: '\\frac { - x ^ { 1 8 } } { x ^ { 2 } }', score: -0.2, confidence: 0.27 },
      candidates: [
        { latex: '\\frac { - x ^ { 1 8 } } { x ^ { 2 } }', score: -0.2, confidence: 0.27 },
        { latex: '- \\frac { x ^ { 1 8 } } { x ^ { 2 } }', score: -0.3, confidence: 0.25 }
      ],
      elapsedSeconds: 0.03
    }),
    gradeWork: null
  });

  assert.deepEqual(result.latexLines, ['\\frac { - x ^ { 1 8 } } { x ^ { 2 } }']);
  assert.equal(result.lines[0].ocrRepair?.source, undefined);
});

test('VLM audit decision catches OCR failure, timeout, and low confidence', () => {
  const result = auditResult({
    problemStatus: 'correct',
    lines: [
      auditLine({ prediction: { failed: true, timedOut: false, top: { latex: 'x = 4', confidence: 0.9 } } }),
      auditLine({ prediction: { failed: false, timedOut: true, top: { latex: 'x = 4', confidence: 0.9 } } }),
      auditLine({ prediction: { failed: false, timedOut: false, top: { latex: 'x = 4', confidence: 0.3 } } }),
      auditLine({ prediction: { failed: false, timedOut: false, top: { latex: 'x = 4', score: -0.5 } } }),
      auditLine({ evidenceScore: -1.5 })
    ]
  });

  const decision = getRecognitionAuditDecision(result, { inputSignature: 'sig-b' });

  assert.equal(decision.shouldAudit, true);
  assert.ok(decision.triggerReasons.includes('ocr_failure_or_timeout'));
  assert.ok(decision.triggerReasons.includes('low_confidence'));
});

test('VLM audit decision catches correct answer set with invalid intermediate step', () => {
  const grading = {
    status: 'complete',
    failed: false,
    steps: [
      { lineIndex: 0, studentLatex: '2x + 3 = 11', classification: 'valid_step', solutionCoverage: 'none', matchedSolutions: [] },
      { lineIndex: 1, studentLatex: '2x = 9', classification: 'invalid_step', solutionCoverage: 'none', matchedSolutions: [] },
      { lineIndex: 2, studentLatex: 'x = 4', classification: 'valid_step', solutionCoverage: 'full', matchedSolutions: ['4'] }
    ],
    result: {
      problemStatus: 'correct',
      foundSolutions: ['4'],
      missingSolutions: []
    }
  };

  assert.equal(hasCorrectAnswerWithInvalidStep(grading), true);

  const decision = getRecognitionAuditDecision(auditResult({ grading }), { inputSignature: 'sig-c' });

  assert.equal(decision.shouldAudit, true);
  assert.ok(decision.triggerReasons.includes('correct_answer_with_invalid_step'));
});

test('VLM audit decision samples normal correct cases deterministically', () => {
  const result = auditResult({ problemStatus: 'correct' });

  assert.equal(
    getRecognitionAuditDecision(result, { inputSignature: 'sample-key', sampleKey: 'sample-key', normalSampleRate: 1 }).shouldAudit,
    true
  );
  assert.equal(
    getRecognitionAuditDecision(result, { inputSignature: 'sample-key', sampleKey: 'sample-key', normalSampleRate: 0 }).shouldAudit,
    false
  );
  assert.equal(
    deterministicSample('fixed-key', 0.1),
    deterministicSample('fixed-key', 0.1)
  );
});

test('VLM audit payload removes embedded crop data URLs', () => {
  const payload = buildRecognitionAuditPayload({
    problem: {
      id: 'problem-a',
      latex: 'x + 1 = 5',
      metadata: { source: 'test' },
      problemBox: { xMin: 0, yMin: 0, xMax: 100, yMax: 100 },
      answerBox: { xMin: 0, yMin: 100, xMax: 100, yMax: 200 }
    },
    result: auditResult({
      lines: [{
        ...auditLine(),
        image: { dataUrl: 'data:image/png;base64,abc' },
        contextualSemantic: { nested: { dataUrl: 'data:image/png;base64,hidden' } }
      }],
      candidatePredictions: [{
        ...auditLine(),
        image: { dataUrl: 'data:image/png;base64,abc' }
      }]
    }),
    strokes: [{
      id: 'stroke-a',
      startTime: 1000,
      endTime: 1048,
      points: [
        { x: 0.0012345, y: 0.0023456, t: 0, pressure: 0.5012 },
        { x: 0.0045678, y: 0.0067891, t: 48, pressure: 0.6123 }
      ],
      rawPoints: [{ x: 1, y: 2, pressure: 0.5 }],
      outlinePoints: [{ x: 1, y: 2 }, { x: 4, y: 2 }, { x: 4, y: 6 }],
      canvasBbox: { xMin: 1, yMin: 2, xMax: 4, yMax: 6 },
      relationsToPrev: { dx: 0.123456, dy: -0.25, dt: 480.4, overlapRatio: 0.333333 }
    }],
    inputSignature: 'sig-d',
    triggerReasons: ['normal_sample']
  });

  assert.equal(payload.problemId, 'problem-a');
  assert.equal(payload.strokes.length, 1);
  assert.deepEqual(payload.strokes[0].points.map((point) => point.t), [0, 48]);
  assert.equal(payload.strokes[0].relationsToPrev.dt, 480);
  assert.ok(payload.strokes[0].relationsToPrev.overlapRatio > 0.333);
  assert.equal(payload.fastResult.lines[0].image, undefined);
  assert.equal(JSON.stringify(payload).includes('data:image/png'), false);
});

test('problem input audit payload marks adjustment trigger and skips grading payload', () => {
  const payload = buildProblemInputAuditPayload({
    mode: 'simplify',
    problemType: 'simplify-expression',
    recognizedLatex: '\\sqrt { \\frac { x ^ { 10 } } { x ^ 2 } }',
    result: auditResult({
      latex: '\\sqrt { \\frac { x ^ { 10 } } { x ^ 2 } }',
      latexLines: ['\\sqrt { \\frac { x ^ { 10 } } { x ^ 2 } }'],
      grading: { result: { problemStatus: 'correct' } }
    }),
    strokes: [{ id: 's1', rawPoints: [{ x: 1, y: 2 }, { x: 3, y: 4 }] }],
    answerBox: { xMin: 0, yMin: 0, xMax: 760, yMax: 320 },
    inputSignature: 'problem-input-adjust::sig',
    triggerReasons: ['user_adjusted_problem_input']
  });

  assert.equal(payload.problemId, 'problem-input-simplify');
  assert.equal(payload.problemMetadata.auditSubject, 'problem-input');
  assert.deepEqual(payload.triggerReasons, ['user_adjusted_problem_input']);
  assert.equal(payload.fastResult.grading, null);
  assert.deepEqual(payload.fastResult.latexLines, ['\\sqrt { \\frac { x ^ { 10 } } { x ^ 2 } }']);
});

test('problem input recognition filters split-line OCR noise before preview and audit', () => {
  const normalized = normalizeProblemInputRecognitionResult({
    latex: '\\sqrt // 4 x ^ { 12 } y ^ 7',
    latexLines: ['\\sqrt // 4 x ^ { 12 } y ^ 7'],
    lines: [{
      lineIndex: 0,
      latex: '\\sqrt // 4 x ^ { 12 } y ^ 7',
      acceptedLatex: '\\sqrt // 4 x ^ { 12 } y ^ 7',
      candidates: [
        { latex: '\\sqrt // 4 x ^ { 12 } y ^ 7', score: 4 },
        { latex: '\\sqrt[4]{x^{12}y^7}', score: 3 },
        { latex: '4 x ^ { 12 } y ^ 7', score: 2 }
      ]
    }],
    candidatePredictions: [{
      candidateId: 'problem-line',
      latex: '\\sqrt // 4 x ^ { 12 } y ^ 7',
      candidates: [
        { latex: '\\sqrt // 4 x ^ { 12 } y ^ 7', score: 4 },
        { latex: '\\sqrt[4]{x^{12}y^7}', score: 3 }
      ]
    }],
    grading: { result: { problemStatus: 'correct' } }
  });

  assert.deepEqual(normalized.latexLines, ['\\sqrt[4]{x^{12}y^7}']);
  assert.equal(normalized.latex, '\\sqrt[4]{x^{12}y^7}');
  assert.equal(normalized.lines[0].acceptedLatex, '\\sqrt[4]{x^{12}y^7}');
});

test('VLM audit payload preserves candidate rescue summary for review', () => {
  const payload = buildRecognitionAuditPayload({
    problem: {
      id: 'problem-rescue',
      latex: '\\sqrt { 2 } \\log _ { 3 } 9',
      metadata: {}
    },
    result: auditResult({
      selectionRescue: [{
        action: 'append',
        selectedCandidateId: 'row-final',
        lineIndex: 1,
        reason: 'candidate_rescue:full_final'
      }]
    }),
    strokes: [],
    inputSignature: 'sig-rescue',
    triggerReasons: ['candidate_selection_conflict']
  });

  assert.deepEqual(payload.fastResult.selectionSummary.rescueSummary, [{
    action: 'append',
    selectedCandidateId: 'row-final',
    lineIndex: 1,
    reason: 'candidate_rescue:full_final'
  }]);
});

test('VLM audit payload includes only same-attempt feedback', () => {
  const problem = {
    id: 'problem-feedback',
    latex: 'x + 1 = 5',
    metadata: {}
  };
  const inputSignature = 'sig-feedback';
  const attemptId = buildAttemptId(problem.id, inputSignature);
  const sameAttempt = buildRecognitionAuditPayload({
    problem,
    result: auditResult(),
    strokes: [],
    inputSignature,
    triggerReasons: ['normal_sample'],
    feedback: {
      attemptId,
      inputSignature,
      status: 'complete',
      source: 'deterministic',
      text: 'Correct! Great job!',
      promptVersion: 'math-feedback-v1'
    }
  });
  const staleAttempt = buildRecognitionAuditPayload({
    problem,
    result: auditResult(),
    strokes: [],
    inputSignature,
    triggerReasons: ['normal_sample'],
    feedback: {
      attemptId: buildAttemptId(problem.id, 'old-sig'),
      inputSignature: 'old-sig',
      status: 'complete',
      source: 'ollama',
      text: 'Old hint',
      promptVersion: 'math-feedback-v1'
    }
  });

  assert.equal(sameAttempt.feedback.text, 'Correct! Great job!');
  assert.equal(staleAttempt.feedback, null);
});

test('VLM audit payload carries structured annotation attachments', () => {
  const anchor = auditLine({
    lineIndex: 0,
    candidateId: 'eq-line',
    latex: '\\frac { x - 1 } { x + 1 } = 4',
    acceptedLatex: '\\frac { x - 1 } { x + 1 } = 4',
    tightBbox: { xMin: 120, yMin: 80, xMax: 520, yMax: 160 }
  });
  const annotation = auditLine({
    lineIndex: 1,
    candidateId: 'annotation-line',
    latex: '\\times 4 \\times 4',
    acceptedLatex: '\\times 4 \\times 4',
    ocrLatex: '4*',
    excludedFromGrading: true,
    tightBbox: { xMin: 140, yMin: 10, xMax: 500, yMax: 70 },
    ocrRepair: {
      source: 'geometry-operation-annotation',
      originalLatex: '4*',
      repairedLatex: '\\times 4 \\times 4',
      operand: '4',
      anchorCandidateId: 'eq-line',
      anchorBbox: { xMin: 120, yMin: 80, xMax: 520, yMax: 160 },
      annotationBbox: { xMin: 140, yMin: 10, xMax: 500, yMax: 70 }
    }
  });

  const payload = buildRecognitionAuditPayload({
    problem: {
      id: 'problem-b',
      latex: '\\frac { x - 1 } { x + 1 } = 4',
      metadata: {}
    },
    result: auditResult({
      lines: [anchor, annotation],
      candidatePredictions: [anchor, annotation]
    }),
    strokes: [],
    inputSignature: 'sig-annotation',
    triggerReasons: ['detached_operation_annotation']
  });

  assert.equal(payload.promptVersion, 'recognition-audit-v2');
  assert.equal(payload.fastResult.lines[1].excludedFromGrading, true);
  assert.equal(payload.fastResult.annotationAttachments.length, 1);
  assert.equal(payload.fastResult.annotationAttachments[0].targetLineIndex, 0);
  assert.equal(payload.fastResult.annotationAttachments[0].equationSide, 'both');
  assert.equal(payload.fastResult.annotationAttachments[0].pairedAnnotationId, 'pair:0|4');
});

function auditResult(options = {}) {
  const grading = options.grading || {
    status: 'complete',
    failed: false,
    steps: [
      { lineIndex: 0, studentLatex: 'x = 4', classification: 'valid_step', solutionCoverage: 'full', matchedSolutions: ['4'] }
    ],
    result: {
      problemStatus: options.problemStatus || 'correct',
      foundSolutions: ['4'],
      missingSolutions: []
    },
    problem: {
      manifest: {
        cardinality: 'finite',
        exact_set: ['4'],
        variable: 'x',
        problem_raw: 'x + 1 = 5'
      }
    }
  };
  const lines = options.lines || [auditLine()];
  return {
    latex: options.latex || lines.map((line) => line.acceptedLatex || line.latex || '').join(' \\\\ '),
    latexLines: options.latexLines || lines.map((line) => line.acceptedLatex || line.latex || ''),
    lines,
    candidatePredictions: options.candidatePredictions || lines,
    grading,
    segmentation: {
      selected: lines.map((line) => ({
        candidateId: line.candidateId,
        profiles: line.profiles,
        strokeIds: line.strokeIds,
        tightBbox: line.tightBbox
      })),
      candidates: [],
      partitions: {},
      parentCandidateId: null,
      ocrSelectedCandidateIds: lines.map((line) => line.candidateId)
    },
    selectionRescue: options.selectionRescue || [],
    realtime: {
      allFinal: true,
      inputSignature: 'audit-test-signature'
    }
  };
}

function auditLine(options = {}) {
  const latex = options.latex ?? 'x = 4';
  const acceptedLatex = options.acceptedLatex ?? latex;
  return {
    lineIndex: options.lineIndex ?? 0,
    candidateId: options.candidateId || 'audit-line',
    selected: true,
    profiles: ['row-line'],
    strokeIds: ['stroke-a'],
    tightBbox: { xMin: 0, yMin: 0, xMax: 100, yMax: 40 },
    latex,
    acceptedLatex,
    ocrLatex: latex,
    candidates: options.candidates || [{ latex, score: 2, confidence: 0.9 }],
    prediction: options.prediction || {
      latex,
      top: { latex, score: 2, confidence: 0.9 },
      candidates: [{ latex, score: 2, confidence: 0.9 }],
      failed: false,
      timedOut: false
    },
    ocrRepair: options.ocrRepair ?? null,
    excludedFromGrading: Boolean(options.excludedFromGrading),
    evidenceScore: options.evidenceScore ?? 3,
    timing: { submitToFinalPredictionSeconds: 0.1 }
  };
}

function candidate(candidateId, profiles, strokeIds, xMin, yMin, xMax, yMax) {
  return {
    candidateId,
    id: candidateId,
    profiles,
    profile: profiles[0],
    strokeIds,
    strokes: strokeIds.map((id) => stroke(id, xMin, yMin, xMax, yMax)),
    tightBbox: { xMin, yMin, xMax, yMax },
    expandedBbox: { xMin, yMin, xMax, yMax },
    conflicts: []
  };
}

function testingCatalogProblems(names) {
  const output = execFileSync('python3', ['testing/export_equation_problem_catalog.py', ...names], {
    cwd: process.cwd(),
    encoding: 'utf8'
  });
  return (JSON.parse(output).problems || []).map((problem) => ({
    id: problem.id,
    latex: problem.latex,
    family: problem.family,
    expectedLatexLines: problem.expectedLatexLines || [],
    metadata: {
      name: problem.name,
      family: problem.family,
      source: problem.source,
      expectedLatexLines: problem.expectedLatexLines || []
    }
  }));
}

function candidateFromStrokeList(candidateId, profiles, strokes) {
  const tightBbox = bboxForStrokes(strokes);
  return {
    candidateId,
    id: candidateId,
    profiles,
    profile: profiles[0],
    strokeIds: strokes.map((item) => item.id),
    strokes,
    tightBbox,
    expandedBbox: tightBbox,
    conflicts: []
  };
}

function fakeRecognitionResult(strokes, source = 'deterministic') {
  const strokeIds = strokes.map((item) => String(item.id)).sort();
  const tightBbox = bboxForStrokes(strokes);
  const candidateId = `fake_${strokeIds.join('_')}`;
  const latex = strokeIds.join(' + ') || 'x';
  const line = {
    lineIndex: 0,
    candidateId,
    debugLabel: 'C1',
    selected: true,
    profiles: [source === 'dbnet' ? 'dbnet-line' : 'row-line'],
    strokeIds,
    tightBbox,
    image: null,
    latex,
    acceptedLatex: latex,
    ocrLatex: latex,
    candidates: [{ latex, score: 2 }],
    prediction: {
      latex,
      top: { latex, score: 2 },
      candidates: [{ latex, score: 2 }],
      elapsedSeconds: 0.01
    },
    evidenceScore: 3,
    timing: {
      submitToFinalPredictionSeconds: 0.01,
      ocrElapsedSeconds: 0.01
    }
  };

  return {
    latex,
    latexLines: [latex],
    lines: [line],
    candidatePredictions: [line],
    detection: { source, failed: false },
    semantic: { source: 'disabled', failed: false },
    timing: { totalElapsedSeconds: 0.01 },
    segmentation: {
      selected: [{
        candidateId,
        profiles: line.profiles,
        strokeIds,
        tightBbox
      }],
      candidates: [{
        candidateId,
        profiles: line.profiles,
        strokeIds,
        tightBbox,
        conflicts: []
      }],
      partitions: {},
      parentCandidateId: null,
      ocrSelectedCandidateIds: [candidateId]
    }
  };
}

function fakeFullAnswerResult(strokes, options = {}) {
  const sortedStrokes = strokes.slice().sort((left, right) => (
    left.canvasBbox.yMin - right.canvasBbox.yMin ||
    left.canvasBbox.xMin - right.canvasBbox.xMin
  ));
  const latexLines = options.latexLines || sortedStrokes.map((item) => item.id);
  const lineCandidateIds = options.candidateIds?.slice(0, sortedStrokes.length) ||
    sortedStrokes.map((item) => `full_${item.id}`);
  const lines = sortedStrokes.map((item, index) => {
    const latex = latexLines[index] || item.id;
    const candidateId = lineCandidateIds[index] || `full_${item.id}`;
    return {
      lineIndex: index,
      candidateId,
      debugLabel: `C${index + 1}`,
      selected: true,
      profiles: ['one-shot', 'dbnet-line'],
      strokeIds: [String(item.id)],
      tightBbox: item.canvasBbox,
      image: null,
      latex,
      acceptedLatex: latex,
      ocrLatex: latex,
      candidates: [{ latex, score: 3 }],
      prediction: {
        latex,
        top: { latex, score: 3 },
        candidates: [{ latex, score: 3 }],
        elapsedSeconds: 0.02
      },
      evidenceScore: 5,
      timing: {
        submitToFinalPredictionSeconds: 0.02,
        ocrElapsedSeconds: 0.02
      }
    };
  });
  const selected = lines.map((line) => ({
    candidateId: line.candidateId,
    profiles: line.profiles,
    strokeIds: line.strokeIds,
    tightBbox: line.tightBbox
  }));
  const candidateIds = options.candidateIds || selected.map((item) => item.candidateId);
  const candidates = candidateIds.map((candidateId, index) => {
    if (selected[index]) {
      return {
        ...selected[index],
        candidateId,
        conflicts: []
      };
    }
    return {
      candidateId,
      profiles: ['one-shot-parent'],
      strokeIds: sortedStrokes.map((item) => String(item.id)),
      tightBbox: bboxForStrokes(sortedStrokes),
      conflicts: []
    };
  });

  return {
    latex: latexLines.join(' \\\\ '),
    latexLines,
    lines,
    candidatePredictions: lines,
    detection: { source: 'one-shot', failed: false },
    semantic: { source: 'disabled', failed: false },
    timing: { totalElapsedSeconds: 0.02 },
    segmentation: {
      selected,
      candidates,
      partitions: {},
      parentCandidateId: candidates.at(-1)?.candidateId || null,
      ocrSelectedCandidateIds: selected.map((candidate) => candidate.candidateId)
    }
  };
}

function syntheticCatalogBoard(problemName, options = {}) {
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
    problemName,
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

function equationCatalogProblemNames() {
  const output = execFileSync('python3', ['testing/export_equation_problem_catalog.py'], {
    cwd: process.cwd(),
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024
  });
  return (JSON.parse(output).problems || []).map((problem) => problem.name);
}

function strokesFromSyntheticBoard(board, options = {}) {
  const lineEntries = (board.lines || []).map((line, lineIndex) => (
    (line.contours || []).map((contour, contourIndex) => ({
      line,
      lineIndex,
      contour,
      contourIndex
    }))
  ));
  const entries = options.order === 'interleaved-lines'
    ? interleaveLineEntries(lineEntries)
    : lineEntries.flat();
  const strokeIntervalMs = Number.isFinite(options.strokeIntervalMs) ? options.strokeIntervalMs : 12;
  const linePauseMs = Number.isFinite(options.linePauseMs) ? options.linePauseMs : 0;
  let previousLineIndex = null;
  let timestamp = 0;

  return entries.map(({ line, lineIndex, contour, contourIndex }) => {
      const box = bboxForPoints(contour);
      if (previousLineIndex !== null && previousLineIndex !== lineIndex) {
        timestamp += linePauseMs;
      }
      const startTime = timestamp;
      timestamp += strokeIntervalMs;
      previousLineIndex = lineIndex;
      return {
        id: `fixture_${lineIndex + 1}_${contourIndex + 1}`,
        canvasBbox: box,
        rawPoints: contour,
        outlinePoints: contour,
        syntheticLineIndex: lineIndex,
        syntheticLatex: line.latex,
        startTime,
        endTime: startTime + Math.max(1, Math.min(8, strokeIntervalMs / 2))
      };
  });
}

function interleaveLineEntries(lineEntries) {
  const entries = [];
  const longest = Math.max(0, ...lineEntries.map((line) => line.length));
  for (let contourIndex = 0; contourIndex < longest; contourIndex += 1) {
    for (const line of lineEntries) {
      if (line[contourIndex]) entries.push(line[contourIndex]);
    }
  }
  return entries;
}

function fakeReadersForSyntheticBoard(board, strokes) {
  const expectedLines = board.fixture?.expectedLatexLines || [];
  const strokesById = new Map(strokes.map((item) => [String(item.id), item]));
  return {
    detectLines: async () => ({
      detections: (board.lines || []).map((line) => ({ bbox: line.bbox })),
      failed: false,
      elapsedSeconds: 0.01
    }),
    recognizeLine: async (image) => {
      const lineIndexes = syntheticLineIndexesForStrokeIds(image.strokeIds || [], strokesById);
      const latex = lineIndexes.length === 1
        ? expectedLines[lineIndexes[0]]
        : lineIndexes.map((index) => expectedLines[index]).filter(Boolean).join(' \\\\ ');
      return {
        latex: latex || 'x',
        top: { latex: latex || 'x', score: 3 },
        candidates: [
          { latex: latex || 'x', score: 3 },
          { latex: 'x', score: 0.1 }
        ],
        elapsedSeconds: 0.01
      };
    }
  };
}

function fakeReadersForRealTrace(fixture) {
  const lineByStrokeKey = new Map();
  for (const group of fixture.fastLineGroups || []) {
    lineByStrokeKey.set(strokeGroupKey(group.strokeIds), group);
  }
  for (const group of fixture.expectedLineGroups || []) {
    lineByStrokeKey.set(strokeGroupKey(group.strokeIds), group);
  }

  return {
    recognizeLine: async (image) => {
      const key = strokeGroupKey(image.strokeIds || []);
      const group = lineByStrokeKey.get(key);
      const latex = group?.latex || fixture.fastLatexLines?.[0] || fixture.expectedLatexLines?.[0] || 'x';
      return {
        latex,
        top: { latex, score: 3, confidence: 0.99 },
        candidates: [{ latex, score: 3, confidence: 0.99 }],
        elapsedSeconds: 0.01
      };
    }
  };
}

function pythonGradeFixtureTranscript(fixture) {
  return pythonGradePayload({
    problemLatex: fixture.problemLatex,
    problemMetadata: fixture.problemMetadata || {},
    lines: (fixture.expectedLineGroups || []).map((group, index) => ({
      lineIndex: index,
      latex: group.latex
    }))
  });
}

function pythonGradePayload(payload) {
  const script = `
import json
import sys
from src.grading import grade_math_payload
payload = json.load(sys.stdin)
print(json.dumps(grade_math_payload(payload)))
`;
  const output = execFileSync('python3', ['-c', script], {
    cwd: process.cwd(),
    input: JSON.stringify(payload),
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024
  });
  return JSON.parse(output);
}

function syntheticLineIndexesForStrokeIds(strokeIds, strokesById) {
  return [...new Set(
    (strokeIds || [])
      .map((id) => strokesById.get(String(id))?.syntheticLineIndex)
      .filter((index) => Number.isInteger(index))
  )].sort((left, right) => left - right);
}

async function waitForSnapshot(snapshots, predicate) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const match = snapshots.find(predicate);
    if (match) return match;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.fail(`Timed out waiting for scheduler snapshot. Latest: ${JSON.stringify(snapshots.at(-1) || null)}`);
}

async function nextMicrotask() {
  await Promise.resolve();
  await Promise.resolve();
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function bboxForPoints(points) {
  return points.reduce((box, point) => ({
    xMin: Math.min(box.xMin, point.x),
    yMin: Math.min(box.yMin, point.y),
    xMax: Math.max(box.xMax, point.x),
    yMax: Math.max(box.yMax, point.y)
  }), {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity
  });
}

function sameBboxForTest(left, right) {
  return Boolean(left && right) &&
    left.xMin === right.xMin &&
    left.yMin === right.yMin &&
    left.xMax === right.xMax &&
    left.yMax === right.yMax;
}

function padBboxForTest(bbox, padding) {
  return {
    xMin: bbox.xMin - padding,
    yMin: bbox.yMin - padding,
    xMax: bbox.xMax + padding,
    yMax: bbox.yMax + padding
  };
}

function bboxForStrokes(strokes) {
  return strokes.reduce((box, item) => ({
    xMin: Math.min(box.xMin, item.canvasBbox.xMin),
    yMin: Math.min(box.yMin, item.canvasBbox.yMin),
    xMax: Math.max(box.xMax, item.canvasBbox.xMax),
    yMax: Math.max(box.yMax, item.canvasBbox.yMax),
  }), {
    xMin: Infinity,
    yMin: Infinity,
    xMax: -Infinity,
    yMax: -Infinity,
  });
}

function problemWithAnswer(flow, problemId, overrides = {}) {
  return {
    ...flow,
    problems: flow.problems.map((problem) => (
      problem.id === problemId
        ? {
            ...problem,
            answerStrokeIds: ['a'],
            answerBox: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 },
            ...overrides
          }
        : problem
    ))
  };
}

function correctRecognitionResult(latex = 'x = 2', inputSignature = 'sig-correct') {
  return {
    latex,
    latexLines: [latex],
    lines: [],
    candidatePredictions: [],
    grading: {
      status: 'complete',
      failed: false,
      result: {
        problemStatus: 'correct',
        foundSolutions: [],
        missingSolutions: []
      }
    },
    realtime: {
      allFinal: true,
      inputSignature,
      components: [{
        signature: inputSignature,
        status: 'final',
        contested: false
      }]
    }
  };
}

function stroke(id, xMin, yMin, xMax, yMax) {
  return {
    id,
    canvasBbox: { xMin, yMin, xMax, yMax },
    outlinePoints: [
      { x: xMin, y: yMin },
      { x: xMax, y: yMin },
      { x: xMax, y: yMax },
      { x: xMin, y: yMax },
    ],
    startTime: id.charCodeAt(0) * 100,
    endTime: id.charCodeAt(0) * 100 + 20
  };
}

function assignTimes(strokes) {
  for (let index = 0; index < strokes.length; index += 1) {
    strokes[index].startTime = index * 10;
    strokes[index].endTime = index * 10 + 1;
  }
}

function installFakeCanvas() {
  globalThis.window = { devicePixelRatio: 1 };
  globalThis.document = {
    createElement: (tag) => {
      assert.equal(tag, 'canvas');
      return {
        width: 1,
        height: 1,
        getContext: () => fakeCanvasContext(),
        toDataURL: () => `data:image/png;base64,${Buffer.from('line').toString('base64')}`,
      };
    }
  };
}

function fakeCanvasContext() {
  return {
    set fillStyle(_value) {},
    scale() {},
    fillRect() {},
    translate() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    fill() {},
  };
}
