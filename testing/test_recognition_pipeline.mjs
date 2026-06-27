#!/usr/bin/env node

import assert from 'node:assert/strict';
import test from 'node:test';
import { translateDetections } from '../src/recognition/segmentationClient.js';
import { previousLatexForSubmission, summarizeRecognitionResult } from '../src/hooks/useProblemFlowController.js';
import { recognizeStudentWriting, shouldUseSemanticLatex } from '../src/recognition/studentWritingPipeline.js';
import {
  scoreRecognitionEvidence,
  segmentMathLines,
  selectCandidateCover,
  strokeBelongsToAnswerBox
} from '../src/recognition/lineSegmentation.js';
import {
  applyProblemRecognitionError,
  applyProblemRecognitionResult,
  createInitialProblemFlow,
  getActiveProblem,
  submitActiveProblem
} from '../src/state/problemFlow.js';
import { TEST_PROBLEMS } from '../src/state/problemFixtures.js';

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
  assert.ok(result.lines[0].prediction.chunkAttempts.filter((attempt) => attempt.fractionSubchunk).length >= 2);
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
      const bestLatex = request.previousLatex.includes('\\eta = 5')
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
      const hasEtaContext = request.previousLatex.includes('\\eta = 5');
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

test('submitted problem flow preserves recognition status and result', () => {
  const initial = createInitialProblemFlow(1200);
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

test('temporary problem fixtures cover linear rational and logarithmic equations', () => {
  assert.equal(TEST_PROBLEMS.length, 3);
  assert.deepEqual(TEST_PROBLEMS.map((problem) => problem.id), ['problem-1', 'problem-2', 'problem-3']);
  assert.ok(TEST_PROBLEMS.every((problem) => problem.latex.includes('=')));
  assert.equal(TEST_PROBLEMS.some((problem) => problem.latex === '2x + 3 = 11'), true);
  assert.equal(TEST_PROBLEMS.some((problem) => problem.latex.includes('\\frac')), true);
  assert.equal(TEST_PROBLEMS.some((problem) => problem.latex.includes('\\log')), true);
});

test('recognition context does not leak previous problem latex into new submissions', () => {
  const initial = createInitialProblemFlow(1200);
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
  const completedFirst = applyProblemRecognitionResult(afterFirstSubmit, firstProblem.id, {
    latex: '\\eta = 5',
    latexLines: ['\\eta = 5'],
    lines: []
  });
  const nextProblem = getActiveProblem(completedFirst);

  assert.deepEqual(previousLatexForSubmission(completedFirst, nextProblem.id), []);
});

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
