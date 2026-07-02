#!/usr/bin/env node

import process from 'node:process';
import { segmentMathLines, selectCandidateCover } from '../src/recognition/lineSegmentation.js';

const order = process.argv[2] || 'line-order';
const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}

const board = JSON.parse(Buffer.concat(chunks).toString('utf8'));
const strokes = strokesForBoard(board, order);
const result = segmentMathLines(strokes, {
  answerBox: board.answerBox || null,
  detections: board.detections || [],
  ignoredStrokeIds: board.visualOnlyStrokeIds || []
});
const scoreByCandidateId = board.scoreByCandidateId || null;
const rescoredSelected = scoreByCandidateId
  ? selectCandidateCover(result.candidates, {
      scoreByCandidateId,
      baselineCandidates: result.selected
    })
  : null;

const payload = {
  order,
  strokeCount: strokes.length,
  expectedLines: expectedLineCount(board),
  candidateCount: result.candidates.length,
  selectedCount: result.selected.length,
  partitions: Object.fromEntries(
    Object.entries(result.partitions).map(([key, values]) => [key, values.length])
  ),
  selected: result.selected.map(summarizeCandidate),
  candidates: result.candidates.map(summarizeCandidate),
  rescoredSelectedCount: rescoredSelected ? rescoredSelected.length : null,
  rescoredSelected: rescoredSelected ? rescoredSelected.map(summarizeCandidate) : null
};

process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);

function strokesForBoard(board, mode) {
  if (Array.isArray(board.strokes)) {
    return directStrokesForOrder(board.strokes, mode);
  }
  return contoursForOrder(board.lines || [], mode);
}

function directStrokesForOrder(strokes, mode) {
  const ordered = strokes
    .filter((stroke) => stroke?.canvasBbox)
    .map((stroke) => ({
      ...stroke,
      rawPoints: stroke.rawPoints || [],
      outlinePoints: stroke.outlinePoints || stroke.rawPoints || []
    }));

  if (mode === 'reverse-lines') {
    return ordered.slice().reverse();
  }
  if (mode === 'interleaved-lines' && ordered.some((stroke) => Number.isInteger(stroke.expectedLineIndex))) {
    return interleaveExpectedLineStrokes(ordered);
  }
  return ordered.slice().sort((a, b) => strokeTime(a) - strokeTime(b));
}

function interleaveExpectedLineStrokes(strokes) {
  const byLine = new Map();
  for (const stroke of strokes) {
    const lineIndex = Number.isInteger(stroke.expectedLineIndex) ? stroke.expectedLineIndex : -1;
    if (!byLine.has(lineIndex)) byLine.set(lineIndex, []);
    byLine.get(lineIndex).push(stroke);
  }
  for (const values of byLine.values()) {
    values.sort((a, b) => strokeTime(a) - strokeTime(b));
  }
  const lineIndexes = [...byLine.keys()].sort((a, b) => a - b);
  const maxLength = Math.max(0, ...[...byLine.values()].map((values) => values.length));
  const ordered = [];
  for (let strokeIndex = 0; strokeIndex < maxLength; strokeIndex += 1) {
    for (const lineIndex of lineIndexes) {
      const stroke = byLine.get(lineIndex)?.[strokeIndex];
      if (stroke) ordered.push(stroke);
    }
  }
  return ordered;
}

function expectedLineCount(board) {
  if (Array.isArray(board.expectedLineGroups)) return board.expectedLineGroups.length;
  if (Array.isArray(board.expectedLatexLines)) return board.expectedLatexLines.length;
  return (board.lines || []).length;
}

function contoursForOrder(lines, mode) {
  const items = orderedContours(lines, mode);
  return items.map((item, index) => {
    const box = bbox(item.contour);
    const startTime = index * 120;
    return {
      id: `fixture_${index + 1}`,
      canvasBbox: box,
      rawPoints: item.contour,
      outlinePoints: item.contour,
      syntheticLineIndex: item.lineIndex,
      syntheticLatex: lines[item.lineIndex]?.latex || null,
      startTime,
      endTime: startTime + 20
    };
  });
}

function orderedContours(lines, mode) {
  if (mode === 'line-order') {
    return lines.flatMap((line, lineIndex) => (
      (line.contours || []).map((contour) => ({ contour, lineIndex }))
    ));
  }
  if (mode === 'reverse-lines') {
    return lines.slice().reverse().flatMap((line, reverseIndex) => {
      const lineIndex = lines.length - 1 - reverseIndex;
      return (line.contours || []).map((contour) => ({ contour, lineIndex }));
    });
  }
  if (mode === 'interleaved-lines') {
    const maxContours = Math.max(0, ...lines.map((line) => (line.contours || []).length));
    const ordered = [];
    for (let contourIndex = 0; contourIndex < maxContours; contourIndex += 1) {
      for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
        const contour = lines[lineIndex].contours?.[contourIndex];
        if (contour) ordered.push({ contour, lineIndex });
      }
    }
    return ordered;
  }
  throw new Error(`Unknown order mode: ${mode}`);
}

function bbox(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    xMin: Math.min(...xs),
    yMin: Math.min(...ys),
    xMax: Math.max(...xs),
    yMax: Math.max(...ys)
  };
}

function summarizeCandidate(candidate) {
  const syntheticLineIndexes = [...new Set(
    (candidate.strokes || [])
      .map((stroke) => stroke.syntheticLineIndex)
      .filter((value) => value !== undefined && value !== null)
  )].sort((a, b) => a - b);
  const expectedLineIndexes = [...new Set(
    (candidate.strokes || [])
      .map((stroke) => stroke.expectedLineIndex)
      .filter((value) => value !== undefined && value !== null)
  )].sort((a, b) => a - b);

  return {
    candidateId: candidate.candidateId,
    profiles: candidate.profiles,
    strokeIds: candidate.strokeIds,
    strokeCount: candidate.strokeIds.length,
    strokes: (candidate.strokes || []).map(summarizeStroke),
    bbox: candidate.tightBbox,
    syntheticLineSets: syntheticLineIndexes,
    syntheticLatex: syntheticLineIndexes.map((index) => candidate.strokes.find((stroke) => (
      stroke.syntheticLineIndex === index
    ))?.syntheticLatex || null),
    expectedLineSets: expectedLineIndexes,
    expectedLatex: expectedLineIndexes.map((index) => candidate.strokes.find((stroke) => (
      stroke.expectedLineIndex === index
    ))?.expectedLatex || null)
  };
}

function summarizeStroke(stroke) {
  return {
    id: stroke.id,
    canvasBbox: stroke.canvasBbox,
    rawPoints: stroke.rawPoints || stroke.outlinePoints || [],
    syntheticLineIndex: stroke.syntheticLineIndex,
    syntheticLatex: stroke.syntheticLatex || null,
    expectedLineIndex: stroke.expectedLineIndex ?? null,
    expectedLatex: stroke.expectedLatex || null,
  };
}

function strokeTime(stroke) {
  const start = Number(stroke?.startTime);
  return Number.isFinite(start) ? start : 0;
}
