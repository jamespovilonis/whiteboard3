import katex from 'katex';
import { Fragment, useMemo } from 'react';
import { boardToScreen } from '../whiteboard/viewport.js';

export default function ProblemLayer({ problems, viewport, debugBoxesEnabled }) {
  return (
    <>
      {problems.map((problem) => (
        <Fragment key={problem.id}>
          <ProblemPrint
            problem={problem}
            viewport={viewport}
          />
          {debugBoxesEnabled && (
            <ProblemDebugBoxes
              problem={problem}
              viewport={viewport}
            />
          )}
        </Fragment>
      ))}
    </>
  );
}

function ProblemPrint({ problem, viewport }) {
  const screenPosition = boardToScreen(problem.boardPosition, viewport);
  const html = useMemo(() => {
    try {
      return katex.renderToString(problem.latex, {
        throwOnError: false,
        displayMode: true
      });
    } catch {
      return problem.latex;
    }
  }, [problem.latex]);

  return (
    <div
      className="problem-print"
      data-problem-id={problem.id}
      data-problem-status={problem.status}
      style={{
        transform: `translate3d(${screenPosition.x}px, ${screenPosition.y}px, 0)`
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function ProblemDebugBoxes({ problem, viewport }) {
  return (
    <>
      <DebugBox
        bbox={problem.problemBox}
        kind="problem"
        problemId={problem.id}
        viewport={viewport}
      />
      <DebugBox
        bbox={problem.answerBox}
        kind={problem.answerBoxFrozen ? 'answer-frozen' : 'answer'}
        problemId={problem.id}
        viewport={viewport}
      />
      <CandidateDebugBoxes
        problem={problem}
        viewport={viewport}
      />
    </>
  );
}

function CandidateDebugBoxes({ problem, viewport }) {
  const candidates = problem.recognition?.result?.candidatePredictions || [];
  if (candidates.length === 0) return null;
  const labelOffsets = buildCandidateLabelOffsets(candidates, viewport);

  return (
    <>
      {candidates.map((candidate, index) => (
        <DebugBox
          key={`${problem.id}-${candidate.candidateId}`}
          bbox={candidate.tightBbox}
          kind={`candidate-${candidateDebugState(candidate)}`}
          label={candidateDebugLabel(candidate, index)}
          labelOffset={labelOffsets.get(candidate.candidateId)}
          problemId={problem.id}
          viewport={viewport}
        />
      ))}
    </>
  );
}

function DebugBox({ bbox, kind, label, labelOffset, problemId, viewport }) {
  if (!bbox) return null;

  const topLeft = boardToScreen({ x: bbox.xMin, y: bbox.yMin }, viewport);
  const width = Math.max(1, (bbox.xMax - bbox.xMin) * viewport.scale);
  const height = Math.max(1, (bbox.yMax - bbox.yMin) * viewport.scale);

  return (
    <div
      className={`debug-board-box debug-board-box-${kind}`}
      data-box-kind={kind}
      data-problem-id={problemId}
      style={{
        width,
        height,
        transform: `translate3d(${topLeft.x}px, ${topLeft.y}px, 0)`
      }}
    >
      {label && (
        <span
          className="debug-board-box-label"
          style={{
            transform: labelOffset
              ? `translate3d(${labelOffset.x}px, ${labelOffset.y}px, 0)`
              : undefined
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

function buildCandidateLabelOffsets(candidates, viewport) {
  const placed = [];
  const offsets = new Map();
  const sorted = candidates
    .map((candidate, index) => ({ candidate, index }))
    .filter(({ candidate }) => candidate.tightBbox)
    .sort((a, b) => (
      (a.candidate.tightBbox.yMin - b.candidate.tightBbox.yMin) ||
      (a.candidate.tightBbox.xMin - b.candidate.tightBbox.xMin) ||
      (a.index - b.index)
    ));

  for (const { candidate } of sorted) {
    const topLeft = boardToScreen({
      x: candidate.tightBbox.xMin,
      y: candidate.tightBbox.yMin
    }, viewport);
    const labelWidth = estimateLabelWidth(candidate);
    let lane = 0;
    let box = labelBoxFor(topLeft, labelWidth, lane);

    while (placed.some((item) => boxesOverlap(item, box)) && lane < 12) {
      lane += 1;
      box = labelBoxFor(topLeft, labelWidth, lane);
    }

    placed.push(box);
    offsets.set(candidate.candidateId, {
      x: 0,
      y: labelYOffsetForLane(lane)
    });
  }

  return offsets;
}

function labelBoxFor(topLeft, width, lane) {
  const height = 22;
  const xMin = topLeft.x - 2;
  const yMin = topLeft.y + labelYOffsetForLane(lane);
  return {
    xMin,
    yMin,
    xMax: xMin + width,
    yMax: yMin + height
  };
}

function labelYOffsetForLane(lane) {
  if (lane === 0) return -26;
  return 4 + (lane - 1) * 24;
}

function boxesOverlap(a, b) {
  const gap = 4;
  return a.xMin < b.xMax + gap &&
    a.xMax + gap > b.xMin &&
    a.yMin < b.yMax + gap &&
    a.yMax + gap > b.yMin;
}

function estimateLabelWidth(candidate) {
  const label = candidateDebugLabel(candidate, 0);
  return Math.min(180, Math.max(64, label.length * 7 + 16));
}

function candidateDebugState(candidate) {
  if (candidate.prediction?.failed || candidate.prediction?.timedOut) return 'failed';
  if (candidate.selected) return 'selected';
  if (!candidate.latex) return 'unread';
  return 'discarded';
}

function candidateDebugLabel(candidate, index) {
  const label = candidate.debugLabel || `C${index + 1}`;
  if (candidate.selected) return `${label} selected`;
  if (candidate.prediction?.failed || candidate.prediction?.timedOut) return `${label} failed`;
  if (!candidate.latex) return `${label} not OCRed`;
  return `${label} discarded`;
}
