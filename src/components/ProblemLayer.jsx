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
    </>
  );
}

function DebugBox({ bbox, kind, problemId, viewport }) {
  if (!bbox) return null;

  const topLeft = boardToScreen({ x: bbox.xMin, y: bbox.yMin }, viewport);
  const width = Math.max(1, (bbox.xMax - bbox.xMin) * viewport.scale);
  const height = Math.max(1, (bbox.yMax - bbox.yMin) * viewport.scale);

  return (
    <div
      className={`debug-board-box debug-board-box-${kind}`}
      data-problem-id={problemId}
      style={{
        width,
        height,
        transform: `translate3d(${topLeft.x}px, ${topLeft.y}px, 0)`
      }}
    />
  );
}
