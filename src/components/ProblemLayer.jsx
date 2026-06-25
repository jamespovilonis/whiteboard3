import katex from 'katex';
import { useMemo } from 'react';
import { boardToScreen } from '../whiteboard/viewport.js';

export default function ProblemLayer({ problem, viewport }) {
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
