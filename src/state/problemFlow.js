import {
  ANSWER_BOX_PADDING,
  INITIAL_VIEWPORT,
  NEXT_PROBLEM_GAP,
  PROBLEM_BOX
} from '../whiteboard/constants.js';
import { bboxOverlap, padBbox, unionBbox } from '../whiteboard/geometry.js';
import { getInitialProblemPosition } from '../whiteboard/viewport.js';
import { TEST_PROBLEMS } from './problemFixtures.js';

export function createInitialProblemFlow(viewportWidth) {
  const firstProblem = createProblemSession({
    definition: TEST_PROBLEMS[0],
    index: 0,
    boardPosition: getInitialProblemPosition(INITIAL_VIEWPORT, viewportWidth),
    viewportWidth,
    viewport: INITIAL_VIEWPORT
  });

  return {
    activeProblemId: firstProblem.id,
    problems: [firstProblem],
    completedCount: 0
  };
}

export function getActiveProblem(flow) {
  return flow.problems.find((problem) => problem.id === flow.activeProblemId) || null;
}

export function getActiveModelResponse(flow) {
  return getActiveProblem(flow)?.modelResponse || {
    before: 'All test problems have been submitted.',
    latex: '\\checkmark',
    after: 'The next step will be to connect this shell to the response API.'
  };
}

export function getCompletedRecognitionResults(flow) {
  return flow.problems
    .filter((problem) => problem.status === 'submitted')
    .map((problem) => ({
      problemId: problem.id,
      problemLatex: problem.latex,
      recognition: problem.recognition
    }));
}

export function reconcileProblemFlowWithStrokes(flow, strokes) {
  const activeProblem = getActiveProblem(flow);
  if (!activeProblem || activeProblem.status !== 'solving') {
    return flow;
  }

  return updateProblem(flow, activeProblem.id, (problem) => {
    if (problem.answerBoxFrozen) return problem;

    const answer = buildActiveAnswerBox(problem, strokes);
    if (
      sameIds(problem.answerStrokeIds, answer.strokeIds) &&
      sameBbox(problem.answerContentBox, answer.contentBox) &&
      sameBbox(problem.answerBox, answer.answerBox)
    ) {
      return problem;
    }

    return {
      ...problem,
      answerStrokeIds: answer.strokeIds,
      answerContentBox: answer.contentBox,
      answerBox: answer.answerBox
    };
  });
}

export function submitActiveProblem(flow, viewportWidth) {
  const activeProblem = getActiveProblem(flow);
  if (!activeProblem) return { flow, targetViewport: null };

  const frozenBottom = getFrozenBottom(activeProblem);
  const completedFlow = updateProblem(flow, activeProblem.id, (problem) => ({
    ...problem,
    status: 'submitted',
    answerBoxFrozen: true,
    recognition: {
      ...problem.recognition,
      status: problem.answerStrokeIds.length > 0 ? 'pending' : 'empty',
      error: null
    }
  }));

  const nextIndex = activeProblem.index + 1;
  if (nextIndex >= TEST_PROBLEMS.length) {
    return {
      flow: {
        ...completedFlow,
        activeProblemId: null,
        completedCount: TEST_PROBLEMS.length
      },
      targetViewport: null
    };
  }

  const nextDefinition = TEST_PROBLEMS[nextIndex];
  const nextPosition = {
    x: activeProblem.boardPosition.x,
    y: frozenBottom + NEXT_PROBLEM_GAP
  };
  const targetViewport = viewportForProblemPosition(nextPosition, viewportWidth);
  const nextProblem = createProblemSession({
    definition: nextDefinition,
    index: nextIndex,
    boardPosition: nextPosition,
    viewportWidth,
    viewport: targetViewport
  });

  return {
    flow: {
      ...completedFlow,
      activeProblemId: nextProblem.id,
      completedCount: nextIndex,
      problems: [...completedFlow.problems, nextProblem]
    },
    targetViewport
  };
}

export function applyProblemRecognitionResult(flow, problemId, result) {
  return updateProblem(flow, problemId, (problem) => ({
    ...problem,
    recognition: {
      ...problem.recognition,
      status: 'complete',
      error: null,
      result,
      completedAt: Date.now()
    }
  }));
}

export function applyProblemRecognitionError(flow, problemId, error) {
  return updateProblem(flow, problemId, (problem) => ({
    ...problem,
    recognition: {
      ...problem.recognition,
      status: 'error',
      error: error instanceof Error ? error.message : String(error || 'Recognition failed'),
      failedAt: Date.now()
    }
  }));
}

export function viewportForProblemPosition(boardPosition, viewportWidth) {
  return {
    x: boardPosition.x - viewportWidth * 0.25,
    y: boardPosition.y - 56,
    scale: 1
  };
}

function createProblemSession({ definition, index, boardPosition, viewportWidth, viewport }) {
  return {
    id: definition.id,
    index,
    status: 'solving',
    latex: definition.latex,
    modelResponse: definition.modelResponse,
    boardPosition,
    problemBox: createProblemBox(boardPosition, viewportWidth, viewport),
    answerStrokeIds: [],
    answerContentBox: null,
    answerBox: null,
    answerBoxFrozen: false,
    recognition: {
      status: 'idle',
      error: null,
      result: null
    }
  };
}

function createProblemBox(boardPosition, viewportWidth, viewport) {
  const width = Math.max(PROBLEM_BOX.minWidth, viewportWidth - PROBLEM_BOX.leftPadding - PROBLEM_BOX.rightPadding);

  return {
    xMin: viewport.x + PROBLEM_BOX.leftPadding,
    yMin: boardPosition.y + PROBLEM_BOX.topOffset,
    xMax: viewport.x + PROBLEM_BOX.leftPadding + width,
    yMax: boardPosition.y + PROBLEM_BOX.topOffset + PROBLEM_BOX.height
  };
}

function buildActiveAnswerBox(problem, strokes) {
  const candidates = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  const selectedIds = new Set();
  let contentBox = null;
  let answerBox = null;
  let didAddStroke = true;

  while (didAddStroke) {
    didAddStroke = false;

    for (const stroke of candidates) {
      if (selectedIds.has(stroke.id)) continue;

      const overlapsProblemBox = bboxOverlap(stroke.canvasBbox, problem.problemBox);
      const overlapsAnswerBox = Boolean(answerBox && bboxOverlap(stroke.canvasBbox, answerBox));

      if (!overlapsProblemBox && !overlapsAnswerBox) continue;

      selectedIds.add(stroke.id);
      contentBox = unionBbox(contentBox, stroke.canvasBbox);
      answerBox = padBbox(contentBox, ANSWER_BOX_PADDING);
      didAddStroke = true;
    }
  }

  return {
    strokeIds: candidates
      .filter((stroke) => selectedIds.has(stroke.id))
      .map((stroke) => stroke.id),
    contentBox,
    answerBox
  };
}

function updateProblem(flow, problemId, updater) {
  let didChange = false;
  const problems = flow.problems.map((problem) => {
    if (problem.id !== problemId) return problem;

    const nextProblem = updater(problem);
    if (nextProblem !== problem) didChange = true;
    return nextProblem;
  });

  if (!didChange) return flow;

  return {
    ...flow,
    problems
  };
}

function getFrozenBottom(problem) {
  if (problem.answerBox) return problem.answerBox.yMax;
  return problem.problemBox.yMax;
}

function sameIds(a, b) {
  if (a.length !== b.length) return false;
  return a.every((id, index) => id === b[index]);
}

function sameBbox(a, b) {
  if (!a || !b) return a === b;
  return a.xMin === b.xMin &&
    a.yMin === b.yMin &&
    a.xMax === b.xMax &&
    a.yMax === b.yMax;
}
