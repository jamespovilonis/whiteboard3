import {
  ANSWER_BOX_PADDING,
  INITIAL_VIEWPORT,
  NEXT_PROBLEM_GAP,
  PROBLEM_BOX
} from '../whiteboard/constants.js';
import { bboxOverlap, padBbox, unionBbox } from '../whiteboard/geometry.js';
import { getInitialProblemPosition } from '../whiteboard/viewport.js';

export function createInitialProblemFlow(viewportWidth, problemDefinitions = []) {
  const definitions = normalizeProblemDefinitions(problemDefinitions);
  if (!definitions.length) {
    return {
      activeProblemId: null,
      problemDefinitions: [],
      problems: [],
      completedCount: 0,
      awaitingEquation: true,
      customProblems: true
    };
  }

  const firstProblem = createProblemSession({
    definition: definitions[0],
    index: 0,
    boardPosition: getInitialProblemPosition(INITIAL_VIEWPORT, viewportWidth),
    viewportWidth,
    viewport: INITIAL_VIEWPORT
  });

  return {
    activeProblemId: firstProblem.id,
    problemDefinitions: definitions,
    problems: [firstProblem],
    completedCount: 0,
    awaitingEquation: false,
    customProblems: false
  };
}

export function getActiveProblem(flow) {
  return flow.problems.find((problem) => problem.id === flow.activeProblemId) || null;
}

export function getActiveModelResponse(flow) {
  if (flow.awaitingEquation) {
    return {
      before: 'Enter an equation to solve.',
      latex: '\\square',
      after: 'The problem will appear on the whiteboard.'
    };
  }

  const activeProblem = getActiveProblem(flow);
  if (activeProblem?.status === 'submitted') {
    return {
      before: submittedProblemStatusLabel(activeProblem),
      latex: '',
      after: '',
      statusOnly: true
    };
  }

  return activeProblem?.modelResponse || {
    before: 'All done.',
    latex: '\\checkmark',
    after: 'You have submitted every equation.'
  };
}

export function getCompletedRecognitionResults(flow) {
  return flow.problems
    .filter((problem) => (
      problem.status === 'submitted' ||
      problem.recognition?.status !== 'idle' ||
      problem.recognition?.result
    ))
    .map((problem) => ({
      problemId: problem.id,
      problemLatex: problem.latex,
      problemMetadata: problem.metadata || {},
      problemStatus: problem.status,
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
  if (activeProblem.status !== 'solving') return { flow, targetViewport: null };
  const recognitionStatus = submissionRecognitionStatus(activeProblem);

  const completedFlow = updateProblem(flow, activeProblem.id, (problem) => ({
    ...problem,
    status: 'submitted',
    answerBoxFrozen: true,
    recognition: {
      ...problem.recognition,
      status: recognitionStatus,
      error: null
    }
  }));
  const completedCount = completedFlow.problems.filter((problem) => (
    problem.status === 'submitted'
  )).length;

  return {
    flow: {
      ...completedFlow,
      completedCount,
      awaitingEquation: false
    },
    targetViewport: null
  };
}

export function requestNextProblem(flow, viewportWidth) {
  let workingFlow = flow;
  let activeProblem = getActiveProblem(workingFlow);
  if (!activeProblem || activeProblem.status !== 'submitted') {
    return { flow: workingFlow, targetViewport: null };
  }

  const definitions = normalizeProblemDefinitions(workingFlow.problemDefinitions || []);
  const frozenBottom = getFrozenBottom(activeProblem);

  const nextIndex = activeProblem.index + 1;
  if (nextIndex >= definitions.length) {
    const completedCount = workingFlow.problems.filter((problem) => (
      problem.status === 'submitted'
    )).length;

    return {
      flow: {
        ...workingFlow,
        problemDefinitions: definitions,
        activeProblemId: null,
        completedCount,
        awaitingEquation: Boolean(workingFlow.customProblems),
        customProblems: Boolean(workingFlow.customProblems)
      },
      targetViewport: null
    };
  }

  const nextDefinition = definitions[nextIndex];
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
      ...workingFlow,
      problemDefinitions: definitions,
      activeProblemId: nextProblem.id,
      completedCount: nextIndex,
      problems: [...workingFlow.problems, nextProblem],
      awaitingEquation: false,
      customProblems: Boolean(workingFlow.customProblems)
    },
    targetViewport
  };
}

export function startCustomProblem(flow, latex, viewportWidth) {
  const normalizedLatex = String(latex || '').trim();
  if (!normalizedLatex) {
    return { flow, targetViewport: null, problem: null };
  }

  const lastProblem = flow.problems[flow.problems.length - 1] || null;
  const index = flow.problems.length;
  const boardPosition = lastProblem
    ? {
        x: lastProblem.boardPosition.x,
        y: getFrozenBottom(lastProblem) + NEXT_PROBLEM_GAP
      }
    : getInitialProblemPosition(INITIAL_VIEWPORT, viewportWidth);
  const targetViewport = lastProblem
    ? viewportForProblemPosition(boardPosition, viewportWidth)
    : INITIAL_VIEWPORT;
  const definition = {
    id: `problem-${index + 1}`,
    kind: 'equation-solving',
    latex: normalizedLatex,
    modelResponse: normalizeModelResponse(null, normalizedLatex, index),
    metadata: {
      source: 'user-latex'
    }
  };
  const problem = createProblemSession({
    definition,
    index,
    boardPosition,
    viewportWidth,
    viewport: targetViewport
  });

  return {
    flow: {
      ...flow,
      activeProblemId: problem.id,
      problems: [...flow.problems, problem],
      awaitingEquation: false,
      customProblems: true
    },
    targetViewport,
    problem
  };
}

export function normalizeProblemDefinitions(problemDefinitions = []) {
  const validDefinitions = (Array.isArray(problemDefinitions) ? problemDefinitions : [])
    .filter((definition) => (
      definition &&
      (!definition.kind || definition.kind === 'equation-solving') &&
      typeof definition.id === 'string' &&
      definition.id.trim() &&
      typeof definition.latex === 'string' &&
      definition.latex.trim()
    ))
    .map((definition, index) => ({
      id: definition.id,
      kind: 'equation-solving',
      latex: definition.latex,
      modelResponse: normalizeModelResponse(definition.modelResponse, definition.latex, index),
      metadata: normalizeProblemMetadata(definition)
    }));

  return validDefinitions;
}

export function applyProblemRecognitionResult(flow, problemId, result) {
  return updateProblem(flow, problemId, (problem) => ({
    ...problem,
    recognition: {
      ...problem.recognition,
      status: 'complete',
      error: null,
      result,
      realtime: result?.realtime || problem.recognition.realtime || null,
      completedAt: Date.now()
    }
  }));
}

export function applyProblemRecognitionProgress(flow, problemId, { status = 'pending', result = null, realtime = null } = {}) {
  return updateProblem(flow, problemId, (problem) => ({
    ...problem,
    recognition: {
      ...problem.recognition,
      status,
      error: null,
      result: result || problem.recognition.result,
      realtime: realtime || result?.realtime || problem.recognition.realtime || null,
      updatedAt: Date.now(),
      completedAt: status === 'complete' ? Date.now() : problem.recognition.completedAt || null
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

export function applyProblemGradingProgress(flow, problemId, grading) {
  return updateProblem(flow, problemId, (problem) => {
    const currentResult = problem.recognition?.result || {};
    return {
      ...problem,
      recognition: {
        ...problem.recognition,
        result: {
          ...currentResult,
          grading
        },
        updatedAt: Date.now()
      }
    };
  });
}

export function isProblemReadyForNext(problem) {
  if (!problem) return false;
  if (problem.status === 'submitted' && ['complete', 'empty', 'error'].includes(problem.recognition?.status)) return true;
  return false;
}

export function submittedProblemStatusLabel(problem) {
  const recognition = problem?.recognition || {};
  const grading = recognition.result?.grading || null;
  const problemStatus = grading?.result?.problemStatus || '';
  if (problemStatus === 'correct') return 'Correct';

  if (recognition.status === 'empty') return 'No answer submitted.';
  if (recognition.status === 'error') return 'Unable to analyze.';
  if (recognition.status !== 'complete') return 'Analyzing';

  if (!grading || grading.status === 'pending') return 'Analyzing';
  if (grading.failed || grading.status === 'failed') return 'Unable to grade.';

  if (!problemStatus) return 'Analyzing';
  return problemStatus
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
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
    metadata: definition.metadata || {},
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
      result: null,
      realtime: null
    }
  };
}

function normalizeProblemMetadata(definition) {
  const {
    id,
    kind,
    latex,
    modelResponse,
    metadata,
    ...rest
  } = definition;

  return {
    ...(metadata && typeof metadata === 'object' ? metadata : {}),
    ...rest
  };
}

function submissionRecognitionStatus(problem) {
  if (!problem.answerStrokeIds.length) return 'empty';
  const status = problem.recognition?.status || 'idle';
  if (status === 'complete' || status === 'error') return status;
  return 'pending';
}

function normalizeModelResponse(modelResponse, latex, index) {
  if (
    modelResponse &&
    typeof modelResponse.before === 'string' &&
    typeof modelResponse.latex === 'string' &&
    typeof modelResponse.after === 'string'
  ) {
    return modelResponse;
  }

  return {
    before: index === 0 ? 'Solve the equation.' : 'Continue solving the next equation.',
    latex,
    after: 'Submit your work when you are ready.'
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
      const continuesAnswerColumn = Boolean(answerBox && strokeContinuesAnswerColumn(stroke, answerBox));
      const startsAnswerNearProblem = !answerBox && strokeStartsAnswerNearProblem(stroke, problem);

      if (!overlapsProblemBox && !overlapsAnswerBox && !continuesAnswerColumn && !startsAnswerNearProblem) continue;

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

function strokeStartsAnswerNearProblem(stroke, problem) {
  const box = stroke?.canvasBbox;
  const problemBox = problem?.problemBox;
  if (!box || !problemBox) return false;

  const centerX = (box.xMin + box.xMax) / 2;
  const centerY = (box.yMin + box.yMax) / 2;
  const horizontalRange = {
    xMin: problemBox.xMin - ANSWER_BOX_PADDING,
    xMax: problemBox.xMax + ANSWER_BOX_PADDING
  };
  const verticalRange = {
    yMin: problemBox.yMin,
    yMax: problemBox.yMax + ANSWER_BOX_PADDING * 3
  };

  return centerX >= horizontalRange.xMin &&
    centerX <= horizontalRange.xMax &&
    centerY >= verticalRange.yMin &&
    centerY <= verticalRange.yMax;
}

function strokeContinuesAnswerColumn(stroke, answerBox) {
  const box = stroke?.canvasBbox;
  if (!box || !answerBox) return false;

  const centerX = (box.xMin + box.xMax) / 2;
  const centerY = (box.yMin + box.yMax) / 2;
  const horizontalRange = {
    xMin: answerBox.xMin - ANSWER_BOX_PADDING,
    xMax: answerBox.xMax + ANSWER_BOX_PADDING
  };
  const verticalRange = {
    yMin: answerBox.yMin,
    yMax: answerBox.yMax + ANSWER_BOX_PADDING * 2.5
  };

  return centerX >= horizontalRange.xMin &&
    centerX <= horizontalRange.xMax &&
    centerY >= verticalRange.yMin &&
    centerY <= verticalRange.yMax;
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
