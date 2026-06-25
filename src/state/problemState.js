import { DEFAULT_PROBLEM, INITIAL_VIEWPORT } from '../whiteboard/constants.js';
import { getInitialProblemPosition } from '../whiteboard/viewport.js';

export function createInitialProblemState(viewportWidth) {
  return {
    ...DEFAULT_PROBLEM,
    boardPosition: getInitialProblemPosition(INITIAL_VIEWPORT, viewportWidth)
  };
}
