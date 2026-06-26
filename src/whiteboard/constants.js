export const BOARD_SIZE = Object.freeze({
  width: 6000,
  height: 4000
});

export const INITIAL_VIEWPORT = Object.freeze({
  x: -80,
  y: -80,
  scale: 1
});

export const RESET_DRIFT_THRESHOLD = 220;

export const PROBLEM_BOX = Object.freeze({
  topOffset: 92,
  leftPadding: 80,
  rightPadding: 80,
  minWidth: 760,
  height: 340
});

export const ANSWER_BOX_PADDING = 90;
export const NEXT_PROBLEM_GAP = 150;

export const DEFAULT_TOOL = 'pen';
export const DEFAULT_PEN_COLOR = '#000000';
export const DEFAULT_PEN_WIDTH = 9;

export const COLOR_SWATCHES = [
  '#000000',
  '#333333',
  '#888888',
  '#ffffff',
  '#e53935',
  '#fb8c00',
  '#fdd835',
  '#43a047',
  '#00acc1',
  '#1e88e5',
  '#8e24aa',
  '#d81b60'
];

export function sliderToWidth(value) {
  return Math.round(2 + (Number(value) - 1) * (22 / 9));
}
