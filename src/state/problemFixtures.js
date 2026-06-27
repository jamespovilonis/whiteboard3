export const TEST_PROBLEMS = Object.freeze([
  // Temporary visual-debug fixtures until the full grading system is connected.
  {
    id: 'problem-1',
    latex: '2x + 3 = 11',
    modelResponse: {
      before: 'Here is one way to think about the next step.',
      latex: '2x = 8',
      after: 'When you are ready, you can ask for a hint or submit your answer.'
    }
  },
  {
    id: 'problem-2',
    latex: '\\frac{x + 1}{2} = \\frac{5}{3}',
    modelResponse: {
      before: 'Clear the denominators with a common multiplier.',
      latex: '3(x + 1) = 10',
      after: 'Then distribute and isolate the variable term.'
    }
  },
  {
    id: 'problem-3',
    latex: '\\log_{2}(x) + 3 = 7',
    modelResponse: {
      before: 'Move the constant before rewriting the logarithm.',
      latex: '\\log_{2}(x) = 4',
      after: 'Use the matching exponential form when you are ready.'
    }
  }
]);
