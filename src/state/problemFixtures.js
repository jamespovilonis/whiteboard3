export const TEST_PROBLEMS = Object.freeze([
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
    latex: '3y - 4 = 14',
    modelResponse: {
      before: 'For this one, isolate the variable term first.',
      latex: '3y = 18',
      after: 'Keep the operation balanced on both sides.'
    }
  },
  {
    id: 'problem-3',
    latex: '\\frac{x}{5} + 2 = 9',
    modelResponse: {
      before: 'Start by removing the constant from both sides.',
      latex: '\\frac{x}{5} = 7',
      after: 'Then use the inverse operation to finish.'
    }
  }
]);
