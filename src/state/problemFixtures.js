export const TEST_PROBLEMS = Object.freeze([
  {
    id: 'problem-1',
    kind: 'equation-solving',
    latex: '2x + 3 = 11',
    modelResponse: {
      before: 'Solve the equation.',
      latex: '2x + 3 = 11',
      after: 'Show each step, then submit your work.'
    }
  },
  {
    id: 'problem-2',
    kind: 'equation-solving',
    latex: '\\frac{x}{2} + 5 = 13',
    modelResponse: {
      before: 'Nice. Now solve this equation.',
      latex: '\\frac{x}{2} + 5 = 13',
      after: 'Keep the algebra steps visible inside the answer box.'
    }
  },
  {
    id: 'problem-3',
    kind: 'equation-solving',
    latex: '3(x - 4) = 18',
    modelResponse: {
      before: 'One more equation.',
      latex: '3(x - 4) = 18',
      after: 'Submit when your final answer is clear.'
    }
  }
]);
