export function problemStatusDisplay(problem, response = {}) {
  if (!problem) {
    return {
      status: 'idle',
      text: response.before || 'All done.'
    };
  }

  if (problem.status !== 'submitted') {
    return {
      status: 'solving',
      text: 'Try your best and press Submit when you are ready.'
    };
  }

  if (problem.recognition?.status === 'empty') {
    return {
      status: 'incomplete',
      text: 'Incomplete'
    };
  }

  if (problem.recognition?.status === 'error') {
    return {
      status: 'incomplete',
      text: 'Incomplete'
    };
  }

  const grading = problem.recognition?.result?.grading || null;
  const status = grading?.result?.problemStatus || '';
  if (status === 'correct') {
    return {
      status,
      text: decisionLabel(status)
    };
  }

  if (!recognitionIsFinal(problem.recognition)) {
    return {
      status: 'analyzing',
      text: 'Analyzing'
    };
  }

  if (status === 'correct' || status === 'incorrect' || status === 'incomplete') {
    return {
      status,
      text: decisionLabel(status)
    };
  }

  if (status === 'not_started') {
    return {
      status: 'incomplete',
      text: 'Incomplete'
    };
  }

  return {
    status: 'analyzing',
    text: 'Analyzing'
  };
}

function recognitionIsFinal(recognition = null) {
  if (recognition?.status !== 'complete') return false;
  const realtime = recognition.result?.realtime || recognition.realtime || null;
  if (realtime?.allFinal === false) return false;
  if (realtime?.components?.some((component) => (
    component.status !== 'final' || component.contested
  ))) {
    return false;
  }
  return true;
}

function decisionLabel(problemStatus) {
  return String(problemStatus || '')
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}
