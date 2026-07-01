export async function installMockRecognitionRoutes(page, options = {}) {
  const calls = [];
  const latexQueue = [...(options.latexLines || [])];

  await page.route('**/__mock_ocr/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname.replace(/^.*\/__mock_ocr/, '') || '/';
    calls.push({
      endpoint,
      method: request.method(),
      url: request.url(),
      postDataJson: parsePostData(request.postData())
    });

    if (endpoint === '/segment-lines') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          detections: options.detections || [],
          imageWidth: 1000,
          imageHeight: 260,
          model: 'mock-dbnet',
          elapsedSeconds: 0.01
        })
      });
      return;
    }

    if (endpoint === '/recognize') {
      const latex = latexQueue.shift() || options.defaultLatex || 'x = 4';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model: 'mock-comer',
          top: {
            latex,
            confidence: 0.99
          },
          candidates: [
            { latex, confidence: 0.99 },
            { latex: options.alternateLatex || 'x = 5', confidence: 0.1 }
          ],
          elapsedSeconds: 0.02
        })
      });
      return;
    }

    if (endpoint === '/score-latex-candidates') {
      const payload = JSON.parse(request.postData() || '{}');
      const problemType = payload.problemType || payload.problemMetadata?.problemType || 'equation-solving';
      const answerManifest = options.answerManifest || {
        problem_raw: payload.problemLatex || '',
        responseKind: problemType === 'evaluate-expression' ? 'numeric_value' : 'solution_set',
        variable: payload.problemMetadata?.solveVariable || 'x',
        cardinality: 'finite',
        exact_set: ['4'],
        decimal_set: [4],
        tolerance: 0.005,
        acceptable_strings: ['x=4', '4']
      };
      const candidateScores = (payload.candidateGroups || []).map((group) => (
        scoreCandidateGroup(group, { answerManifest })
      ));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answerManifest,
          candidateScores,
          elapsedSeconds: 0.01,
          failed: false
        })
      });
      return;
    }

    if (endpoint === '/grade-equation-work' || endpoint === '/grade-math-work') {
      const payload = JSON.parse(request.postData() || '{}');
      const lines = payload.lines || [];
      const problemType = payload.problemType || payload.problemMetadata?.problemType || 'equation-solving';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(options.gradingResponse || {
          problem: {
            latex: payload.problemLatex || '',
            solveVariable: payload.problemMetadata?.solveVariable || 'x',
            cardinality: 'finite',
            solutionSet: [],
            decimalSet: [],
            tolerance: 0.005,
            manifest: {
              responseKind: problemType === 'evaluate-expression' ? 'numeric_value' : 'solution_set',
              problem_raw: payload.problemLatex || '',
              cardinality: 'finite'
            }
          },
          steps: lines.map((line, index) => ({
            lineIndex: line.lineIndex ?? index,
            studentLatex: line.acceptedLatex || line.latex || '',
            classification: 'valid_step',
            selectedCandidateIndex: 0,
            solutionCoverage: 'none',
            matchedSolutions: []
          })),
          result: {
            problemStatus: options.problemStatus || 'correct',
            breakdownLineIndex: null,
            foundSolutions: [],
            missingSolutions: []
          },
          elapsedSeconds: 0.01,
          failed: false
        })
      });
      return;
    }

    await route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: `Unhandled mock endpoint: ${endpoint}` })
    });
  });

  return {
    calls,
    endpoints() {
      return calls.map((call) => call.endpoint);
    }
  };
}

function parsePostData(postData) {
  if (!postData) return null;
  try {
    return JSON.parse(postData);
  } catch (_error) {
    return null;
  }
}

function scoreCandidateGroup(group, { answerManifest }) {
  const bestLatex = firstLatex(group);
  const candidateScores = (group.candidates?.length ? group.candidates : [{ latex: bestLatex }])
    .map((candidate) => ({
      latex: candidate.latex || bestLatex,
      sound: true,
      semanticScore: candidate.latex === bestLatex ? 3 : 1,
      equivalentToProblem: false,
      equivalentToPrevious: false
    }));
  const selectedCandidateIndex = 0;
  const selectedLatex = candidateScores[selectedCandidateIndex]?.latex || bestLatex;
  const matchedSolutions = String(selectedLatex).includes('4') ? ['4'] : [];

  return {
    candidateId: group.candidateId,
    lineIndex: group.lineIndex ?? null,
    bestLatex,
    sound: true,
    semanticScore: 3,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    grading: {
      studentLatex: selectedLatex,
      classification: 'valid_step',
      selectedCandidateIndex,
      solutionCoverage: matchedSolutions.length ? 'full' : 'none',
      matchedSolutions,
      candidateVerdicts: candidateScores.map((candidate, index) => ({
        candidateIndex: index,
        latex: candidate.latex,
        studentLatex: candidate.latex,
        classification: index === selectedCandidateIndex ? 'valid_step' : 'invalid_step',
        selectedCandidateIndex: 0,
        solutionCoverage: index === selectedCandidateIndex && matchedSolutions.length ? 'full' : 'none',
        matchedSolutions: index === selectedCandidateIndex ? matchedSolutions : []
      }))
    },
    answerManifest,
    candidateScores
  };
}

function firstLatex(group) {
  const candidateLatex = (group.candidates || [])
    .map((candidate) => String(candidate?.latex || '').trim())
    .find(Boolean);
  return candidateLatex || String(group.latex || '').trim() || 'x = 4';
}
