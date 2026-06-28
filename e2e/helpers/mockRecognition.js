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
      const candidateScores = (payload.candidateGroups || []).map(scoreCandidateGroup);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          candidateScores,
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

function scoreCandidateGroup(group) {
  const bestLatex = firstLatex(group);
  const candidateScores = (group.candidates?.length ? group.candidates : [{ latex: bestLatex }])
    .map((candidate) => ({
      latex: candidate.latex || bestLatex,
      sound: true,
      semanticScore: candidate.latex === bestLatex ? 3 : 1,
      equivalentToProblem: false,
      equivalentToPrevious: false
    }));

  return {
    candidateId: group.candidateId,
    lineIndex: group.lineIndex ?? null,
    bestLatex,
    sound: true,
    semanticScore: 3,
    equivalentToProblem: false,
    equivalentToPrevious: false,
    candidateScores
  };
}

function firstLatex(group) {
  const candidateLatex = (group.candidates || [])
    .map((candidate) => String(candidate?.latex || '').trim())
    .find(Boolean);
  return candidateLatex || String(group.latex || '').trim() || 'x = 4';
}
