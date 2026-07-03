import katex from 'katex';
import { useEffect, useMemo, useRef, useState } from 'react';
import { problemStatusDisplay } from './problemStatusDisplay.js';

const SHELL_TRANSITION_MS = 450;

export default function ModelShell({
  response,
  activeProblem = null,
  recognitionResults = [],
  auditByProblemId = {},
  debugMode = false,
  submitDisabled = false,
  nextProblemDisabled = false,
  onNextProblem,
  onSubmitAnswer,
  onAuditNoteSubmit
}) {
  const [mode, setMode] = useState('open');
  const closeTimerRef = useRef(null);
  const equationHtml = useMemo(() => {
    return katex.renderToString(response.latex, {
      throwOnError: false,
      displayMode: true
    });
  }, [response.latex]);
  const problemStatus = problemStatusDisplay(activeProblem, response);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const openShell = () => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    setMode('open');
  };

  const closeShell = () => {
    setMode('closing');
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      setMode('closed');
    }, SHELL_TRANSITION_MS);
  };

  const panelVisible = debugMode || mode !== 'closed';
  const shouldShowToggle = !debugMode && (mode === 'closed' || mode === 'closing');
  const statusText = activeProblem?.status === 'submitted' && response.feedbackText
    ? response.feedbackText
    : problemStatus.text;

  return (
    <>
      {shouldShowToggle && (
        <button
          className={`model-shell-toggle ${mode === 'closing' ? 'is-returning' : ''}`}
          type="button"
          aria-label="Open model response"
          aria-hidden={mode === 'closing'}
          tabIndex={mode === 'closing' ? -1 : 0}
          onClick={openShell}
        >
          M
        </button>
      )}

      {panelVisible && (
        <aside
          className={`model-shell-panel ${debugMode ? 'is-debug' : ''} ${mode === 'closing' && !debugMode ? 'is-closing' : ''}`}
          aria-label={debugMode ? 'Recognition debugger' : 'Model response'}
          aria-hidden={mode === 'closing' && !debugMode}
        >
          {!debugMode && (
            <button
              className="model-shell-minimize"
              type="button"
              aria-label="Minimize model response"
              onClick={closeShell}
            >
              -
            </button>
          )}

          {debugMode ? (
            <RecognitionDebugInspector
              results={recognitionResults}
              auditByProblemId={auditByProblemId}
              onAuditNoteSubmit={onAuditNoteSubmit}
            />
          ) : (
            <>
              <div className="model-shell-copy">
                <p
                  className="model-shell-status"
                  data-status={problemStatus.status}
                >
                  {statusText}
                </p>
                {activeProblem?.status !== 'submitted' && (
                  <div
                    className="model-shell-equation"
                    dangerouslySetInnerHTML={{ __html: equationHtml }}
                  />
                )}
              </div>
            </>
          )}

          <div className={`model-shell-actions ${debugMode ? 'is-debug' : ''}`}>
            <button type="button">Ask for help</button>
            <button
              type="button"
              data-testid="submit-answer"
              disabled={submitDisabled}
              onClick={onSubmitAnswer}
            >
              Submit
            </button>
            <button
              type="button"
              data-testid="next-problem"
              disabled={nextProblemDisabled}
              onClick={onNextProblem}
            >
              Next Problem
            </button>
          </div>
        </aside>
      )}
    </>
  );
}

function RecognitionDebugInspector({ results = [], auditByProblemId = {}, onAuditNoteSubmit }) {
  const submitted = results.slice().reverse();

  return (
    <div className="recognition-debug">
      <div className="recognition-debug-heading">
        <span>Recognition Debug</span>
        <span>{submitted.length ? `${submitted.length} tracked` : 'No reads'}</span>
      </div>

      {submitted.length === 0 && (
        <p className="recognition-message">Write an answer to inspect line candidates.</p>
      )}

      {submitted.map((entry) => (
        <DebugRecognitionResult
          key={entry.problemId}
          entry={entry}
          audit={auditByProblemId[entry.problemId] || null}
          onAuditNoteSubmit={onAuditNoteSubmit}
        />
      ))}
    </div>
  );
}

function DebugRecognitionResult({ entry, audit = null, onAuditNoteSubmit }) {
  const recognition = entry.recognition || {};
  const status = recognition.status || 'idle';
  const result = recognition.result || null;
  const candidates = result?.candidatePredictions || [];
  const lines = result?.lines || [];

  return (
    <section className="recognition-debug-result" data-status={status}>
      <div className="recognition-result-top">
        <span>Problem {entry.problemId.replace('problem-', '')}</span>
        <span>{debugStatusLabel(status, result)}</span>
      </div>

      {status === 'pending' && (
        <p className="recognition-message">Recognition is running.</p>
      )}

      {status === 'error' && (
        <p className="recognition-message">{recognition.error || 'Recognition failed.'}</p>
      )}

      {status === 'empty' && (
        <p className="recognition-message">No answer strokes.</p>
      )}

      {result && (
        <>
          <DebugMetrics result={result} />
          <AuditDebugPanel
            audit={audit}
            problemId={entry.problemId}
            onAuditNoteSubmit={onAuditNoteSubmit}
          />
          <GradingDebugPanel grading={result.grading} />
          {candidates.length > 0 && (
            <div className="recognition-debug-candidates">
              {candidates.map((candidate, index) => (
                <DebugCandidateCard
                  key={candidate.candidateId}
                  candidate={candidate}
                  index={index}
                />
              ))}
            </div>
          )}
          <FinalLinePost lines={lines} />
        </>
      )}
    </section>
  );
}

function AuditDebugPanel({ audit = null, problemId = '', onAuditNoteSubmit }) {
  const status = audit?.status || 'waiting';
  const triggerReasons = audit?.triggerReasons || [];
  const discrepancyCount = Number(audit?.discrepancyCount);
  const label = audit?.label || (status === 'waiting' ? 'Audit waiting' : auditStatusDebugLabel(status));
  const [note, setNote] = useState('');
  const [submitState, setSubmitState] = useState('idle');
  const [submitError, setSubmitError] = useState('');

  useEffect(() => {
    setNote('');
    setSubmitState('idle');
    setSubmitError('');
  }, [audit?.auditId, problemId]);

  const handleSubmitNote = async (event) => {
    event.preventDefault();
    const trimmed = note.trim();
    if (!trimmed || submitState === 'saving') return;
    setSubmitState('saving');
    setSubmitError('');
    try {
      await onAuditNoteSubmit?.({
        auditId: audit?.auditId || null,
        problemId,
        note: trimmed
      });
      setNote('');
      setSubmitState('saved');
    } catch (error) {
      setSubmitError(error?.message || 'Could not save note.');
      setSubmitState('error');
    }
  };

  return (
    <div className="audit-debug" data-status={status}>
      <div className="audit-debug-top">
        <span>VLM Audit</span>
        <span>{label}</span>
      </div>
      <dl className="recognition-metrics audit-debug-metrics">
        <div>
          <dt>Taking place</dt>
          <dd>{auditTakingPlaceLabel(status)}</dd>
        </div>
        <div>
          <dt>Audit ID</dt>
          <dd>{audit?.auditId || 'n/a'}</dd>
        </div>
        <div>
          <dt>Log</dt>
          <dd>{status === 'logged' ? 'Entered' : 'Pending'}</dd>
        </div>
      </dl>
      {status === 'logged' && Number.isFinite(discrepancyCount) && (
        <p className="recognition-message">
          {discrepancyCount === 0
            ? 'Audit logged with no discrepancies.'
            : `Audit logged with ${discrepancyCount} discrepancy${discrepancyCount === 1 ? '' : 'ies'}.`}
        </p>
      )}
      {status === 'error' && (
        <p className="recognition-message">{audit?.error || 'Audit failed.'}</p>
      )}
      {triggerReasons.length > 0 && (
        <p className="audit-debug-reasons">{triggerReasons.join(', ')}</p>
      )}
      <form className="audit-debug-note" onSubmit={handleSubmitNote}>
        <textarea
          value={note}
          maxLength={2000}
          rows={2}
          placeholder="Add a personal audit note..."
          aria-label="Personal audit note"
          onChange={(event) => {
            setNote(event.target.value);
            if (submitState !== 'saving') {
              setSubmitState('idle');
              setSubmitError('');
            }
          }}
        />
        <div className="audit-debug-note-actions">
          <span role={submitState === 'error' ? 'alert' : 'status'}>
            {submitState === 'saving' && 'Saving note'}
            {submitState === 'saved' && 'Note saved'}
            {submitState === 'error' && submitError}
            {submitState === 'idle' && audit?.personalNote && `Last note: ${audit.personalNote}`}
          </span>
          <button
            type="submit"
            disabled={!note.trim() || submitState === 'saving' || !onAuditNoteSubmit}
          >
            Submit
          </button>
        </div>
      </form>
    </div>
  );
}

function GradingDebugPanel({ grading = null }) {
  if (!grading) return null;

  const status = grading.status || (grading.failed ? 'failed' : 'complete');
  const problemStatus = grading.result?.problemStatus || (status === 'pending' ? 'pending' : 'n/a');
  const steps = grading.steps || [];
  const problem = grading.problem || {};
  const solutionSet = problem.solutionSet || [];

  return (
    <div className="grading-debug" data-status={problemStatus}>
      <div className="grading-debug-top">
        <span>Grading</span>
        <span>{gradingDecisionLabel(problemStatus, status)}</span>
      </div>

      {grading.failed && (
        <p className="recognition-message">{grading.error || 'Grading failed.'}</p>
      )}

      {!grading.failed && (
        <>
          <dl className="recognition-metrics grading-debug-metrics">
            <div>
              <dt>Status</dt>
              <dd>{gradingDecisionLabel(problemStatus, status)}</dd>
            </div>
            <div>
              <dt>Variable</dt>
              <dd>{problem.solveVariable || 'n/a'}</dd>
            </div>
            <div>
              <dt>Solutions</dt>
              <dd>{solutionSet.length ? solutionSet.join(', ') : problem.cardinality || 'n/a'}</dd>
            </div>
          </dl>

          {steps.length > 0 ? (
            <ol className="grading-debug-lines">
              {steps.map((step, index) => (
                <li key={`${step.lineIndex ?? index}-${step.studentLatex || ''}`}>
                  <div>
                    <span>Line {(step.lineIndex ?? index) + 1}</span>
                    <span>{lineClassificationLabel(step.classification)}</span>
                  </div>
                  <p>{step.studentLatex || 'No LaTeX'}</p>
                  <small>{gradingLineDetail(step)}</small>
                </li>
              ))}
            </ol>
          ) : (
            status === 'pending'
              ? <p className="recognition-message">Grading is running.</p>
              : <p className="recognition-message">No graded lines returned.</p>
          )}
        </>
      )}
    </div>
  );
}

function DebugMetrics({ result }) {
  const detection = result.detection || {};
  const segmentation = result.segmentation || {};
  const selected = segmentation.selected || [];
  const candidates = segmentation.candidates || [];

  return (
    <dl className="recognition-metrics recognition-debug-metrics">
      <div>
        <dt>Total</dt>
        <dd>{formatSeconds(result.timing?.totalElapsedSeconds)}</dd>
      </div>
      <div>
        <dt>Detector</dt>
        <dd>{detectionLabel(detection)}</dd>
      </div>
      <div>
        <dt>Cover</dt>
        <dd>{selected.length}/{candidates.length}</dd>
      </div>
    </dl>
  );
}

function DebugCandidateCard({ candidate, index }) {
  const status = candidateDebugStatus(candidate);
  const label = candidate.debugLabel || `C${index + 1}`;
  const semantic = candidate.sequentialSemantic || candidate.contextualSemantic || candidate.semantic || null;

  return (
    <article className="recognition-debug-candidate" data-status={status}>
      <div className="recognition-debug-candidate-top">
        <span>{label}</span>
        <span>{candidateStatusLabel(candidate, status)}</span>
      </div>

      {candidate.image?.dataUrl && (
        <img
          className="recognition-debug-crop"
          src={candidate.image.dataUrl}
          alt={`${label} rasterized crop`}
        />
      )}

      <dl className="recognition-debug-detail">
        <div>
          <dt>Total</dt>
          <dd>{formatSeconds(candidate.timing?.submitToFinalPredictionSeconds)}</dd>
        </div>
        <div>
          <dt>OCR</dt>
          <dd>{formatSeconds(candidate.timing?.ocrElapsedSeconds)}</dd>
        </div>
        <div>
          <dt>Score</dt>
          <dd>{formatNumber(candidate.evidenceScore)}</dd>
        </div>
        <div>
          <dt>Profiles</dt>
          <dd>{(candidate.profiles || []).join(', ') || 'candidate'}</dd>
        </div>
        <div>
          <dt>BBox</dt>
          <dd>{bboxLabel(candidate.tightBbox)}</dd>
        </div>
        <div>
          <dt>Semantic</dt>
          <dd>{semanticScoreLabel(semantic, candidate.timing)}</dd>
        </div>
      </dl>

      {candidate.prediction?.error && (
        <p className="recognition-message">{candidate.prediction.error}</p>
      )}

      <CandidateList
        candidates={candidate.candidates}
        semantic={semantic}
        limit={5}
      />
    </article>
  );
}

function FinalLinePost({ lines = [] }) {
  if (!lines.length) {
    return <p className="recognition-message">No selected lines returned.</p>;
  }

  return (
    <div className="recognition-debug-final">
      <span>Final Lines</span>
      <ol className="recognition-lines">
        {lines.map((line, index) => (
          <li key={`${line.candidateId}-${index}`}>
            <span>Line {index + 1}: </span>
            <RecognizedLatexLineText latex={line.acceptedLatex || line.latex} />
          </li>
        ))}
      </ol>
    </div>
  );
}

function RecognitionResults({ results }) {
  const recent = results.slice(-3).reverse();

  return (
    <div className="model-shell-recognition" aria-live="polite">
      {recent.map((entry) => (
        <RecognitionResult
          key={entry.problemId}
          entry={entry}
        />
      ))}
    </div>
  );
}

function RecognitionResult({ entry }) {
  const { recognition } = entry;
  const status = recognition?.status || 'idle';
  const result = recognition?.result || null;
  const lines = result?.latexLines || [];

  return (
    <div className="recognition-result" data-status={status}>
      <div className="recognition-result-top">
        <span>Problem {entry.problemId.replace('problem-', '')}</span>
        <span>{statusLabel(status)}</span>
      </div>

      {status === 'complete' && lines.length > 0 && (
        <ol className="recognition-lines">
          {lines.map((line, index) => (
            <RecognizedLatexLine
              key={`${entry.problemId}-${index}`}
              latex={line}
            />
          ))}
        </ol>
      )}

      {status === 'complete' && result?.grading && (
        <GradingVerdict grading={result.grading} />
      )}

      {status === 'complete' && result && (
        <RecognitionEvidence result={result} />
      )}

      {status === 'complete' && lines.length === 0 && (
        <p className="recognition-message">No LaTeX returned.</p>
      )}

      {status === 'error' && (
        <p className="recognition-message">{recognition.error || 'Recognition failed.'}</p>
      )}

      {status === 'empty' && (
        <p className="recognition-message">No answer strokes.</p>
      )}
    </div>
  );
}

function GradingVerdict({ grading }) {
  const gradingStatus = grading.status || (grading.failed ? 'failed' : 'complete');
  const problemStatus = grading.result?.problemStatus || (gradingStatus === 'pending' ? 'pending' : '');
  const foundSolutions = grading.result?.foundSolutions || [];
  const missingSolutions = grading.result?.missingSolutions || [];

  if (!problemStatus && gradingStatus !== 'failed') return null;

  return (
    <div className="recognition-grading" data-status={problemStatus || gradingStatus}>
      <span>Work</span>
      <span>{gradingDecisionLabel(problemStatus, gradingStatus)}</span>
      {(foundSolutions.length > 0 || missingSolutions.length > 0) && (
        <small>
          {gradingSolutionSummary(foundSolutions, missingSolutions)}
        </small>
      )}
      {grading.failed && (
        <small>{grading.error || 'Grading failed.'}</small>
      )}
    </div>
  );
}

function RecognitionEvidence({ result }) {
  const detection = result.detection || {};
  const segmentation = result.segmentation || {};
  const semantic = result.semantic || {};
  const selected = segmentation.selected || [];
  const candidates = segmentation.candidates || [];
  const lines = result.lines || [];

  return (
    <details className="recognition-evidence">
      <summary>Evidence</summary>

      <dl className="recognition-metrics">
        <div>
          <dt>Detector</dt>
          <dd>{detectionLabel(detection)}</dd>
        </div>
        <div>
          <dt>Segments</dt>
          <dd>{selected.length}/{candidates.length}</dd>
        </div>
        <div>
          <dt>Semantic</dt>
          <dd>{semanticLabel(semantic)}</dd>
        </div>
      </dl>

      {semantic.failed && semantic.error && (
        <p className="recognition-message">
          Semantic scoring failed: {semantic.error}
        </p>
      )}

      {lines.length > 0 && (
        <ul className="recognition-evidence-lines">
          {lines.map((line) => (
            <li key={`${line.candidateId}-${line.lineIndex}`}>
              <span className="recognition-evidence-title">
                {lineLabel(line)}
              </span>
              <span className="recognition-evidence-profiles">
                {(line.profiles || []).join(', ') || 'candidate'}
              </span>
              <CandidateList candidates={line.candidates} />
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

function CandidateList({ candidates = [], semantic = null, limit = 3 }) {
  const visible = candidates.slice(0, limit).filter((candidate) => candidate?.latex);
  if (visible.length === 0) return null;

  return (
    <ol className="recognition-candidates">
      {visible.map((candidate, index) => (
        <li key={`${candidate.latex}-${index}`}>
          <span>{candidate.latex}</span>
          <span>{candidateScoreLabel(candidate, semantic)}</span>
        </li>
      ))}
    </ol>
  );
}

function RecognizedLatexLine({ latex }) {
  return (
    <li>
      <RecognizedLatexLineText latex={latex} />
    </li>
  );
}

function RecognizedLatexLineText({ latex }) {
  const html = useMemo(() => {
    if (!latex) return '';
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode: false
    });
  }, [latex]);

  return (
    <span dangerouslySetInnerHTML={{ __html: html || latex }} />
  );
}

function candidateDebugStatus(candidate) {
  if (candidate.prediction?.failed || candidate.prediction?.timedOut) return 'failed';
  if (candidate.realtimeStatus === 'contested') return 'contested';
  if (candidate.realtimeStatus === 'provisional' || candidate.provisional) return 'provisional';
  if (candidate.realtimeStatus === 'pending' || candidate.realtimeStatus === 'running') return 'unread';
  if (candidate.selected) return 'selected';
  if (!candidate.latex) return 'unread';
  return 'discarded';
}

function candidateStatusLabel(candidate, status) {
  if (status === 'selected') return `selected L${Number(candidate.selectedLineIndex) + 1}`;
  if (status === 'contested') return 'contested';
  if (status === 'provisional') return candidate.prediction?.cached ? 'cached partial' : 'partial';
  if (status === 'failed') return 'failed';
  if (candidate.realtimeStatus === 'pending') return 'pending';
  if (candidate.realtimeStatus === 'running') return 'reading';
  if (status === 'unread') return 'not OCRed';
  if (candidate.prediction?.cached) return 'cached';
  return 'discarded';
}

function candidateScoreLabel(candidate, semantic) {
  const semanticCandidate = (semantic?.candidateScores || []).find((item) => (
    String(item.latex || '').trim() === String(candidate.latex || '').trim()
  ));
  if (typeof semanticCandidate?.score === 'number') return semanticCandidate.score.toFixed(2);
  if (typeof candidate.score === 'number') return candidate.score.toFixed(2);
  return '';
}

function semanticScoreLabel(semantic, timing) {
  const score = formatNumber(semantic?.semanticScore);
  const elapsed = timing?.sequentialSemanticElapsedSeconds ??
    timing?.contextualSemanticElapsedSeconds ??
    timing?.semanticElapsedSeconds;
  if (score === 'n/a') return formatSeconds(elapsed);
  return `${score} | ${formatSeconds(elapsed)}`;
}

function gradingDecisionLabel(problemStatus, status = '') {
  if (status === 'pending') return 'Pending';
  if (status === 'failed') return 'Failed';
  if (!problemStatus || problemStatus === 'n/a') return 'n/a';
  return problemStatus
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function auditStatusDebugLabel(status = '') {
  if (status === 'skipped') return 'Audit not selected';
  if (status === 'queued') return 'Audit queued';
  if (status === 'processing') return 'Audit processing';
  if (status === 'logged') return 'Log entered';
  if (status === 'disabled') return 'Audit disabled';
  if (status === 'error') return 'Audit error';
  return 'Audit waiting';
}

function auditTakingPlaceLabel(status = '') {
  if (['queued', 'processing', 'logged'].includes(status)) return 'Yes';
  if (status === 'skipped') return 'No';
  if (status === 'disabled') return 'Disabled';
  if (status === 'error') return 'Attempted';
  return 'Pending';
}

function lineClassificationLabel(classification = '') {
  if (!classification) return 'n/a';
  return classification
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function gradingLineDetail(step = {}) {
  const parts = [];
  if (step.solutionCoverage && step.solutionCoverage !== 'none') {
    parts.push(`coverage ${step.solutionCoverage}`);
  }
  if (step.matchedSolutions?.length) {
    parts.push(`matched ${step.matchedSolutions.join(', ')}`);
  }
  if (Number.isFinite(Number(step.selectedCandidateIndex))) {
    parts.push(`candidate ${Number(step.selectedCandidateIndex) + 1}`);
  }
  return parts.join(' | ') || 'no solution match';
}

function gradingSolutionSummary(foundSolutions = [], missingSolutions = []) {
  const parts = [];
  if (foundSolutions.length) parts.push(`found ${foundSolutions.join(', ')}`);
  if (missingSolutions.length) parts.push(`missing ${missingSolutions.join(', ')}`);
  return parts.join(' | ');
}

function bboxLabel(bbox) {
  if (!bbox) return 'n/a';
  return `${Math.round(bbox.xMin)},${Math.round(bbox.yMin)} ${Math.round(bbox.xMax - bbox.xMin)}x${Math.round(bbox.yMax - bbox.yMin)}`;
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : 'n/a';
}

function formatSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : 'n/a';
}

function debugStatusLabel(status, result) {
  if (status === 'pending' && result?.realtime?.components?.some((component) => component.contested)) {
    return 'Reviewing overlap';
  }
  if (status === 'pending' && result?.realtime?.components?.some((component) => component.hasResult)) {
    return 'Reading partial';
  }
  if (status !== 'complete') return statusLabel(status);
  const total = formatSeconds(result?.timing?.totalElapsedSeconds);
  return total === 'n/a' ? 'Complete' : total;
}

function lineLabel(line) {
  const score = Number(line.evidenceScore);
  const scoreText = Number.isFinite(score) ? ` | ${score.toFixed(2)}` : '';
  return `Line ${(line.lineIndex ?? 0) + 1}${scoreText}`;
}

function detectionLabel(detection) {
  if (detection.failed) return 'failed';
  if (detection.source === 'detector') {
    const elapsed = Number(detection.elapsedSeconds);
    return Number.isFinite(elapsed) ? `DBNet ${elapsed.toFixed(2)}s` : 'DBNet';
  }
  return detection.source || 'none';
}

function semanticLabel(semantic) {
  if (!semantic || semantic.source === 'disabled') return 'off';
  if (semantic.failed) return 'failed';
  const contextual = semantic.contextual?.candidateScores?.length || 0;
  const sequential = semantic.sequential?.lineScores?.length || 0;
  if (contextual || sequential) return `${contextual}/${sequential}`;
  return semantic.source || 'on';
}

function statusLabel(status) {
  if (status === 'pending') return 'Reading';
  if (status === 'complete') return 'LaTeX';
  if (status === 'error') return 'Offline';
  if (status === 'empty') return 'Empty';
  return 'Ready';
}
