import katex from 'katex';
import { useEffect, useMemo, useRef, useState } from 'react';

const SHELL_TRANSITION_MS = 450;

export default function ModelShell({ response, onSubmitAnswer }) {
  const [mode, setMode] = useState('closed');
  const closeTimerRef = useRef(null);
  const equationHtml = useMemo(() => {
    return katex.renderToString(response.latex, {
      throwOnError: false,
      displayMode: true
    });
  }, [response.latex]);

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

  const shouldShowToggle = mode === 'closed' || mode === 'closing';

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

      {mode !== 'closed' && (
        <aside
          className={`model-shell-panel ${mode === 'closing' ? 'is-closing' : ''}`}
          aria-label="Model response"
          aria-hidden={mode === 'closing'}
        >
          <button
            className="model-shell-minimize"
            type="button"
            aria-label="Minimize model response"
            onClick={closeShell}
          >
            -
          </button>

          <div className="model-shell-copy">
            <p>{response.before}</p>
            <div
              className="model-shell-equation"
              dangerouslySetInnerHTML={{ __html: equationHtml }}
            />
            <p>{response.after}</p>
          </div>

          <div className="model-shell-actions">
            <button type="button">Ask for help</button>
            <button type="button" onClick={onSubmitAnswer}>Submit</button>
          </div>
        </aside>
      )}
    </>
  );
}
