import katex from 'katex';
import { useMemo, useState } from 'react';

const PLACEHOLDER_RESPONSE = {
  before: 'Here is one way to think about the next step.',
  latex: '2x = 8',
  after: 'When you are ready, you can ask for a hint or submit your answer.'
};

export default function ModelShell() {
  const [isOpen, setIsOpen] = useState(false);
  const equationHtml = useMemo(() => {
    return katex.renderToString(PLACEHOLDER_RESPONSE.latex, {
      throwOnError: false,
      displayMode: true
    });
  }, []);

  if (!isOpen) {
    return (
      <button
        className="model-shell-toggle"
        type="button"
        aria-label="Open model response"
        onClick={() => setIsOpen(true)}
      >
        M
      </button>
    );
  }

  return (
    <aside className="model-shell-panel" aria-label="Model response">
      <button
        className="model-shell-minimize"
        type="button"
        aria-label="Minimize model response"
        onClick={() => setIsOpen(false)}
      >
        -
      </button>

      <div className="model-shell-copy">
        <p>{PLACEHOLDER_RESPONSE.before}</p>
        <div
          className="model-shell-equation"
          dangerouslySetInnerHTML={{ __html: equationHtml }}
        />
        <p>{PLACEHOLDER_RESPONSE.after}</p>
      </div>

      <div className="model-shell-actions">
        <button type="button">Ask for help</button>
        <button type="button">Submit answer</button>
      </div>
    </aside>
  );
}
