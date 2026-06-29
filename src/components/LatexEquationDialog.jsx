import katex from 'katex';
import { useMemo, useState } from 'react';

const DEFAULT_LATEX = '';

export default function LatexEquationDialog({ onSubmit }) {
  const [latex, setLatex] = useState(DEFAULT_LATEX);
  const trimmedLatex = latex.trim();
  const previewHtml = useMemo(() => {
    const previewLatex = trimmedLatex || '\\square';
    try {
      return katex.renderToString(previewLatex, {
        throwOnError: false,
        displayMode: true
      });
    } catch {
      return previewLatex;
    }
  }, [trimmedLatex]);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!trimmedLatex) return;
    onSubmit?.(trimmedLatex);
    setLatex(DEFAULT_LATEX);
  };

  return (
    <div className="latex-dialog-backdrop" role="presentation">
      <form
        className="latex-dialog"
        aria-labelledby="latex-dialog-title"
        onSubmit={handleSubmit}
      >
        <div className="latex-dialog-copy">
          <h1 id="latex-dialog-title">Enter an equation</h1>
          <p>Add the LaTeX problem you want to solve on the whiteboard.</p>
        </div>

        <label className="latex-dialog-field">
          <span>LaTeX equation</span>
          <textarea
            value={latex}
            rows={3}
            autoFocus
            data-testid="latex-equation-input"
            placeholder="\\frac{x}{2} + 5 = 13"
            onChange={(event) => setLatex(event.target.value)}
          />
        </label>

        <div
          className="latex-dialog-preview"
          aria-label="Equation preview"
          dangerouslySetInnerHTML={{ __html: previewHtml }}
        />

        <div className="latex-dialog-actions">
          <button
            type="submit"
            data-testid="latex-equation-submit"
            disabled={!trimmedLatex}
          >
            Start problem
          </button>
        </div>
      </form>
    </div>
  );
}
