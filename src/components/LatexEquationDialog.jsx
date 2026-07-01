import katex from 'katex';
import { useMemo, useState } from 'react';

const DEFAULT_LATEX = '';
const PROBLEM_MODES = {
  solve: {
    problemType: 'equation-solving',
    title: 'Enter an equation',
    copy: 'Add the LaTeX problem you want to solve on the whiteboard.',
    label: 'LaTeX equation',
    placeholder: '\\frac{x}{2} + 5 = 13',
    previewLabel: 'Equation preview'
  },
  evaluate: {
    problemType: 'evaluate-expression',
    title: 'Enter an expression',
    copy: 'Add the numeric LaTeX expression you want to evaluate on the whiteboard.',
    label: 'LaTeX expression',
    placeholder: '\\frac{1}{2} + \\frac{2}{4}',
    previewLabel: 'Expression preview'
  }
};

export default function LatexEquationDialog({ onSubmit }) {
  const [latex, setLatex] = useState(DEFAULT_LATEX);
  const [mode, setMode] = useState('solve');
  const modeConfig = PROBLEM_MODES[mode] || PROBLEM_MODES.solve;
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
    onSubmit?.({
      latex: trimmedLatex,
      problemType: modeConfig.problemType
    });
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
          <h1 id="latex-dialog-title">{modeConfig.title}</h1>
          <p>{modeConfig.copy}</p>
        </div>

        <div className="latex-dialog-mode" role="group" aria-label="Problem type">
          <button
            type="button"
            className={mode === 'solve' ? 'is-selected' : ''}
            aria-pressed={mode === 'solve'}
            data-testid="latex-problem-type-solve"
            onClick={() => setMode('solve')}
          >
            Solve
          </button>
          <button
            type="button"
            className={mode === 'evaluate' ? 'is-selected' : ''}
            aria-pressed={mode === 'evaluate'}
            data-testid="latex-problem-type-evaluate"
            onClick={() => setMode('evaluate')}
          >
            Evaluate
          </button>
        </div>

        <label className="latex-dialog-field">
          <span>{modeConfig.label}</span>
          <textarea
            value={latex}
            rows={3}
            autoFocus
            data-testid="latex-equation-input"
            placeholder={modeConfig.placeholder}
            onChange={(event) => setLatex(event.target.value)}
          />
        </label>

        <div
          className="latex-dialog-preview"
          aria-label={modeConfig.previewLabel}
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
