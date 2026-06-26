import { useEffect, useRef, useState } from 'react';
import { COLOR_SWATCHES } from '../whiteboard/constants.js';

const TOOLBAR_TRANSITION_MS = 450;

const TOOLS = [
  { id: 'mouse', label: 'Pan tool', icon: '/toolbar-icons/cursor.png', shortcut: 'M' },
  { id: 'pen', label: 'Pen tool', icon: '/toolbar-icons/pencil.png', shortcut: 'P' },
  { id: 'eraser', label: 'Eraser tool', icon: '/toolbar-icons/eraser.png', shortcut: 'E' }
];

const ACTION_ICONS = {
  undo: '/toolbar-icons/undo.png',
  redo: '/toolbar-icons/forward.png',
  clear: '/toolbar-icons/trash-can.png'
};

export default function Toolbar({
  activeTool,
  penColor,
  sliderValue,
  onToolChange,
  onColorChange,
  onSliderChange,
  onUndo,
  onRedo,
  onClear,
  forceCollapsed,
  onRequestOpen
}) {
  const [isPointerOpen, setIsPointerOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const closeTimerRef = useRef(null);
  const activeToolConfig = TOOLS.find((tool) => tool.id === activeTool) || TOOLS[1];
  const isFullyCollapsed = forceCollapsed && !isClosing;
  const isOpen = isPointerOpen && !forceCollapsed && !isClosing;

  useEffect(() => {
    if (!forceCollapsed) {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
      setIsClosing(false);
      return undefined;
    }

    setIsPointerOpen(false);
    setIsClosing(true);
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      setIsClosing(false);
    }, TOOLBAR_TRANSITION_MS);

    return () => {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
    };
  }, [forceCollapsed]);

  return (
    <div
      className={`toolbar ${isOpen ? 'is-open' : ''} ${isClosing ? 'is-closing' : ''} ${isFullyCollapsed ? 'force-collapsed' : ''}`}
      role="toolbar"
      aria-label="Drawing tools"
      onPointerEnter={() => {
        if (isClosing) return;
        setIsPointerOpen(true);
        onRequestOpen?.();
      }}
      onPointerLeave={() => setIsPointerOpen(false)}
      onPointerDown={() => {
        if (forceCollapsed) onRequestOpen?.();
      }}
    >
      <span className="toolbar-toggle" aria-hidden="true">
        <IconImage src={activeToolConfig.icon} alt="" />
      </span>

      <div className="toolbar-content">
        {TOOLS.map(({ id, label, icon, shortcut }) => (
          <button
            key={id}
            className={`tool-btn ${activeTool === id ? 'active' : ''}`}
            type="button"
            title={`${label} (${shortcut})`}
            aria-label={label}
            onClick={() => onToolChange(id)}
          >
            <IconImage src={icon} alt="" />
          </button>
        ))}

        <div className="toolbar-divider" role="separator" />

        <ColorPicker
          color={penColor}
          onColorChange={onColorChange}
        />

        <input
          id="sizePicker"
          type="range"
          min="1"
          max="10"
          value={sliderValue}
          title="Brush size"
          aria-label="Brush size"
          onChange={(event) => onSliderChange(event.target.value)}
        />
        <span id="sizeValue" aria-live="polite">{sliderValue}</span>

        <div className="toolbar-divider" role="separator" />

        <button className="tool-btn" type="button" title="Undo (Ctrl+Z)" aria-label="Undo last stroke" onClick={onUndo}>
          <IconImage src={ACTION_ICONS.undo} alt="" />
        </button>
        <button className="tool-btn" type="button" title="Redo (Ctrl+Shift+Z)" aria-label="Redo last undo" onClick={onRedo}>
          <IconImage src={ACTION_ICONS.redo} alt="" />
        </button>
        <button className="tool-btn" type="button" title="Clear all (Shift+C)" aria-label="Clear all" onClick={onClear}>
          <IconImage src={ACTION_ICONS.clear} alt="" />
        </button>
      </div>
    </div>
  );
}

function IconImage({ src, alt }) {
  return (
    <img
      className="tool-icon"
      src={src}
      alt={alt}
      draggable="false"
    />
  );
}

function ColorPicker({ color, onColorChange }) {
  return (
    <div className="color-picker-wrap" title="Color" aria-label="Stroke color">
      <button
        className="color-btn"
        style={{ background: color }}
        type="button"
        aria-label="Change color"
      />
      <div className="color-picker-dropdown">
        {COLOR_SWATCHES.map((swatch) => (
          <button
            key={swatch}
            className={`color-swatch ${swatch === color ? 'active' : ''}`}
            type="button"
            aria-label={`Set stroke color ${swatch}`}
            style={{ background: swatch }}
            onClick={() => onColorChange(swatch)}
          />
        ))}
      </div>
    </div>
  );
}
