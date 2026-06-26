import { useEffect } from 'react';

export function useWhiteboardShortcuts({
  engineRef,
  onSelectTool,
  onToggleDebugBoxes
}) {
  useEffect(() => {
    const isEditingText = () => {
      const tag = document.activeElement?.tagName || '';
      return tag === 'INPUT' || tag === 'TEXTAREA';
    };

    const handleKeyDown = (event) => {
      if (isEditingText()) return;

      const key = event.key.toLowerCase();
      const isCommand = event.ctrlKey || event.metaKey;

      if (isCommand && key === 'b') {
        event.preventDefault();
        onToggleDebugBoxes();
        return;
      }

      if (key === 'p' && !isCommand && !event.altKey) {
        event.preventDefault();
        onSelectTool('pen');
        return;
      }

      if (key === 'm' && !isCommand && !event.altKey) {
        event.preventDefault();
        onSelectTool('mouse');
        return;
      }

      if (key === 'e' && !isCommand && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        onSelectTool('eraser');
        return;
      }

      if (isCommand && key === 'z' && !event.shiftKey) {
        event.preventDefault();
        engineRef.current?.undo();
        return;
      }

      if ((isCommand && key === 'z' && event.shiftKey) || (isCommand && key === 'y')) {
        event.preventDefault();
        engineRef.current?.redo();
        return;
      }

      if (key === 'c' && event.shiftKey && !isCommand && !event.altKey) {
        event.preventDefault();
        engineRef.current?.clear();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [engineRef, onSelectTool, onToggleDebugBoxes]);
}
