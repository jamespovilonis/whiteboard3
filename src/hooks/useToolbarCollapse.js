import { useCallback, useState } from 'react';
import { DEFAULT_TOOL } from '../whiteboard/constants.js';

export function useToolbarCollapse(initialTool = DEFAULT_TOOL) {
  const [activeTool, setActiveTool] = useState(initialTool);
  const [toolbarForceCollapsed, setToolbarForceCollapsed] = useState(false);

  const selectTool = useCallback((tool) => {
    setActiveTool(tool);
    setToolbarForceCollapsed(false);
  }, []);

  const collapseToolbarForDrawing = useCallback(() => {
    setToolbarForceCollapsed(true);
  }, []);

  const requestToolbarOpen = useCallback(() => {
    setToolbarForceCollapsed(false);
  }, []);

  return {
    activeTool,
    toolbarForceCollapsed,
    selectTool,
    collapseToolbarForDrawing,
    requestToolbarOpen
  };
}
