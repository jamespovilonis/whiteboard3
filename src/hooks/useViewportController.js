import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { INITIAL_VIEWPORT, RESET_DRIFT_THRESHOLD } from '../whiteboard/constants.js';
import { isAwayFromViewport } from '../whiteboard/viewport.js';

export function useViewportController(initialViewport = INITIAL_VIEWPORT) {
  const [viewport, setViewport] = useState(initialViewport);
  const [homeViewport, setHomeViewport] = useState(initialViewport);
  const animationRef = useRef(null);

  const cancelViewportAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  const animateViewportTo = useCallback((targetViewport, duration = 260) => {
    cancelViewportAnimation();

    const start = { ...viewport };
    const startedAt = performance.now();

    const tick = (now) => {
      const t = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - Math.pow(1 - t, 3);

      setViewport({
        x: start.x + (targetViewport.x - start.x) * eased,
        y: start.y + (targetViewport.y - start.y) * eased,
        scale: start.scale + (targetViewport.scale - start.scale) * eased
      });

      if (t < 1) {
        animationRef.current = requestAnimationFrame(tick);
      } else {
        animationRef.current = null;
      }
    };

    animationRef.current = requestAnimationFrame(tick);
  }, [cancelViewportAnimation, viewport]);

  const moveHomeViewport = useCallback((targetViewport, duration = 420) => {
    setHomeViewport(targetViewport);
    animateViewportTo(targetViewport, duration);
  }, [animateViewportTo]);

  const resetViewport = useCallback(() => {
    animateViewportTo(homeViewport, 260);
  }, [animateViewportTo, homeViewport]);

  const showReset = useMemo(() => (
    isAwayFromViewport(viewport, homeViewport, RESET_DRIFT_THRESHOLD)
  ), [homeViewport, viewport]);

  useEffect(() => cancelViewportAnimation, [cancelViewportAnimation]);

  return {
    viewport,
    setViewport,
    homeViewport,
    showReset,
    resetViewport,
    animateViewportTo,
    moveHomeViewport
  };
}
