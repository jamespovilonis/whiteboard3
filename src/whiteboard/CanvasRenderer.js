export class CanvasRenderer {
  constructor({ bgCanvas, fgCanvas, viewport }) {
    this.bgCanvas = bgCanvas;
    this.fgCanvas = fgCanvas;
    this.bgCtx = bgCanvas.getContext('2d');
    this.fgCtx = fgCanvas.getContext('2d');
    this.viewport = { ...viewport };
    this.dpr = window.devicePixelRatio || 1;
  }

  setViewport(viewport) {
    this.viewport = { ...viewport };
  }

  resize() {
    const width = this.fgCanvas.clientWidth;
    const height = this.fgCanvas.clientHeight;
    this.dpr = window.devicePixelRatio || 1;

    for (const canvas of [this.bgCanvas, this.fgCanvas]) {
      canvas.width = Math.max(1, Math.round(width * this.dpr));
      canvas.height = Math.max(1, Math.round(height * this.dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
  }

  redraw(strokes, strokeSmoother) {
    this.clearBackground();
    this.drawBoardSurface();
    this.withBoardTransform(this.bgCtx, () => {
      for (const stroke of strokes) {
        if (stroke.outlinePoints && stroke.outlinePoints.length >= 3) {
          strokeSmoother.render(this.bgCtx, stroke.outlinePoints, stroke.color);
        }
      }
    });
    this.clearForeground();
  }

  renderLiveStroke(outline, color, strokeSmoother) {
    this.clearForeground();
    this.withBoardTransform(this.fgCtx, () => {
      strokeSmoother.render(this.fgCtx, outline, color);
    });
  }

  renderEraserPreview(point, eraserSize) {
    this.clearForeground();
    this.withBoardTransform(this.fgCtx, () => {
      this.fgCtx.beginPath();
      this.fgCtx.arc(point.x, point.y, eraserSize / 2, 0, Math.PI * 2);
      this.fgCtx.strokeStyle = 'rgba(180, 180, 200, 0.6)';
      this.fgCtx.lineWidth = 2 / this.viewport.scale;
      this.fgCtx.stroke();
      this.fgCtx.fillStyle = 'rgba(180, 180, 200, 0.15)';
      this.fgCtx.fill();
    });
  }

  clearForeground() {
    this.clearCanvas(this.fgCtx, this.fgCanvas);
  }

  clearBackground() {
    this.clearCanvas(this.bgCtx, this.bgCanvas);
  }

  drawBoardSurface() {
    this.bgCtx.save();
    this.bgCtx.fillStyle = '#fefff1';
    this.bgCtx.fillRect(0, 0, this.bgCanvas.width / this.dpr, this.bgCanvas.height / this.dpr);
    this.bgCtx.restore();
  }

  withBoardTransform(ctx, draw) {
    ctx.save();
    ctx.translate(-this.viewport.x * this.viewport.scale, -this.viewport.y * this.viewport.scale);
    ctx.scale(this.viewport.scale, this.viewport.scale);
    draw();
    ctx.restore();
  }

  clearCanvas(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width / this.dpr, canvas.height / this.dpr);
  }
}
