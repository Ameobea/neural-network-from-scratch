import React, { useRef } from 'react';

class CoordPickerEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private coord: Float32Array;
  private canvasSize: { width: number; height: number } = { width: 0, height: 0 };
  private onChange: () => void;

  constructor(canvas: HTMLCanvasElement, coord: Float32Array, onChange: () => void) {
    this.canvas = canvas;
    this.registerCanvasEvents();
    this.onChange = onChange;

    this.ctx = canvas.getContext('2d')!;
    this.canvasSize = { width: canvas.width, height: canvas.height };
    this.coord = coord;
    this.render();
  }

  private registerCanvasEvents() {
    this.canvas.onmousedown = this.handleMouseDown;
    this.canvas.onmouseup = this.handleMouseUp;
    this.canvas.ontouchmove = this.handleTouchMove;
    this.canvas.ontouchstart = this.handleTouchStart;
    this.canvas.ontouchend = this.handleTouchEnd;
  }

  private render() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvasSize.width, this.canvasSize.height);
    ctx.fillStyle = '#ddffdd';
    // Draw crosshair
    ctx.beginPath();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    const x = this.coord[0] * this.canvasSize.width;
    const y = (1 - this.coord[1]) * this.canvasSize.height;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, this.canvasSize.height);
    ctx.moveTo(0, y);
    ctx.lineTo(this.canvasSize.width, y);
    ctx.stroke();
  }

  private getCoordFromMouseEvent = (e: MouseEvent) => {
    const { canvas } = this;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = 1 - (e.clientY - rect.top) / rect.height;
    return [x, y];
  };

  private getCoordFromTouchEvent = (e: TouchEvent) => {
    const { canvas } = this;
    const rect = canvas.getBoundingClientRect();
    const x = (e.touches[0].clientX - rect.left) / rect.width;
    const y = 1 - (e.touches[0].clientY - rect.top) / rect.height;
    return [x, y];
  };

  private handleMouseDown = (e: MouseEvent) => {
    const coord = this.getCoordFromMouseEvent(e);
    this.coord.set(coord);
    this.render();
    this.onChange();

    this.canvas.onmousemove = this.handleMouseMove;
    this.canvas.ontouchmove = this.handleTouchMove;
  };

  private handleMouseUp = (_e: MouseEvent) => {
    this.canvas.onmousemove = null;
    this.canvas.ontouchmove = null;
  };

  private handleMouseMove = (e: MouseEvent) => {
    const coord = this.getCoordFromMouseEvent(e);
    this.coord.set(coord);
    this.render();
    this.onChange();
  };

  private handleTouchStart = (e: TouchEvent) => {
    const coord = this.getCoordFromTouchEvent(e);
    this.coord.set(coord);
    this.render();
    this.onChange();
    e.preventDefault();

    this.canvas.onmousemove = this.handleMouseMove;
    this.canvas.ontouchmove = this.handleTouchMove;
  };

  private handleTouchEnd = (_e: TouchEvent) => {
    this.canvas.onmousemove = null;
    this.canvas.ontouchmove = null;
  };

  private handleTouchMove = (e: TouchEvent) => {
    const coord = this.getCoordFromTouchEvent(e);
    this.coord.set(coord);
    this.render();
    this.onChange();
    e.preventDefault();
  };

  public dispose() {
    this.canvas.onmousedown = null;
    this.canvas.onmouseup = null;
    this.canvas.onmousemove = null;
    this.canvas.ontouchmove = null;
    this.canvas.ontouchstart = null;
    this.canvas.ontouchend = null;
  }
}

interface CoordPickerProps {
  coord: Float32Array;
  onChange: () => void;
  style: React.CSSProperties;
}

const CoordPicker: React.FC<CoordPickerProps> = ({ coord, onChange, style }) => {
  const engine = useRef<CoordPickerEngine | null>(null);

  return (
    <div className='coord-picker' style={style}>
      <canvas
        width={250}
        height={250}
        ref={canvas => {
          if (!canvas) {
            engine.current?.dispose();
            engine.current = null;
            return;
          }

          engine.current?.dispose();
          engine.current = new CoordPickerEngine(canvas, coord, onChange);
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          transform: 'translate(-18px, 9px)',
        }}
      >
        0, 0
      </div>
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          transform: 'translate(-19px, -13px)',
        }}
      >
        0, 1
      </div>
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          right: 0,
          transform: 'translate(19px, 9px)',
        }}
      >
        1, 0
      </div>
      <div
        style={{
          position: 'absolute',
          top: 0,
          right: 0,
          transform: 'translate(19px, -13px)',
        }}
      >
        1, 1
      </div>
    </div>
  );
};

export default React.memo(CoordPicker);
