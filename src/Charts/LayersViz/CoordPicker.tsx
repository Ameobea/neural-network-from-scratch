import React, { useEffect, useRef } from 'react';

class CoordPickerEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private coord: Float32Array;
  private canvasSize: { width: number; height: number } = { width: 0, height: 0 };
  private onChange: () => void;

  constructor(
    canvas: HTMLCanvasElement,
    coord: Float32Array,
    onChange: () => void
  ) {
    canvas.onmousedown = this.handleMouseDown;
    canvas.onmouseup = this.handleMouseUp;
    this.canvas = canvas;
    this.onChange = onChange;

    this.ctx = canvas.getContext('2d')!;
    this.canvasSize = { width: canvas.width, height: canvas.height };
    this.coord = coord;
    this.render();
  }

  private render() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvasSize.width, this.canvasSize.height);
    ctx.fillStyle = '#ff0000';
    ctx.beginPath();
    ctx.arc(
      this.coord[0] * this.canvasSize.width,
      (1 - this.coord[1]) * this.canvasSize.height,
      4,
      0,
      2 * Math.PI
    );
    ctx.fill();
  }

  private handleMouseDown = (e: MouseEvent) => {
    const x = e.clientX - this.canvas.offsetLeft;
    const y = e.clientY - this.canvas.offsetTop;
    this.coord[0] = x / this.canvasSize.width;
    this.coord[1] = 1 - y / this.canvasSize.height;
    this.render();
    this.onChange();

    this.canvas.onmousemove = this.handleMouseMove;
  };

  private handleMouseUp = (_e: MouseEvent) => {
    this.canvas.onmousemove = null;
  };

  private handleMouseMove = (e: MouseEvent) => {
    const x = e.clientX - this.canvas.offsetLeft;
    const y = e.clientY - this.canvas.offsetTop;
    this.coord[0] = x / this.canvasSize.width;
    this.coord[1] = 1 - y / this.canvasSize.height;
    this.render();
    this.onChange();
  };

  public dispose() {
    this.canvas.onmousedown = null;
    this.canvas.onmouseup = null;
  }
}

interface CoordPickerProps {
  coord: Float32Array;
  onChange: () => void;
}

const CoordPicker: React.FC<CoordPickerProps> = ({ coord, onChange }) => {
  const engine = useRef<CoordPickerEngine | null>(null);

  useEffect(() => () => {
    if (engine.current) {
      engine.current.dispose();
    }
  });

  return (
    <div className='coord-picker'>
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
    </div>
  );
};

export default React.memo(CoordPicker);
