import React, { useEffect, useRef } from 'react';

class CoordPickerEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private coord: React.MutableRefObject<Float32Array>;
  private canvasSize: { width: number; height: number } = { width: 0, height: 0 };
  private onChange: () => void;

  constructor(
    canvas: HTMLCanvasElement,
    coord: React.MutableRefObject<Float32Array>,
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
    ctx.fillRect(
      this.coord.current[0] * this.canvasSize.width,
      this.coord.current[1] * this.canvasSize.height,
      2,
      2
    );
  }

  private handleMouseDown = (e: MouseEvent) => {
    console.log('fu');
    const x = e.clientX - this.canvas.offsetLeft;
    const y = e.clientY - this.canvas.offsetTop;
    this.coord.current[0] = x / this.canvasSize.width;
    this.coord.current[1] = y / this.canvasSize.height;
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
    this.coord.current[0] = x / this.canvasSize.width;
    this.coord.current[1] = y / this.canvasSize.height;
    this.render();
    this.onChange();
  };

  public dispose() {
    this.canvas.onmousedown = null;
    this.canvas.onmouseup = null;
  }
}

interface CoordPickerProps {
  coord: React.MutableRefObject<Float32Array>;
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
        width={100}
        height={100}
        ref={canvas => {
          if (!canvas) {
            console.log('wtf');
            engine.current?.dispose();
            engine.current = null;
            return;
          }

          console.log('re-init');
          engine.current?.dispose();
          engine.current = new CoordPickerEngine(canvas, coord, onChange);
        }}
      />
    </div>
  );
};

export default React.memo(CoordPicker);
