import React from 'react';

import type { NNContext } from 'src/NNContext';
import { delay } from 'src/util';

interface ColorsScaleLegendProps {
  nnCtx: NNContext;
}

const CANVAS_WIDTH = 42;
const CANVAS_HEIGHT = 260;
const SCALE_WIDTH = 18;
const SCALE_HEIGHT = 235;
const MAX = 2.5;
const MIN = -2.5;

const dpr = Math.floor(window.devicePixelRatio);

const drawColorsScaleLegend = async (ctx: CanvasRenderingContext2D, nnCtx: NNContext) => {
  const colorData = await Promise.race([
    nnCtx.getColorScaleLegend(MIN, MAX, SCALE_WIDTH * dpr, SCALE_HEIGHT * dpr),
    delay(50),
  ] as const);
  if (!colorData) {
    setTimeout(() => drawColorsScaleLegend(ctx, nnCtx), 50);
    return;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  const imageData = new ImageData(
    new Uint8ClampedArray(colorData.buffer),
    SCALE_WIDTH * dpr,
    SCALE_HEIGHT * dpr
  );
  ctx.putImageData(imageData, 0, (CANVAS_HEIGHT - SCALE_HEIGHT) / 2);

  ctx.font = '12px "PT Sans"';
  ctx.fillStyle = '#ccc';
  ctx.fillText(`${MAX.toFixed(1)}`, 22, (CANVAS_HEIGHT - SCALE_HEIGHT) / 2 + 6);
  ctx.fillText('0', 22, CANVAS_HEIGHT / 2 + 3);
  ctx.fillText(`${MIN.toFixed(1)}`, 22, (CANVAS_HEIGHT + SCALE_HEIGHT) / 2 + 3);
};

const ColorsScaleLegend: React.FC<ColorsScaleLegendProps> = ({ nnCtx }) => (
  <canvas
    className='colors-scale-legend'
    width={CANVAS_WIDTH * dpr}
    height={CANVAS_HEIGHT * dpr}
    style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}
    ref={canvas => {
      if (!canvas) {
        return;
      }
      drawColorsScaleLegend(canvas.getContext('2d')!, nnCtx);
    }}
  />
);

export default React.memo(ColorsScaleLegend);
