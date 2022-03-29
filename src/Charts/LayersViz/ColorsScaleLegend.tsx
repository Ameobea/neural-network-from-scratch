import React from 'react';

import type { NNContext } from 'src/NNContext';

interface ColorsScaleLegendProps {
  nnCtx: NNContext;
}

const HEIGHT = 280;
const WIDTH = 42;
const MAX = 2.5;
const MIN = -2.5;

const dpr = Math.floor(window.devicePixelRatio);

const drawColorsScaleLegend = async (ctx: CanvasRenderingContext2D, nnCtx: NNContext) => {
  const colorData: Uint8Array = await nnCtx.getColorScaleLegend(MIN, MAX, 18, 260);
  const imageData = new ImageData(new Uint8ClampedArray(colorData.buffer), 18, 260);
  ctx.putImageData(imageData, 0, (HEIGHT - 260) / 2);

  ctx.font = '12px "PT Sans"';
  ctx.fillStyle = '#ccc';
  ctx.fillText(`${MAX.toFixed(1)}`, 22, (HEIGHT - 260) / 2 + 6);
  ctx.fillText('0', 22, HEIGHT / 2 + 3);
  ctx.fillText(`${MIN.toFixed(1)}`, 22, (HEIGHT + 260) / 2 + 3);
};

const ColorsScaleLegend: React.FC<ColorsScaleLegendProps> = ({ nnCtx }) => (
  <canvas
    className='colors-scale-legend'
    width={WIDTH * dpr}
    height={HEIGHT * dpr}
    style={{ width: WIDTH, height: HEIGHT }}
    ref={canvas => {
      if (!canvas) {
        return;
      }
      drawColorsScaleLegend(canvas.getContext('2d')!, nnCtx);
    }}
  />
);

export default React.memo(ColorsScaleLegend);
