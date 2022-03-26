import React, { useCallback, useRef } from 'react';

import { NNContext } from 'src/NNContext';
import { useInterval } from 'src/util';
import CoordPicker from './CoordPicker';

import './LayersViz.css';

interface LayersVizProps {
  nnCtx: NNContext;
}

const VIZ_SCALE_MULTIPLIER = 16;

const drawLayer = (ctx: CanvasRenderingContext2D, layerColors: Uint8Array, layerIx: number) => {
  const colorData: Uint8ClampedArray = new Uint8ClampedArray(layerColors.buffer);
  const pixelCount = colorData.length / 4;
  const width = pixelCount / VIZ_SCALE_MULTIPLIER;
  const height = pixelCount / width;
  const imageData = new ImageData(colorData, width, height);
  imageData.data.set(colorData, 0);
  ctx.putImageData(imageData, 0, (layerIx + 1) * (VIZ_SCALE_MULTIPLIER * 2.5));
};

const LayersViz: React.FC<LayersVizProps> = ({ nnCtx }) => {
  const coord = useRef<Float32Array>(new Float32Array([0, 0]));
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const isRendering = useRef(false);

  const maybeRender = useCallback(
    async (force = false) => {
      if ((!nnCtx.isRunning && !force) || isRendering.current) {
        return;
      }

      const ctx = ctxRef.current;
      if (!ctx) {
        return;
      }

      isRendering.current = true;
      const vizData = await nnCtx.getVizData(new Float32Array(coord.current));
      isRendering.current = false;

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      if (!vizData) {
        return;
      }

      for (
        let hiddenLayerIx = 0;
        hiddenLayerIx < vizData.hiddenLayerColors.length;
        hiddenLayerIx++
      ) {
        drawLayer(ctx, vizData.hiddenLayerColors[hiddenLayerIx], hiddenLayerIx);
      }

      drawLayer(ctx, vizData.outputLayerColors, vizData.hiddenLayerColors.length);
    },
    [nnCtx]
  );

  useInterval(maybeRender, 200);

  return (
    <div className='layers-viz'>
      <canvas
        width={VIZ_SCALE_MULTIPLIER * 128}
        height={200}
        ref={canvas => {
          if (!canvas) {
            ctxRef.current = null;
            return;
          }

          const ctx = canvas.getContext('2d')!;
          ctxRef.current = ctx;
        }}
      />
      <CoordPicker coord={coord} onChange={useCallback(() => maybeRender(true), [maybeRender])} />
    </div>
  );
};

export default LayersViz;
