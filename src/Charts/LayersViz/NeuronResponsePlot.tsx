import React from 'react';
import * as R from 'ramda';

import { NNContext } from 'src/NNContext';

const NEURON_RESPONSE_PLOT_SIZE = 250;

class NeuronResponsePlotEngine {
  private nnCtx: NNContext;
  private ctx: CanvasRenderingContext2D;
  private selectedNeuron: { layerIx: number; neuronIx: number } | null = null;

  constructor(nnCtx: NNContext, ctx: CanvasRenderingContext2D) {
    this.nnCtx = nnCtx;
    this.ctx = ctx;
  }

  private async render() {
    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

    if (!this.selectedNeuron) {
      return;
    }

    const response = await this.nnCtx.getNeuronResponse(
      this.selectedNeuron.layerIx,
      this.selectedNeuron.neuronIx,
      NEURON_RESPONSE_PLOT_SIZE
    );
    if (!response) {
      return;
    }

    const imgData = new ImageData(
      new Uint8ClampedArray(response.buffer),
      NEURON_RESPONSE_PLOT_SIZE,
      NEURON_RESPONSE_PLOT_SIZE
    );
    this.ctx.putImageData(imgData, 0, 0);
  }

  public setSelectedNeuron(selectedNeuron: { layerIx: number; neuronIx: number } | null) {
    if (R.equals(this.selectedNeuron, selectedNeuron)) {
      return;
    }

    this.selectedNeuron = selectedNeuron;
    this.render();
  }
}

interface NeuronResponsePlotProps {
  nnCtx: NNContext;
  selectedNeuron: { layerIx: number; neuronIx: number } | null;
}

class NeuronResponsePlot extends React.Component<NeuronResponsePlotProps> {
  private engine: NeuronResponsePlotEngine | null = null;

  public componentDidUpdate() {
    this.engine?.setSelectedNeuron(this.props.selectedNeuron);
  }

  public render = () => {
    return (
      <canvas
        className='neuron-response-plot'
        width={NEURON_RESPONSE_PLOT_SIZE}
        height={NEURON_RESPONSE_PLOT_SIZE}
        ref={canvas => {
          if (!canvas) {
            this.engine = null;
            return;
          }

          this.engine = new NeuronResponsePlotEngine(this.props.nnCtx, canvas.getContext('2d')!);
          this.engine.setSelectedNeuron(this.props.selectedNeuron);
        }}
      />
    );
  };
}

export default NeuronResponsePlot;
