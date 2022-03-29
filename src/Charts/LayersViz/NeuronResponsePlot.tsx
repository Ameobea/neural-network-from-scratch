import React from 'react';

import { NNContext } from 'src/NNContext';
import { deregisterVizUpdateCB, registerVizUpdateCB } from '../vizControls';

const NEURON_RESPONSE_PLOT_SIZE = 250;

class NeuronResponsePlotEngine {
  private nnCtx: NNContext;
  private ctx: CanvasRenderingContext2D;
  private selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null = {
    layerIx: 'init_output',
    neuronIx: 0,
  };
  private intervalHandle: number | null = null;
  private isRendering = false;

  constructor(nnCtx: NNContext, ctx: CanvasRenderingContext2D) {
    this.nnCtx = nnCtx;
    this.ctx = ctx;
    this.intervalHandle = setInterval(this.render, 100);

    registerVizUpdateCB(this.forceRender);
  }

  private render = async (force = false) => {
    if (this.isRendering || (!force && !this.nnCtx.isRunning)) {
      return;
    }

    if (!this.selectedNeuron) {
      this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
      return;
    }

    this.isRendering = true;
    const response = await this.nnCtx.getNeuronResponse(
      this.selectedNeuron.layerIx === 'init_output' ? -1 : this.selectedNeuron.layerIx,
      this.selectedNeuron.neuronIx,
      NEURON_RESPONSE_PLOT_SIZE
    );
    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    if (!response) {
      this.isRendering = false;
      return;
    }

    const imgData = new ImageData(
      new Uint8ClampedArray(response.buffer),
      NEURON_RESPONSE_PLOT_SIZE,
      NEURON_RESPONSE_PLOT_SIZE
    );
    this.ctx.putImageData(imgData, 0, 0);
    this.isRendering = false;
  };

  public forceRender = () => this.render(true);

  public setSelectedNeuron = (
    selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null
  ) => {
    if (
      (!this.selectedNeuron && !selectedNeuron) ||
      (this.selectedNeuron &&
        selectedNeuron &&
        this.selectedNeuron.layerIx === selectedNeuron.layerIx &&
        this.selectedNeuron.neuronIx === selectedNeuron.neuronIx)
    ) {
      return;
    }

    this.selectedNeuron = selectedNeuron;
    this.forceRender();
  };

  public dispose() {
    deregisterVizUpdateCB(this.forceRender);
    if (this.intervalHandle !== null) {
      clearInterval(this.intervalHandle);
    }
  }
}

interface NeuronResponsePlotProps {
  nnCtx: NNContext;
  selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null;
  style: React.CSSProperties;
}

class NeuronResponsePlot extends React.Component<NeuronResponsePlotProps> {
  private engine: NeuronResponsePlotEngine | null = null;

  public componentDidUpdate() {
    this.engine?.setSelectedNeuron(this.props.selectedNeuron);
  }

  private canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (this.engine) {
      this.engine.dispose();
      this.engine = null;
    }

    if (!canvas) {
      this.engine = null;
      return;
    }

    this.engine = new NeuronResponsePlotEngine(this.props.nnCtx, canvas.getContext('2d')!);
    this.engine.setSelectedNeuron(this.props.selectedNeuron);
    this.engine.forceRender();
  };

  public render = () => (
    <canvas
      style={this.props.style}
      className='neuron-response-plot'
      width={NEURON_RESPONSE_PLOT_SIZE}
      height={NEURON_RESPONSE_PLOT_SIZE}
      ref={this.canvasRef}
    />
  );
}

export default NeuronResponsePlot;
