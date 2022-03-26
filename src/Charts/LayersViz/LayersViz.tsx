import React from 'react';

import { NNContext } from 'src/NNContext';
import CoordPicker from './CoordPicker';

import './LayersViz.css';

const PADDING_TOP = 10;
const VIZ_SCALE_MULTIPLIER = 16;
const LAYER_SPACING_Y = VIZ_SCALE_MULTIPLIER * 2.5;

interface LayersVizProps {
  nnCtx: NNContext;
  setSelectedNeuron: (selectedNeuron: { layerIx: number; neuronIx: number } | null) => void;
}

interface LayersVizState {
  cursor: string;
}

class LayersViz extends React.Component<LayersVizProps, LayersVizState> {
  private ctx: CanvasRenderingContext2D | null = null;
  private isRendering = false;
  private coord = new Float32Array([0.5, 0.5]);
  private intervalHandle: number | null = null;
  private _selectedNeuron: { layerIx: number; neuronIx: number } | null = null;

  private get selectedNeuron() {
    return this._selectedNeuron;
  }

  private set selectedNeuron(value: { layerIx: number; neuronIx: number } | null) {
    this._selectedNeuron = value;
    this.props.setSelectedNeuron(value);
  }

  private layerSizes: { input: number; hidden: number[]; output: number } = {
    input: 0,
    hidden: [],
    output: 0,
  };

  constructor(props: LayersVizProps) {
    super(props);
    this.state = { cursor: 'default' };
  }

  private drawLayer = (layerColors: Uint8Array, layerIx: number) => {
    const colorData: Uint8ClampedArray = new Uint8ClampedArray(layerColors.buffer);
    const pixelCount = colorData.length / 4;
    const width = pixelCount / VIZ_SCALE_MULTIPLIER;
    const height = pixelCount / width;
    const imageData = new ImageData(colorData, width, height);
    imageData.data.set(colorData, 0);
    this.ctx!.putImageData(imageData, 0, PADDING_TOP + layerIx * LAYER_SPACING_Y);
  };

  private getNeuronAtPosition = (
    x: number,
    y: number
  ): { layerIx: number; neuronIx: number } | null => {
    if (y < PADDING_TOP) {
      return null;
    }
    y -= PADDING_TOP;

    const layerIx = Math.floor(y / LAYER_SPACING_Y);
    if (layerIx >= 1 + this.layerSizes.hidden.length + 1) {
      return null;
    }

    const offset = y - layerIx * LAYER_SPACING_Y;
    if (offset > VIZ_SCALE_MULTIPLIER) {
      return null;
    }

    // We know we are in a valid y for a layer that exists, now we just have to determine if we are in a valid x
    let neuronCountForLayer = 0;
    if (layerIx === 0) {
      neuronCountForLayer = this.layerSizes.input;
    } else if (layerIx < 1 + this.layerSizes.hidden.length) {
      neuronCountForLayer = this.layerSizes.hidden[layerIx - 1];
    } else {
      neuronCountForLayer = this.layerSizes.output;
    }
    const neuronIx = Math.floor(x / VIZ_SCALE_MULTIPLIER);
    if (neuronIx >= neuronCountForLayer) {
      return null;
    }

    return { layerIx, neuronIx };
  };

  private handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const neuronUnderPointer = this.getNeuronAtPosition(
      e.nativeEvent.offsetX,
      e.nativeEvent.offsetY
    );
    if (
      !neuronUnderPointer ||
      (this.selectedNeuron &&
        this.selectedNeuron.layerIx === neuronUnderPointer.layerIx &&
        this.selectedNeuron.neuronIx === neuronUnderPointer.neuronIx)
    ) {
      if (this.selectedNeuron) {
        this.selectedNeuron = null;
        this.forceRender();
      }
      return;
    }

    this.selectedNeuron = neuronUnderPointer;
    this.forceRender();
  };

  private handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const neuronUnderPointer = this.getNeuronAtPosition(
      e.nativeEvent.offsetX,
      e.nativeEvent.offsetY
    );
    const cursor = neuronUnderPointer ? 'pointer' : 'default';
    if (cursor !== this.state.cursor) {
      this.setState({ cursor });
    }
  };

  private maybeRenderSelectedNeuron = () => {
    if (!this.selectedNeuron) {
      return;
    }

    const ctx = this.ctx!;
    const offsetX = this.selectedNeuron.neuronIx * VIZ_SCALE_MULTIPLIER;
    const offsetY = PADDING_TOP + this.selectedNeuron.layerIx * LAYER_SPACING_Y;
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(offsetX, offsetY, VIZ_SCALE_MULTIPLIER, VIZ_SCALE_MULTIPLIER);
  };

  private maybeRender = async (force = false) => {
    if ((!this.props.nnCtx.isRunning && !force) || this.isRendering) {
      return;
    }

    const ctx = this.ctx;
    if (!ctx) {
      return;
    }

    this.isRendering = true;
    const vizData = await this.props.nnCtx.getVizData(new Float32Array(this.coord));
    this.isRendering = false;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (!vizData) {
      return;
    }

    this.layerSizes.input =
      vizData.inputLayerColors.length / 4 / VIZ_SCALE_MULTIPLIER / VIZ_SCALE_MULTIPLIER;
    this.layerSizes.output =
      vizData.outputLayerColors.length / 4 / VIZ_SCALE_MULTIPLIER / VIZ_SCALE_MULTIPLIER;
    this.layerSizes.hidden = vizData.hiddenLayerColors.map(
      layer => layer.length / 4 / VIZ_SCALE_MULTIPLIER / VIZ_SCALE_MULTIPLIER
    );

    this.drawLayer(vizData.inputLayerColors, 0);

    for (let hiddenLayerIx = 0; hiddenLayerIx < vizData.hiddenLayerColors.length; hiddenLayerIx++) {
      this.drawLayer(vizData.hiddenLayerColors[hiddenLayerIx], hiddenLayerIx + 1);
    }

    this.drawLayer(vizData.outputLayerColors, 1 + vizData.hiddenLayerColors.length);

    this.maybeRenderSelectedNeuron();
  };

  forceRender = () => this.maybeRender(true);

  public componentDidMount = () => {
    this.intervalHandle = setInterval(this.maybeRender, 200);
  };

  public componentWillUnmount = () => {
    if (this.intervalHandle !== null) {
      clearInterval(this.intervalHandle);
    }
  };

  public render = () => (
    <div className='layers-viz'>
      <canvas
        width={VIZ_SCALE_MULTIPLIER * 128}
        height={200}
        style={{ cursor: this.state.cursor }}
        ref={canvas => {
          if (!canvas) {
            this.ctx = null;
            return;
          }

          this.ctx = canvas.getContext('2d')!;
        }}
        onMouseDown={this.handleCanvasMouseDown}
        onMouseMove={this.handleCanvasMouseMove}
      />
      <CoordPicker coord={this.coord} onChange={this.forceRender} />
    </div>
  );
}

export default LayersViz;
