import React from 'react';

import { NNContext } from 'src/NNContext';
import { AppStyles } from 'src/sizing';
import { deregisterVizUpdateCB, registerVizUpdateCB } from '../vizControls';
import ColorsScaleLegend from './ColorsScaleLegend';
import CoordPicker from './CoordPicker';

import './LayersViz.css';
import NeuronResponsePlot from './NeuronResponsePlot';

const dpr = Math.floor(window.devicePixelRatio);

const PADDING_TOP = 20 * dpr;
const VIZ_SCALE_MULTIPLIER = 24 * dpr;
const LAYER_SPACING_Y = VIZ_SCALE_MULTIPLIER * 2.5;

export interface LayerSizes {
  input: number;
  hidden: number[];
  output: number;
}

interface LayersVizProps {
  nnCtx: NNContext;
  appStyles: AppStyles;
}

interface LayersVizState {
  cursor: string;
  selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null;
  hiddenLayerCount: number;
  maxHiddenLayerSize: number;
}

class LayersViz extends React.Component<LayersVizProps, LayersVizState> {
  private ctx: CanvasRenderingContext2D | null = null;
  private isRendering = false;
  private coord = new Float32Array([0.5, 0.5]);
  private intervalHandle: number | null = null;

  private layerSizes: { input: number; hidden: number[]; output: number } = {
    input: 0,
    hidden: [],
    output: 0,
  };

  constructor(props: LayersVizProps) {
    super(props);
    this.state = {
      cursor: 'default',
      selectedNeuron: { layerIx: 'init_output', neuronIx: 0 },
      hiddenLayerCount: 2,
      maxHiddenLayerSize: 64,
    };
  }

  private drawLayer = (layerColors: Uint8Array, layerIx: number) => {
    const colorData: Uint8ClampedArray = new Uint8ClampedArray(layerColors.buffer);
    const pixelCount = colorData.length / 4;
    const width = pixelCount / VIZ_SCALE_MULTIPLIER;
    const height = pixelCount / width;
    const imageData = new ImageData(colorData, width, height);
    imageData.data.set(colorData, 0);
    this.ctx?.putImageData(imageData, 0, PADDING_TOP + layerIx * LAYER_SPACING_Y);
  };

  private renderOverlay() {
    if (!this.ctx) {
      return;
    }

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);
    this.ctx.font = `${14}px "PT Sans"`;
    this.ctx.fillStyle = '#ccc';
    this.ctx.strokeStyle = '#ccc';
    this.ctx.textBaseline = 'bottom';
    this.ctx.fillText('Inputs', 1, PADDING_TOP / dpr - 2);

    for (
      let hiddenLayerIx = 0;
      hiddenLayerIx < this.props.nnCtx.definition.hiddenLayers.length;
      hiddenLayerIx++
    ) {
      const y = PADDING_TOP / dpr + ((hiddenLayerIx + 1) * LAYER_SPACING_Y) / dpr - 2;
      this.ctx.fillText(`Hidden Layer ${hiddenLayerIx + 1} Outputs (post-activation)`, 1, y);
    }
    this.ctx.fillText(
      'Network Output',
      1,
      PADDING_TOP / dpr +
        ((this.props.nnCtx.definition.hiddenLayers.length + 1) * LAYER_SPACING_Y) / dpr -
        2
    );
  }

  private getNeuronAtPosition = (
    x: number,
    y: number
  ): { layerIx: number; neuronIx: number } | null => {
    x = x * dpr;
    y = y * dpr;
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
      (this.state.selectedNeuron &&
        this.state.selectedNeuron.layerIx === neuronUnderPointer.layerIx &&
        this.state.selectedNeuron.neuronIx === neuronUnderPointer.neuronIx)
    ) {
      if (this.state.selectedNeuron) {
        this.setState({ selectedNeuron: null }, this.forceRender);
      }
      return;
    }

    this.setState({ selectedNeuron: neuronUnderPointer }, this.forceRender);
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

  private maybeRenderSelectedNeuron = (hiddenLayerCount: number) => {
    const { selectedNeuron } = this.state;
    if (!selectedNeuron) {
      return;
    }

    const ctx = this.ctx;
    if (!ctx) {
      return;
    }

    if (
      typeof selectedNeuron.layerIx === 'number' &&
      selectedNeuron.layerIx > this.layerSizes.hidden.length + 1
    ) {
      this.setState({ selectedNeuron: null });
      return;
    }

    let neuronCountForLayer;
    if (selectedNeuron.layerIx === 0) {
      neuronCountForLayer = this.layerSizes.input;
    } else if (selectedNeuron.layerIx === this.layerSizes.hidden.length + 1) {
      neuronCountForLayer = this.layerSizes.output;
    } else {
      neuronCountForLayer = this.layerSizes.hidden[selectedNeuron.neuronIx - 1];
    }
    if (selectedNeuron.neuronIx >= neuronCountForLayer) {
      this.setState({ selectedNeuron: null });
      return;
    }

    const offsetX = selectedNeuron.neuronIx * (VIZ_SCALE_MULTIPLIER / dpr);
    const offsetY =
      PADDING_TOP / dpr +
      (selectedNeuron.layerIx === 'init_output' ? hiddenLayerCount + 1 : selectedNeuron.layerIx) *
        (LAYER_SPACING_Y / dpr);
    ctx.strokeStyle = '#00ee00';
    ctx.lineWidth = 3;
    ctx.strokeRect(offsetX, offsetY, VIZ_SCALE_MULTIPLIER / dpr, VIZ_SCALE_MULTIPLIER / dpr);
  };

  private renderInputWeightLines = (selectedNeuronInputWeights: Uint8Array) => {
    const selectedNeuron = this.state.selectedNeuron;
    if (!selectedNeuron || selectedNeuron.layerIx === 'init_output') {
      return;
    }

    const ctx = this.ctx!;
    if (selectedNeuronInputWeights.length % 4 !== 0) {
      throw new Error('Unexpected weights colors length');
    }

    const startX =
      selectedNeuron.neuronIx * (VIZ_SCALE_MULTIPLIER / dpr) + VIZ_SCALE_MULTIPLIER / dpr / 2;
    const startY =
      PADDING_TOP / dpr +
      (selectedNeuron.layerIx * (LAYER_SPACING_Y / dpr) + VIZ_SCALE_MULTIPLIER / dpr);
    const endY = startY + LAYER_SPACING_Y / dpr - VIZ_SCALE_MULTIPLIER / dpr;

    ctx.lineWidth = 1.3;
    for (let i = 0; i < selectedNeuronInputWeights.length / 4; i += 1) {
      const [r, g, b] = [
        selectedNeuronInputWeights[i * 4],
        selectedNeuronInputWeights[i * 4 + 1],
        selectedNeuronInputWeights[i * 4 + 2],
      ];
      ctx.beginPath();
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.moveTo(startX, startY);
      ctx.lineTo(i * (VIZ_SCALE_MULTIPLIER / dpr) + VIZ_SCALE_MULTIPLIER / dpr / 2, endY);
      ctx.stroke();
    }
  };

  private maybeRender = async (force = false) => {
    const maxHiddenLayerSize = Math.max(
      ...this.props.nnCtx.definition.hiddenLayers.map(l => l.neuronCount),
      2
    );
    if (
      this.state.hiddenLayerCount !== this.props.nnCtx.definition.hiddenLayers.length ||
      this.state.maxHiddenLayerSize !== maxHiddenLayerSize
    ) {
      this.setState({
        hiddenLayerCount: this.props.nnCtx.definition.hiddenLayers.length,
        maxHiddenLayerSize,
      });
    }

    const ctx = this.ctx;
    if (!ctx) {
      return;
    }

    if ((!this.props.nnCtx.isRunning && !force) || this.isRendering) {
      return;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.isRendering = true;
    const vizData = await this.props.nnCtx.getVizData(
      this.coord,
      this.state.selectedNeuron,
      VIZ_SCALE_MULTIPLIER
    );
    this.isRendering = false;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (!vizData) {
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.font = '18px "PT Sans"';
      ctx.fillStyle = '#ccc';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('Train some examples to see this visualization', 10, 30);
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

    this.maybeRenderSelectedNeuron(vizData.hiddenLayerColors.length);
    if (vizData.selectedNeuronInputWeights) {
      this.renderInputWeightLines(vizData.selectedNeuronInputWeights);
    }

    this.renderOverlay();
  };

  forceRender = () => this.maybeRender(true);

  public componentDidMount = () => {
    setTimeout(() => registerVizUpdateCB(this.forceRender));
    this.intervalHandle = setInterval(this.maybeRender, 200);
  };

  public componentWillUnmount = () => {
    if (this.intervalHandle !== null) {
      clearInterval(this.intervalHandle);
    }
    deregisterVizUpdateCB(this.forceRender);
  };

  public render = () => (
    <div className='layers-viz' style={this.props.appStyles.layersViz}>
      <div className='layers-viz-canvas-wrapper'>
        <canvas
          width={Math.max(VIZ_SCALE_MULTIPLIER * this.state.maxHiddenLayerSize, 400 * dpr)}
          height={(this.state.hiddenLayerCount + 2) * LAYER_SPACING_Y}
          style={{
            cursor: this.state.cursor,
            width: Math.max((VIZ_SCALE_MULTIPLIER / dpr) * this.state.maxHiddenLayerSize, 400),
            height: ((this.state.hiddenLayerCount + 2) * LAYER_SPACING_Y) / dpr,
          }}
          ref={canvas => {
            if (!canvas) {
              this.ctx = null;
              return;
            }

            this.ctx = canvas.getContext('2d')!;
            this.forceRender();
          }}
          onMouseDown={this.handleCanvasMouseDown}
          onMouseMove={this.handleCanvasMouseMove}
        />
      </div>

      <div className='bottom-vizs'>
        <div className='header'>
          <h3>Neuron Response Plot</h3>
        </div>
        <CoordPicker
          coord={this.coord}
          onChange={this.forceRender}
          style={this.props.appStyles.bottomVizs.coordPicker}
        />
        <NeuronResponsePlot
          selectedNeuron={this.state.selectedNeuron}
          nnCtx={this.props.nnCtx}
          style={this.props.appStyles.bottomVizs.neuronResponsePlot}
        />
        <ColorsScaleLegend nnCtx={this.props.nnCtx} />
      </div>
    </div>
  );
}

export default LayersViz;
