/**
 * Runs the training + inferrence for the neural network on a worker thread leaving the UI thread
 * free for displaying the output visualization.
 */

import * as Comlink from 'comlink';
import { UnreachableException } from 'ameo-utils';

import {
  buildDefaultNetworkDefinition,
  buildValueInitializerFunctionDefinition,
  buildWeightInitParts,
  NeuralNetworkDefinition,
} from './types';

const engineModule = import('./wasm_interface');

export class NNWorkerCtx {
  private engine: typeof import('./wasm_interface');
  private ctxPtr: number | null = null;
  private hiddenLayerCount = 2;
  private definition: NeuralNetworkDefinition = buildDefaultNetworkDefinition();

  constructor(engine: typeof import('./wasm_interface')) {
    this.engine = engine;
  }

  private buildCtx(def: NeuralNetworkDefinition) {
    if (this.ctxPtr) {
      this.engine.free_nn_ctx(this.ctxPtr);
    }

    def.hiddenLayers.forEach((hiddenLayer, layerIx) => {
      const initWeightsFnParts = buildWeightInitParts(
        buildValueInitializerFunctionDefinition(hiddenLayer.initWeightsFnDefinition)
      );
      const initBiasesFnParts = buildWeightInitParts(
        buildValueInitializerFunctionDefinition(hiddenLayer.initBiasesFnDefinition)
      );
      this.engine.store_hidden_layer_definition(
        layerIx,
        hiddenLayer.activationFunctionType,
        hiddenLayer.neuronCount,
        initWeightsFnParts.type,
        initWeightsFnParts.args[0],
        initWeightsFnParts.args[1],
        initBiasesFnParts.type,
        initBiasesFnParts.args[0],
        initBiasesFnParts.args[1]
      );
    });

    const outputLayerWeightInitParts = buildWeightInitParts(
      def.outputLayer.initWeightsFnDefinition
    );
    this.hiddenLayerCount = def.hiddenLayers.length;
    this.ctxPtr = this.engine.create_nn_ctx(
      def.inputLayer.neuronCount,
      def.outputLayer.neuronCount,
      def.hiddenLayers.length,
      def.outputLayer.learningRate,
      def.outputLayer.activationFunctionType,
      def.outputLayer.costFunctionType,
      outputLayerWeightInitParts.type,
      outputLayerWeightInitParts.args[0],
      outputLayerWeightInitParts.args[1]
    );
  }

  public init(def: NeuralNetworkDefinition | null) {
    if (def) {
      this.definition = def;
      this.buildCtx(def);
    } else {
      if (this.ctxPtr) {
        this.engine.free_nn_ctx(this.ctxPtr);
      }
      this.ctxPtr = null;
    }
  }

  public predict(inputs: Float32Array) {
    if (!this.ctxPtr) {
      throw new UnreachableException('Not initialized');
    }

    return this.engine.predict(this.ctxPtr, inputs);
  }

  public computeResponseMatrix(steps: number, inputRange: [number, number]): Float32Array {
    const responseMatrix = new Float32Array(
      steps *
        steps *
        (this.definition.inputLayer.neuronCount + this.definition.outputLayer.neuronCount)
    );

    if (
      this.definition.inputLayer.neuronCount !== 2 ||
      this.definition.outputLayer.neuronCount !== 1
    ) {
      throw new UnreachableException('Only 2-input, 1-output networks are supported');
    }
    if (!this.ctxPtr) {
      throw new UnreachableException('Not initialized');
    }

    const range = inputRange[1] - inputRange[0];
    const stepSize = range / steps;
    let i = 0;
    for (let aStepIx = 0; aStepIx < steps; aStepIx++) {
      const a = aStepIx * stepSize + inputRange[0];
      const example = new Float32Array([a, 0]);
      const outputs = this.engine.predict_batch(
        this.ctxPtr,
        example,
        1,
        inputRange[0],
        inputRange[1],
        steps
      );
      outputs.forEach((c, bStepIx) => {
        const b = bStepIx * stepSize + inputRange[0];
        responseMatrix[i] = a;
        responseMatrix[i + 1] = b;
        responseMatrix[i + 2] = c;
        i += 3;
      });
    }

    return Comlink.transfer(responseMatrix, [responseMatrix.buffer]);
  }

  public getIsInitialized() {
    return !!this.ctxPtr;
  }

  public trainBatch(
    examples: Float32Array,
    expecteds: Float32Array,
    learningRate: number
  ): Float32Array {
    if (!this.ctxPtr) {
      throw new UnreachableException('Not initialized');
    }

    return this.engine.train_many_examples(this.ctxPtr, examples, expecteds, learningRate);
  }

  public getVizData(
    example: Float32Array,
    selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null
  ) {
    if (!this.ctxPtr) {
      return null;
    }

    this.engine.update_viz(this.ctxPtr, example);

    const hiddenLayerColors = [];
    for (let i = 0; i < this.hiddenLayerCount; i++) {
      hiddenLayerColors.push(this.engine.get_viz_hidden_layer_colors(this.ctxPtr, i));
    }
    const inputLayerColors = this.engine.get_viz_input_layer_colors(this.ctxPtr);
    const outputLayerColors = this.engine.get_viz_output_layer_colors(this.ctxPtr);
    const selectedNeuronInputWeights = selectedNeuron
      ? this.engine.get_input_weights_for_next_layer(
          this.ctxPtr,
          selectedNeuron.layerIx === 'init_output' ? -1 : selectedNeuron.layerIx,
          selectedNeuron.neuronIx
        )
      : null;

    return Comlink.transfer(
      {
        hiddenLayerColors,
        inputLayerColors,
        outputLayerColors,
        selectedNeuronInputWeights: selectedNeuronInputWeights?.length
          ? selectedNeuronInputWeights
          : null,
      },
      [...hiddenLayerColors.map(c => c.buffer), outputLayerColors.buffer, inputLayerColors.buffer]
    );
  }

  public getNeuronResponse(layerIx: number, neuronIx: number, size: number) {
    if (!this.ctxPtr) {
      return null;
    }

    const response = this.engine.build_neuron_response_viz(this.ctxPtr, layerIx, neuronIx, size);
    if (response.length === 0) {
      return null;
    }
    return Comlink.transfer(response, [response.buffer]);
  }
}

const init = async () => {
  const engine = await engineModule;
  Comlink.expose(new NNWorkerCtx(engine));
};

init();
