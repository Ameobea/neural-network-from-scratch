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

    // const inputDims = this.definition.inputLayer.neuronCount;
    // const outputDims = this.definition.outputLayer.neuronCount;
    // const iterations = examples.length / inputDims;
    // const costs = new Float32Array(iterations);

    // for (let i = 0; i < iterations; i++) {
    //   const example = examples.subarray(i * inputDims, i * inputDims + inputDims);
    //   const expected = expecteds.subarray(i * outputDims, i * outputDims + outputDims);
    //   const cost = this.engine.train(this.ctxPtr!, example, expected, learningRate);
    //   costs[i] = cost;
    // }

    // return costs;
  }
}

const init = async () => {
  const engine = await engineModule;
  Comlink.expose(new NNWorkerCtx(engine));
};

init();
