import { UnreachableException } from 'ameo-utils';
import * as Comlink from 'comlink';

import { randomFloatInRange } from './util';
import { buildDefaultNetworkDefinition, NeuralNetworkDefinition } from './types';
import type { NNWorkerCtx } from './nn-worker.worker';
import type { ResponseMatrix } from './Charts/ResponseViz';

const nnWorker = Comlink.wrap<NNWorkerCtx>(
  new Worker(new URL('./nn-worker.worker.ts', import.meta.url))
);
(window as any).nnWorker = nnWorker;

export class NNContext {
  public definition: NeuralNetworkDefinition = buildDefaultNetworkDefinition();

  public isRunning = false;

  public async trainWithSourceFunction(
    sourceFn: (inputs: Float32Array) => Float32Array,
    iterations: number,
    inputRange: [number, number]
  ): Promise<Float32Array> {
    if (!(await nnWorker.getIsInitialized())) {
      throw new UnreachableException('Not initialized');
    }

    const inputDims = this.definition.inputLayer.neuronCount;
    const outputDims = this.definition.outputLayer.neuronCount;
    const examples: Float32Array = new Float32Array(inputDims * iterations);
    const expecteds: Float32Array = new Float32Array(outputDims * iterations);
    const inputCount = this.definition.inputLayer.neuronCount;

    for (let i = 0; i < iterations; i++) {
      for (let inputIx = 0; inputIx < inputCount; inputIx++) {
        examples[inputDims * i + inputIx] = randomFloatInRange(inputRange[0], inputRange[1]);
      }
      const inputs = examples.subarray(inputDims * i, inputDims * (i + 1));
      const expected = sourceFn(inputs);
      examples.set(inputs, inputDims * i);
      expecteds.set(expected, outputDims * i);
    }

    return nnWorker.trainBatch(
      Comlink.transfer(examples, [examples.buffer]),
      Comlink.transfer(expecteds, [expecteds.buffer]),
      this.definition.outputLayer.learningRate
    );
  }

  public async computeResponseMatrix(
    steps: number,
    inputRange: [number, number]
  ): Promise<ResponseMatrix> {
    const matrix = await nnWorker.computeResponseMatrix(steps, inputRange);
    const out: ResponseMatrix = [];
    for (let i = 0; i < steps * steps; i++) {
      out.push([matrix[i * 3], matrix[i * 3 + 1], matrix[i * 3 + 2]]);
    }
    return out;
  }

  public getIsInitialized() {
    return nnWorker.getIsInitialized();
  }

  public init(definition: NeuralNetworkDefinition) {
    if (this.isRunning) {
      alert('Cannot initialize while already running');
      return;
    }

    return nnWorker.init(definition);
  }

  public uninit() {
    if (this.isRunning) {
      alert('Cannot uninitialize while already running');
      return;
    }

    return nnWorker.init(null);
  }

  public getVizData(
    example: Float32Array,
    selectedNeuron: { layerIx: number | 'init_output'; neuronIx: number } | null,
    vizScaleMultiplier: number
  ) {
    return nnWorker.getVizData(example, selectedNeuron, vizScaleMultiplier);
  }

  public getNeuronResponse(layerIx: number, neuronIx: number, size: number) {
    return nnWorker.getNeuronResponse(layerIx, neuronIx, size);
  }
}
