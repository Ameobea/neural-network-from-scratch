import { PromiseResolveType, UnreachableException } from 'ameo-utils';
import { randomFloatInRange } from './util';

const engineModule = import('./wasm_interface');

export enum ActivationFunctionType {
  Identity = 0,
  Sigmoid = 1,
  Tanh = 2,
  ReLU = 3,
}

export enum CostFunctionType {
  MeanSquaredError = 0,
}

export type InitWeightsFnDefinition =
  | { type: 'constant'; val: number }
  | { type: 'continuousUniformDistribution'; min: number; max: number };

export interface InputLayerDefinition {
  neuronCount: number;
}

export interface HiddenLayerDefinition {
  neuronCount: number;
  activationFunctionType: ActivationFunctionType;
  initWeightsFnDefinition: InitWeightsFnDefinition;
  initBiasesFnDefinition: InitWeightsFnDefinition;
}

export interface OutputLayerDefinition {
  neuronCount: number;
  activationFunctionType: ActivationFunctionType;
  initWeightsFnDefinition: InitWeightsFnDefinition;
  costFunctionType: CostFunctionType;
}

export interface NeuralNetworkDefinition {
  inputLayer: InputLayerDefinition;
  hiddenLayers: HiddenLayerDefinition[];
  outputLayer: OutputLayerDefinition;
  learningRate: number;
}

export const buildDefaultHiddenLayerDefinition = (): HiddenLayerDefinition => ({
  neuronCount: 8,
  activationFunctionType: ActivationFunctionType.Sigmoid,
  initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
  initBiasesFnDefinition: { type: 'constant', val: 0 },
});

export const buildDefaultNetworkDefinition = (): NeuralNetworkDefinition => ({
  inputLayer: { neuronCount: 2 },
  hiddenLayers: [
    {
      neuronCount: 8,
      activationFunctionType: ActivationFunctionType.Sigmoid,
      initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
      initBiasesFnDefinition: { type: 'constant', val: 0 },
    },
    {
      neuronCount: 8,
      activationFunctionType: ActivationFunctionType.Sigmoid,
      initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
      initBiasesFnDefinition: { type: 'constant', val: 0 },
    },
  ],
  outputLayer: {
    neuronCount: 1,
    activationFunctionType: ActivationFunctionType.Sigmoid,
    costFunctionType: CostFunctionType.MeanSquaredError,
    initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
  },
  learningRate: 0.5,
});

const buildWeightInitParts = (def: InitWeightsFnDefinition) => {
  switch (def.type) {
    case 'constant':
      return { type: 0, args: [def.val, 0] as const };
    case 'continuousUniformDistribution':
      return { type: 1, args: [def.min, def.max] as const };
  }
};

export class NNContext {
  public engine: PromiseResolveType<typeof engineModule>;
  public ctxPtr: number | null = null;
  public definition: NeuralNetworkDefinition = buildDefaultNetworkDefinition();

  public isRunning = false;

  private buildCtx(def: NeuralNetworkDefinition) {
    if (this.isRunning) {
      throw new UnreachableException('Tried to create a new context while network was running');
    }

    if (this.ctxPtr) {
      this.engine.free_nn_ctx(this.ctxPtr);
    }

    def.hiddenLayers.forEach((hiddenLayer, layerIx) => {
      const initWeightsFnParts = buildWeightInitParts(hiddenLayer.initWeightsFnDefinition);
      const initBiasesFnParts = buildWeightInitParts(hiddenLayer.initBiasesFnDefinition);
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
      def.learningRate,
      def.outputLayer.activationFunctionType,
      def.outputLayer.costFunctionType,
      outputLayerWeightInitParts.type,
      outputLayerWeightInitParts.args[0],
      outputLayerWeightInitParts.args[1]
    );
  }

  constructor(engine: PromiseResolveType<typeof engineModule>) {
    this.engine = engine;
  }

  public init(def: NeuralNetworkDefinition) {
    if (this.isRunning) {
      alert('Cannot initialize while already running');
      return;
    }

    this.definition = def;
    this.buildCtx(def);
  }

  public trainWithSourceFunction(
    sourceFn: (inputs: Float32Array) => Float32Array,
    iterations: number,
    inputRange: [number, number]
  ) {
    if (!this.ctxPtr) {
      throw new UnreachableException('Not initialized');
    }

    const inputCount = this.definition.inputLayer.neuronCount;
    for (let i = 0; i < iterations; i++) {
      const inputs = new Float32Array(inputCount)
        .fill(0)
        .map(() => randomFloatInRange(inputRange[0], inputRange[1]));
      const outputs = sourceFn(inputs);
      this.engine.train(this.ctxPtr, inputs, outputs, this.definition.learningRate);
    }
  }

  public predict(inputs: Float32Array) {
    if (!this.ctxPtr) {
      throw new UnreachableException('Not initialized');
    }

    return this.engine.predict(this.ctxPtr, inputs);
  }
}

export const buildNNContext = () => engineModule.then(engine => new NNContext(engine));
