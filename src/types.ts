export const buildDefaultNetworkDefinition = (): NeuralNetworkDefinition => ({
  inputLayer: { neuronCount: 2 },
  hiddenLayers: [
    {
      neuronCount: 128,
      activationFunctionType: ActivationFunctionType.ReLU,
      initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
      initBiasesFnDefinition: { type: 'constant', val: 0 },
    },
    {
      neuronCount: 64,
      activationFunctionType: ActivationFunctionType.ReLU,
      initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
      initBiasesFnDefinition: { type: 'constant', val: 0 },
    },
    {
      neuronCount: 32,
      activationFunctionType: ActivationFunctionType.ReLU,
      initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
      initBiasesFnDefinition: { type: 'constant', val: 0 },
    },
  ],
  outputLayer: {
    neuronCount: 1,
    activationFunctionType: ActivationFunctionType.Sigmoid,
    costFunctionType: CostFunctionType.MeanSquaredError,
    initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
    learningRate: 0.003,
  },
});

export const buildWeightInitParts = (def: InitWeightsFnDefinition) => {
  switch (def.type) {
    case 'constant':
      return { type: 0, args: [def.val, 0] as const };
    case 'continuousUniformDistribution':
      return { type: 1, args: [def.min, def.max] as const };
  }
};

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
  learningRate: number;
}

export interface NeuralNetworkDefinition {
  inputLayer: InputLayerDefinition;
  hiddenLayers: HiddenLayerDefinition[];
  outputLayer: OutputLayerDefinition;
}

export const buildDefaultHiddenLayerDefinition = (): HiddenLayerDefinition => ({
  neuronCount: 8,
  activationFunctionType: ActivationFunctionType.Sigmoid,
  initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
  initBiasesFnDefinition: { type: 'constant', val: 0 },
});
