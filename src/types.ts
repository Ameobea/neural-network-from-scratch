export const buildDefaultNetworkDefinition = (isMobile: boolean): NeuralNetworkDefinition => ({
  inputLayer: { neuronCount: 2 },
  hiddenLayers: [
    {
      neuronCount: isMobile ? 40 : 64,
      activationFunctionType: ActivationFunctionType.LeakyReLU,
      initWeightsFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
      initBiasesFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
    },
    {
      neuronCount: isMobile ? 20 : 32,
      activationFunctionType: ActivationFunctionType.LeakyReLU,
      initWeightsFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
      initBiasesFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
    },
  ],
  outputLayer: {
    neuronCount: 1,
    activationFunctionType: ActivationFunctionType.Sigmoid,
    costFunctionType: CostFunctionType.MeanSquaredError,
    initWeightsFnDefinition: { type: 'continuousUniformDistribution', min: -1, max: 1 },
    learningRate: 0.15,
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
  LeakyReLU = 4,
  GCU = 5,
  Gaussian = 6,
  Swish = 7,
  Ameo = 8,
}

export enum CostFunctionType {
  MeanSquaredError = 0,
}

export type InitWeightsFnDefinition =
  | { type: 'constant'; val: number }
  | { type: 'continuousUniformDistribution'; min: number; max: number };

export enum ValueInitializerType {
  AllZero = 0,
  AllOne = 1,
  RandomNegOneTenthPositiveOneTenth = 2,
  RandomNegOnePositiveOne = 3,
  RandomZeroToPositiveOneTenth = 4,
  RandomZeroToPositiveOne = 5,
}

export const buildValueInitializerFunctionDefinition = (
  initializerType: ValueInitializerType
): InitWeightsFnDefinition => {
  switch (initializerType) {
    case ValueInitializerType.AllZero:
      return { type: 'constant', val: 0 };
    case ValueInitializerType.AllOne:
      return { type: 'constant', val: 1 };
    case ValueInitializerType.RandomNegOneTenthPositiveOneTenth:
      return { type: 'continuousUniformDistribution', min: -0.1, max: 0.1 };
    case ValueInitializerType.RandomNegOnePositiveOne:
      return { type: 'continuousUniformDistribution', min: -1, max: 1 };
    case ValueInitializerType.RandomZeroToPositiveOneTenth:
      return { type: 'continuousUniformDistribution', min: 0, max: 0.1 };
    case ValueInitializerType.RandomZeroToPositiveOne:
      return { type: 'continuousUniformDistribution', min: 0, max: 1 };
  }
};

export interface InputLayerDefinition {
  neuronCount: number;
}

export interface HiddenLayerDefinition {
  neuronCount: number;
  activationFunctionType: ActivationFunctionType;
  initWeightsFnDefinition: ValueInitializerType;
  initBiasesFnDefinition: ValueInitializerType;
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
  initWeightsFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
  initBiasesFnDefinition: ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
});
