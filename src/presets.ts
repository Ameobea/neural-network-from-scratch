import type { NeuralNetworkDefinition } from './types';

interface Preset {
  definition: NeuralNetworkDefinition;
  defaultViz: 'response' | 'layers';
  defaultTargetFunction?: number;
}

const Presets: {
  [key: string]: string;
} = {
  '4-layer-neuron-response':
    '{"defaultTargetFunction":5,"defaultViz":"layers","definition":{"inputLayer":{"neuronCount":2},"hiddenLayers":[{"neuronCount":64,"activationFunctionType":4,"initWeightsFnDefinition":2,"initBiasesFnDefinition":2},{"neuronCount":32,"activationFunctionType":4,"initWeightsFnDefinition":2,"initBiasesFnDefinition":2},{"neuronCount":16,"activationFunctionType":4,"initWeightsFnDefinition":2,"initBiasesFnDefinition":2},{"neuronCount":8,"activationFunctionType":4,"initWeightsFnDefinition":2,"initBiasesFnDefinition":2}],"outputLayer":{"neuronCount":1,"activationFunctionType":1,"costFunctionType":0,"initWeightsFnDefinition":{"type":"continuousUniformDistribution","min":-1,"max":1},"learningRate":0.08}}}',
};

export const initPresetByID = (id: string) => {
  console.log('Loading preset', id);
  const serializedPreset = Presets[id];
  if (!serializedPreset) {
    console.error(`Preset with id ${id} not found`);
    return;
  }

  const { definition, defaultViz, defaultTargetFunction }: Preset = JSON.parse(serializedPreset);
  (window as any).defaultDefinition = definition;
  (window as any).defaultViz = defaultViz;
  (window as any).defaultTargetFunction = defaultTargetFunction;

  console.log('Loaded preset', id);
};
