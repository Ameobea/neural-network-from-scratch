import type { NeuralNetworkDefinition } from './types';

interface Preset {
  definition: NeuralNetworkDefinition;
  defaultViz: 'response' | 'layers';
}

const Presets: {
  [key: string]: string;
} = {
  test: '{"defaultViz":"layers","definition":{"inputLayer":{"neuronCount":2},"hiddenLayers":[{"neuronCount":22,"activationFunctionType":4,"initWeightsFnDefinition":2,"initBiasesFnDefinition":2}],"outputLayer":{"neuronCount":1,"activationFunctionType":1,"costFunctionType":0,"initWeightsFnDefinition":{"type":"continuousUniformDistribution","min":-1,"max":1},"learningRate":0.15}}}',
};

export const initPresetByID = (id: string) => {
  console.log('Loading preset', id);
  const serializedPreset = Presets[id];
  if (!serializedPreset) {
    console.error(`Preset with id ${id} not found`);
    return;
  }

  const { definition, defaultViz }: Preset = JSON.parse(serializedPreset);
  (window as any).defaultDefinition = definition;
  (window as any).defaultViz = defaultViz;

  console.log('Loaded preset', id);
};
