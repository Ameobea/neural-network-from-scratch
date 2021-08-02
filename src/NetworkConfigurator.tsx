import React, { useCallback, useMemo, useState } from 'react';
import ControlPanel from 'react-control-panel';
import * as R from 'ramda';
import { UnreachableException } from 'ameo-utils';

import {
  ActivationFunctionType,
  buildDefaultHiddenLayerDefinition,
  buildDefaultNetworkDefinition,
  CostFunctionType,
  HiddenLayerDefinition,
  InputLayerDefinition,
  NeuralNetworkDefinition,
  OutputLayerDefinition,
} from './types';
import './NetworkConfigurator.css';
import type { NNContext } from './NNContext';

interface NetworkConfiguratorProps {
  nnCtx: NNContext;
}

interface InputLayerConfiguratorProps {
  layer: InputLayerDefinition;
  onChange: (newLayer: InputLayerDefinition) => void;
}

const INPUT_LAYER_SETTINGS = [
  { type: 'range', label: 'input dimensions', min: 1, max: 3, step: 1 },
];

const InputLayerConfigurator: React.FC<InputLayerConfiguratorProps> = ({ layer, onChange }) => {
  const state = useMemo(() => ({ 'input dimensions': layer.neuronCount }), [layer.neuronCount]);

  return (
    <ControlPanel
      title='input layer'
      style={{ marginBottom: 20, width: 400 }}
      state={state}
      settings={INPUT_LAYER_SETTINGS}
      onChange={(key: string, val: any) => {
        const newDef = { ...layer };
        switch (key) {
          case 'input dimensions': {
            if (typeof val !== 'number') {
              throw new UnreachableException();
            }
            newDef.neuronCount = val;
            break;
          }
          default: {
            console.error('Unhandled key in `InputLayerConfigurator`: ', key);
          }
        }

        onChange(newDef);
      }}
    />
  );
};

const buildHiddenLayerSettings = (onDelete: () => void) => [
  { type: 'range', label: 'neuron count', min: 1, max: 128, step: 1 },
  {
    type: 'select',
    label: 'activation function',
    options: {
      sigmoid: ActivationFunctionType.Sigmoid,
      relu: ActivationFunctionType.ReLU,
      tanh: ActivationFunctionType.Tanh,
      identity: ActivationFunctionType.Identity,
    },
  },
  { type: 'button', label: 'delete', action: onDelete },
];

interface HiddenLayerConfiguratorProps {
  layerIx: number;
  layer: HiddenLayerDefinition;
  onChange: (newLayer: HiddenLayerDefinition) => void;
  onDelete: () => void;
}

const HiddenLayerConfigurator: React.FC<HiddenLayerConfiguratorProps> = ({
  layerIx,
  layer,
  onChange,
  onDelete,
}) => {
  const state = useMemo(
    () => ({
      'neuron count': layer.neuronCount,
      'activation function': layer.activationFunctionType,
    }),
    [layer.activationFunctionType, layer.neuronCount]
  );

  return (
    <ControlPanel
      title={`hidden layer ${layerIx + 1}`}
      style={{ width: 400 }}
      state={state}
      settings={buildHiddenLayerSettings(onDelete)}
      onChange={(key: string, val: any) => {
        const newDef = { ...layer };
        switch (key) {
          case 'neuron count': {
            if (typeof val !== 'number') {
              throw new UnreachableException();
            }
            newDef.neuronCount = val;
            break;
          }
          case 'activation function': {
            newDef.activationFunctionType = +val;
            break;
          }
          default: {
            console.error('Unhandled key in `HiddenLayerConfigurator`: ', key);
          }
        }
        onChange(newDef);
      }}
    />
  );
};

interface OutputLayerConfiguratorProps {
  layer: OutputLayerDefinition;
  onChange: (newLayer: OutputLayerDefinition) => void;
}

const OUTPUT_LAYER_SETTINGS = [
  { type: 'range', label: 'neuron count', min: 1, max: 5, step: 1 },
  {
    type: 'select',
    label: 'activation function',
    options: {
      sigmoid: ActivationFunctionType.Sigmoid,
      relu: ActivationFunctionType.ReLU,
      tanh: ActivationFunctionType.Tanh,
      identity: ActivationFunctionType.Identity,
    },
  },
  {
    type: 'select',
    label: 'cost function',
    options: {
      'mean squared error': CostFunctionType.MeanSquaredError,
    },
  },
  {
    type: 'range',
    label: 'learning rate',
    scale: 'log',
    min: 0.0001,
    max: 1,
    initial: 0.5,
  },
];

const OutputLayerConfigurator: React.FC<OutputLayerConfiguratorProps> = ({ layer, onChange }) => {
  const state = useMemo(
    () => ({ 'neuron count': layer.neuronCount, 'learning rate': layer.learningRate }),
    [layer.learningRate, layer.neuronCount]
  );

  return (
    <ControlPanel
      style={{ marginTop: 20, width: 400 }}
      title='output layer'
      state={state}
      settings={OUTPUT_LAYER_SETTINGS}
      onChange={(key: string, val: any) => {
        const newDef = { ...layer };
        switch (key) {
          case 'neuron count': {
            if (typeof val !== 'number') {
              throw new UnreachableException();
            }
            newDef.neuronCount = val;
            break;
          }
          case 'activation function': {
            newDef.activationFunctionType = +val;
            break;
          }
          case 'cost function': {
            newDef.costFunctionType = +val;
            break;
          }
          case 'learning rate': {
            if (Number.isNaN(+val)) {
              return;
            }
            newDef.learningRate = +val;
            break;
          }
          default: {
            console.error('Unhandled key in `OutputLayerConfigurator`: ', key);
          }
        }
        onChange(newDef);
      }}
    />
  );
};

const NetworkConfigurator: React.FC<NetworkConfiguratorProps> = ({ nnCtx }) => {
  const [definition, setDefinitionInner] = useState<NeuralNetworkDefinition>(
    buildDefaultNetworkDefinition()
  );

  const setDefinition = useCallback(
    newDef => {
      setDefinitionInner(newDef);
      nnCtx.definition = newDef;
    },
    [nnCtx]
  );

  return (
    <div className='network-configurator'>
      <InputLayerConfigurator
        layer={definition.inputLayer}
        onChange={newInputLayer => setDefinition({ ...definition, inputLayer: newInputLayer })}
      />
      {definition.hiddenLayers.map((hiddenLayer, layerIx) => (
        <HiddenLayerConfigurator
          key={layerIx}
          layerIx={layerIx}
          layer={hiddenLayer}
          onChange={newHiddenLayer => {
            setDefinition(R.set(R.lensPath(['hiddenLayers', layerIx]), newHiddenLayer, definition));
          }}
          onDelete={() => {
            if (definition.hiddenLayers.length === 1) {
              return;
            }

            setDefinition({
              ...definition,
              hiddenLayers: R.remove(layerIx, 1, definition.hiddenLayers),
            });
          }}
        />
      ))}
      <ControlPanel
        style={{ width: 400 }}
        settings={[
          {
            type: 'button',
            label: 'add hidden layer',
            action: () =>
              setDefinition({
                ...definition,
                hiddenLayers: [...definition.hiddenLayers, buildDefaultHiddenLayerDefinition()],
              }),
          },
        ]}
      />
      <OutputLayerConfigurator
        layer={definition.outputLayer}
        onChange={newOutputLayer => setDefinition({ ...definition, outputLayer: newOutputLayer })}
      />
    </div>
  );
};

export default NetworkConfigurator;
