import React, { useCallback, useMemo, useState } from 'react';
import ControlPanel from 'react-control-panel';
import * as R from 'ramda';
import { UnreachableException, useWindowSize } from 'ameo-utils';

import {
  ActivationFunctionType,
  buildDefaultHiddenLayerDefinition,
  buildDefaultNetworkDefinition,
  CostFunctionType,
  HiddenLayerDefinition,
  NeuralNetworkDefinition,
  OutputLayerDefinition,
  ValueInitializerType,
} from './types';
import './NetworkConfigurator.css';
import type { NNContext } from './NNContext';
import { getSentry } from './sentry';
import ExpandCollapseButton from './Components/ExpandCollapseButton';

interface NetworkConfiguratorProps {
  nnCtx: NNContext;
  isConstrainedLayout: boolean;
  isExpanded: boolean;
  setIsExpanded: (isExpanded: boolean) => void;
  style?: React.CSSProperties;
}

const buildValueInitializerOptions = () => ({
  'all 0': ValueInitializerType.AllZero,
  'all 1': ValueInitializerType.AllOne,
  'random [-0.1, 0.1]': ValueInitializerType.RandomNegOneTenthPositiveOneTenth,
  'random [-1, 1]': ValueInitializerType.RandomNegOnePositiveOne,
  'random [0, 0.1]': ValueInitializerType.RandomZeroToPositiveOneTenth,
  'random [0, 1]': ValueInitializerType.RandomZeroToPositiveOne,
});

const buildHiddenLayerSettings = (onDelete: () => void) => [
  { type: 'range', label: 'neuron count', min: 1, max: 128, step: 1 },
  {
    type: 'select',
    label: 'activation function',
    options: {
      sigmoid: ActivationFunctionType.Sigmoid,
      relu: ActivationFunctionType.ReLU,
      'leaky relu': ActivationFunctionType.LeakyReLU,
      tanh: ActivationFunctionType.Tanh,
      identity: ActivationFunctionType.Identity,
      gcu: ActivationFunctionType.GCU,
      gaussian: ActivationFunctionType.Gaussian,
      swish: ActivationFunctionType.Swish,
      ameo: ActivationFunctionType.Ameo,
    },
  },
  {
    type: 'select',
    label: 'weight initializer',
    options: buildValueInitializerOptions(),
  },
  {
    type: 'select',
    label: 'bias initializer',
    options: buildValueInitializerOptions(),
  },
  {
    type: 'button',
    label: 'delete',
    action: () => {
      getSentry()?.captureMessage('Delete hidden layer button clicked');
      onDelete();
    },
  },
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
      'weight initializer': layer.initWeightsFnDefinition,
      'bias initializer': layer.initBiasesFnDefinition,
    }),
    [
      layer.activationFunctionType,
      layer.initBiasesFnDefinition,
      layer.initWeightsFnDefinition,
      layer.neuronCount,
    ]
  );
  const viewportWidth = useWindowSize().width;
  const width = viewportWidth < 850 ? viewportWidth : 400;

  return (
    <ControlPanel
      title={`hidden layer ${layerIx + 1}`}
      style={{ width }}
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
          case 'weight initializer': {
            newDef.initWeightsFnDefinition = +val;
            break;
          }
          case 'bias initializer': {
            newDef.initBiasesFnDefinition = +val;
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
  // { type: 'range', label: 'neuron count', min: 1, max: 5, step: 1 },
  {
    type: 'select',
    label: 'activation function',
    options: {
      sigmoid: ActivationFunctionType.Sigmoid,
      relu: ActivationFunctionType.ReLU,
      'leaky relu': ActivationFunctionType.LeakyReLU,
      tanh: ActivationFunctionType.Tanh,
      identity: ActivationFunctionType.Identity,
      gcu: ActivationFunctionType.GCU,
      gaussian: ActivationFunctionType.Gaussian,
      swish: ActivationFunctionType.Swish,
      ameo: ActivationFunctionType.Ameo,
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
  const viewportWidth = useWindowSize().width;
  const width = viewportWidth < 850 ? viewportWidth : 400;

  return (
    <ControlPanel
      style={{ marginTop: viewportWidth < 850 ? 0 : 20, width }}
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

const NetworkConfigurator: React.FC<NetworkConfiguratorProps> = ({
  nnCtx,
  isConstrainedLayout,
  isExpanded,
  setIsExpanded,
  style,
}) => {
  const [definition, setDefinitionInner] = useState<NeuralNetworkDefinition>(
    buildDefaultNetworkDefinition(window.innerWidth < 850)
  );
  const setDefinition = useCallback(
    newDef => {
      setDefinitionInner(newDef);
      nnCtx.definition = newDef;
    },
    [nnCtx]
  );
  const viewportWidth = useWindowSize().width;
  const width = viewportWidth < 850 ? viewportWidth : 400;

  return (
    <div
      className={`network-configurator${isConstrainedLayout ? ' constrained-layout' : ''}`}
      style={style}
    >
      {isConstrainedLayout ? null : (
        <div
          style={{
            fontFamily: "'Hack', 'Oxygen Mono', 'Input', monospace",
            fontSize: 10,
            textAlign: 'center',
            fontStyle: 'italic',
            marginTop: viewportWidth < 850 ? 8 : 0,
            marginBottom: 8,
            color: '#999',
          }}
        >
          Input layer has 2 dimensions, each with a range of [0, 1].
        </div>
      )}
      {isConstrainedLayout && !isExpanded ? (
        <div className='collapsed-network-configurator' onClick={() => setIsExpanded(true)}>
          <ExpandCollapseButton
            isExpanded={false}
            setExpanded={setIsExpanded}
            style={{ marginTop: 2, marginBottom: -2 }}
          />
          <div>Click to open network config</div>
        </div>
      ) : (
        <>
          {definition.hiddenLayers.map((hiddenLayer, layerIx) => (
            <HiddenLayerConfigurator
              key={layerIx}
              layerIx={layerIx}
              layer={hiddenLayer}
              onChange={newHiddenLayer => {
                setDefinition(
                  R.set(R.lensPath(['hiddenLayers', layerIx]), newHiddenLayer, definition)
                );
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
            style={{ width }}
            settings={[
              {
                type: 'button',
                label: 'add hidden layer',
                action: () => {
                  getSentry()?.captureMessage('Add hidden layer button clicked');
                  setDefinition({
                    ...definition,
                    hiddenLayers: [...definition.hiddenLayers, buildDefaultHiddenLayerDefinition()],
                  });
                },
              },
            ]}
          />
          <OutputLayerConfigurator
            layer={definition.outputLayer}
            onChange={newOutputLayer =>
              setDefinition({ ...definition, outputLayer: newOutputLayer })
            }
          />
        </>
      )}
    </div>
  );
};

export default NetworkConfigurator;
