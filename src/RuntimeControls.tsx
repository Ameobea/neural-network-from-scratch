import React, { Suspense, useCallback, useMemo, useState, useReducer } from 'react';
import ControlPanel from 'react-control-panel';
import { UnreachableException, useWindowSize } from 'ameo-utils';

import type { NNContext } from './NNContext';
import type { ResponseMatrix } from './Charts/ResponseViz';
import './RuntimeControls.css';
import Loading from './Loading';
import { getSentry } from './sentry';
import ExpandCollapseButton from './Components/ExpandCollapseButton';
import LayersViz from './Charts/LayersViz/LayersViz';
import { updateViz } from './Charts/vizControls';
import { AppStyles, RESPONSE_VIZ_RESOLUTION } from './sizing';

const Charts = import('./Charts');

const LazyResponseViz = React.lazy(() => Charts.then(charts => ({ default: charts.ResponseViz })));
const LazyCostsPlot = React.lazy(() => Charts.then(charts => ({ default: charts.CostsPlot })));

interface OutputData {
  responseMatrix: ResponseMatrix;
  costs: number[];
}

interface VizPickerProps {
  selectedViz: 'response' | 'layers';
  setSelectedViz: (selectedViz: 'response' | 'layers') => void;
}

const VizPicker: React.FC<VizPickerProps> = ({ selectedViz, setSelectedViz }) => (
  <div className='viz-picker'>
    <div
      role='tab'
      data-selected={selectedViz === 'response'}
      className='viz-picker-tab'
      onClick={() => setSelectedViz('response')}
    >
      Network Response Plot
    </div>
    <div
      role='tab'
      data-selected={selectedViz === 'layers'}
      className='viz-picker-tab'
      onClick={() => setSelectedViz('layers')}
    >
      Layer Outputs Viz
    </div>
  </div>
);

interface OutputDataDisplayProps extends OutputData {
  sourceFn: (inputs: Float32Array) => Float32Array;
  isConstrainedLayout: boolean;
  nnCtx: NNContext;
  appStyles: AppStyles;
  expandRuntimeControls: () => void;
}

const OutputDataDisplay: React.FC<OutputDataDisplayProps> = ({
  nnCtx,
  responseMatrix,
  sourceFn,
  costs,
  isConstrainedLayout,
  appStyles,
  expandRuntimeControls,
}) => {
  const [selectedViz, setSelectedViz] = useState<'response' | 'layers'>('response');

  if (appStyles.showSideBySizeResponseViz) {
    return (
      <div className='charts'>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
          <Suspense fallback={<Loading style={{ textAlign: 'center', height: 514 }} />}>
            <LazyResponseViz
              data={responseMatrix}
              sourceFn={sourceFn}
              inputRange={[0, 1]}
              isConstrainedLayout={isConstrainedLayout}
              style={appStyles.responseViz}
            />
          </Suspense>
          <LayersViz nnCtx={nnCtx} appStyles={appStyles} />
        </div>
        <LazyCostsPlot costs={costs} />
      </div>
    );
  }

  return (
    <div className='charts'>
      <VizPicker
        selectedViz={selectedViz}
        setSelectedViz={selectedViz => {
          if (selectedViz === 'layers') {
            expandRuntimeControls();
          }
          setSelectedViz(selectedViz);
        }}
      />
      {selectedViz === 'response' ? (
        <Suspense fallback={<Loading style={{ textAlign: 'center', height: 514 }} />}>
          <LazyResponseViz
            data={responseMatrix}
            sourceFn={sourceFn}
            inputRange={[0, 1]}
            isConstrainedLayout={isConstrainedLayout}
            style={appStyles.responseViz}
          />
        </Suspense>
      ) : (
        <LayersViz nnCtx={nnCtx} appStyles={appStyles} />
      )}
      {selectedViz === 'response' ? <LazyCostsPlot costs={costs} /> : null}
    </div>
  );
};

interface RuntimeControlsProps {
  nnCtx: NNContext;
  isConstrainedLayout: boolean;
  isExpanded: boolean;
  setExpanded: (isExpanded: boolean) => void;
  appStyles: AppStyles;
}

enum SourceFnType {
  Ternary1,
  Multiply,
  Max,
  Min,
  ComplexFancy,
  ComplexFancy2,
  Bowl,
  Random,
  Ridges,
  Xor,
}

const buildSourceFn = (fnType: SourceFnType) => {
  switch (fnType) {
    case SourceFnType.Ternary1:
      return (inputs: Float32Array) =>
        new Float32Array([inputs[0] > 0.5 || inputs[1] > inputs[0] ? 1 : 0]);
    case SourceFnType.Multiply:
      return (inputs: Float32Array) => new Float32Array([inputs[0] * inputs[1]]);
    case SourceFnType.Max:
      return (inputs: Float32Array) => new Float32Array([Math.max(inputs[0], inputs[1])]);
    case SourceFnType.Min:
      return (inputs: Float32Array) => new Float32Array([Math.min(inputs[0], inputs[1])]);
    case SourceFnType.ComplexFancy:
      return (inputs: Float32Array) => {
        const [a, b] = [inputs[0], inputs[1]];

        const val =
          a < 0.1 || b < 0.1 || a > 0.9 || b > 0.9
            ? Math.max(a, b)
            : Math.abs(Math.sin(inputs[0] * 6)) * Math.abs(Math.sin(inputs[1] * 6));
        return new Float32Array([val]);
      };
    case SourceFnType.ComplexFancy2:
      return (inputs: Float32Array) => {
        const [aRaw, bRaw] = [inputs[0], inputs[1]];

        if (aRaw >= 0.4 && aRaw <= 0.6 && bRaw >= 0.4 && bRaw <= 0.6) {
          return new Float32Array([1]);
        }

        const scaler = 4;
        const a = aRaw * scaler - Math.trunc(aRaw * scaler);
        const b = bRaw * scaler - Math.trunc(bRaw * scaler);

        return new Float32Array([Math.sqrt(a * 1 * b * 1)]);
      };
    case SourceFnType.Bowl:
      return (inputs: Float32Array) => {
        const [aRaw, bRaw] = [inputs[0], inputs[1]];
        const [a, b] = [aRaw, bRaw].map(x => x * 2 - 1);
        const val = Math.pow(Math.max(Math.abs(a), Math.abs(b)), 1.5);
        return new Float32Array([val]);
      };
    case SourceFnType.Ridges:
      return (inputs: Float32Array) => {
        const x = inputs[0] * 5;
        const y = inputs[1] * 5;
        const flag = Math.floor(x) % 2 === 1 || Math.floor(y) % 2 === 1;
        return new Float32Array([flag ? 1 : 0]);
      };
    case SourceFnType.Random:
      return (_inputs: Float32Array) => new Float32Array([Math.random()]);
    case SourceFnType.Xor:
      return (inputs: Float32Array) => {
        const a = inputs[0] < 0.5;
        const b = inputs[1] > 0.5;
        return new Float32Array([a !== b ? 1 : 0]);
      };
    default:
      throw new UnreachableException();
  }
};

const buildSettings = (
  nnCtx: NNContext,
  sourceFn: (inputs: Float32Array) => Float32Array,
  setOutputData: (newOutputData: {
    responseMatrix: ResponseMatrix;
    costs: Float32Array | number[] | null;
  }) => void,
  viewportWidth: number,
  onTrain1mmStart: () => void
) => [
  {
    type: 'select',
    label: 'target functions',
    options: {
      'a > 0.5 || b > a ? 1 : 0': SourceFnType.Ternary1,
      'a * b': SourceFnType.Multiply,
      'max(a, b)': SourceFnType.Max,
      'min(a, b)': SourceFnType.Min,
      'fancy sine thing': SourceFnType.ComplexFancy,
      bowl: SourceFnType.Bowl,
      'tiled squareroot thing': SourceFnType.ComplexFancy2,
      xor: SourceFnType.Xor,
      ridges: SourceFnType.Ridges,
      'math.random': SourceFnType.Random,
    },
    initial: SourceFnType.ComplexFancy,
  },
  {
    type: 'button',
    label: viewportWidth < 850 ? 'load config + reset' : 'load config + reset training',
    action: async () => {
      if (nnCtx.isRunning) {
        return;
      }

      getSentry()?.captureMessage('Reset button clicked');
      setOutputData({ responseMatrix: [], costs: null });
      await nnCtx.uninit();
      updateViz();
    },
  },
  {
    type: 'button',
    label: viewportWidth < 850 ? 'train 250k examples' : 'train 1 million examples',
    action: async () => {
      onTrain1mmStart();
      if (nnCtx.isRunning) {
        return;
      }

      if (!(await nnCtx.getIsInitialized())) {
        await nnCtx.init(nnCtx.definition);
      }
      updateViz();

      getSentry()?.captureMessage(
        `Train ${viewportWidth < 850 ? '250k' : '1mm'} examples button clicked`
      );
      nnCtx.isRunning = true;
      const costs = await nnCtx.trainWithSourceFunction(sourceFn, 1_000, [0, 1]);

      const responseMatrix = await nnCtx.computeResponseMatrix(RESPONSE_VIZ_RESOLUTION, [0, 1]);
      setOutputData({ responseMatrix, costs });

      const batchSize = 25_000;
      const iters = viewportWidth < 850 ? 10 : 40;
      const batchCosts: Float32Array[] = [];
      for (let i = 0; i < iters; i++) {
        const costs = await nnCtx.trainWithSourceFunction(sourceFn, batchSize, [0, 1]);
        batchCosts.push(costs);

        if (i % 3 === 0 || i === iters - 1) {
          const responseMatrix = await nnCtx.computeResponseMatrix(RESPONSE_VIZ_RESOLUTION, [0, 1]);
          const allCosts = batchCosts.reduce((acc, costs) => [...acc, ...costs], [] as number[]);
          batchCosts.length = 0;
          setTimeout(() => setOutputData({ responseMatrix, costs: allCosts }));
        }
      }

      nnCtx.isRunning = false;
      updateViz();
    },
  },
  {
    type: 'button',
    label: 'train 1000 examples',
    action: async () => {
      if (nnCtx.isRunning) {
        return;
      }

      if (!(await nnCtx.getIsInitialized())) {
        await nnCtx.init(nnCtx.definition);
        setOutputData({ responseMatrix: [], costs: null });
      }
      updateViz();

      getSentry()?.captureMessage('Train 1k button clicked');
      nnCtx.isRunning = true;
      const costs = await nnCtx.trainWithSourceFunction(sourceFn, 1_000, [0, 1]);

      const responseMatrix = await nnCtx.computeResponseMatrix(RESPONSE_VIZ_RESOLUTION, [0, 1]);
      setOutputData({ responseMatrix, costs });
      nnCtx.isRunning = false;
      updateViz();
    },
  },
  {
    type: 'button',
    label: 'train 1 example',
    action: async () => {
      if (nnCtx.isRunning) {
        return;
      }

      if (!(await nnCtx.getIsInitialized())) {
        await nnCtx.init(nnCtx.definition);
      }
      updateViz();

      getSentry()?.captureMessage('Train 1 example button clicked');
      nnCtx.isRunning = true;
      const costs = await nnCtx.trainWithSourceFunction(sourceFn, 1, [0, 1]);

      const responseMatrix = await nnCtx.computeResponseMatrix(RESPONSE_VIZ_RESOLUTION, [0, 1]);
      setOutputData({ responseMatrix, costs });
      nnCtx.isRunning = false;
      updateViz();
    },
  },
];

type OutputDataAction = {
  responseMatrix: ResponseMatrix;
  costs: Float32Array | null;
};

const outputDataReducer = (state: OutputData, action: OutputDataAction): OutputData => ({
  responseMatrix: action.responseMatrix,
  costs: action.costs ? [...state.costs, ...action.costs] : [],
});

const RuntimeControls: React.FC<RuntimeControlsProps> = ({
  nnCtx,
  isConstrainedLayout,
  isExpanded,
  setExpanded,
  appStyles,
}) => {
  const [outputData, dispatchOutputData] = useReducer(outputDataReducer, {
    responseMatrix: [],
    costs: [],
  } as OutputData);
  // Need to wrap in an object because we can't have raw functions as state due to the ability to pass
  // callbacks into `setState`
  const [{ sourceFn }, setSourceFn] = useState({
    sourceFn: buildSourceFn(SourceFnType.ComplexFancy),
  });
  const setOutputData = useCallback((action: OutputDataAction) => dispatchOutputData(action), []);
  const viewportWidth = useWindowSize().width;
  const settings = useMemo(
    () => buildSettings(nnCtx, sourceFn, setOutputData, viewportWidth, () => setExpanded(true)),
    [nnCtx, setExpanded, setOutputData, sourceFn, viewportWidth]
  );

  if (!isExpanded && isConstrainedLayout) {
    return (
      <div className='runtime-controls' style={appStyles.runtimeControls}>
        <div className='collapsed-runtime-controls' onClick={() => setExpanded(true)}>
          <ExpandCollapseButton
            isExpanded={false}
            setExpanded={setExpanded}
            style={{ position: 'absolute', zIndex: 2, marginTop: 10, marginLeft: 8 }}
          />
          <ControlPanel
            style={{ width: '100%' }}
            settings={settings.filter(
              setting => setting.label.includes('million') || setting.label.includes('250k')
            )}
            onChange={(_key: string, val: any) => setSourceFn({ sourceFn: buildSourceFn(+val) })}
          />
        </div>
        <OutputDataDisplay
          nnCtx={nnCtx}
          sourceFn={sourceFn}
          isConstrainedLayout={isConstrainedLayout}
          appStyles={appStyles}
          expandRuntimeControls={() => setExpanded(true)}
          {...outputData}
        />
      </div>
    );
  }

  return (
    <div className='runtime-controls' style={appStyles.runtimeControls}>
      <ControlPanel
        style={{ width: '100%' }}
        settings={settings}
        onChange={(_key: string, val: any) => setSourceFn({ sourceFn: buildSourceFn(+val) })}
      />
      <OutputDataDisplay
        nnCtx={nnCtx}
        sourceFn={sourceFn}
        isConstrainedLayout={isConstrainedLayout}
        appStyles={appStyles}
        expandRuntimeControls={() => setExpanded(true)}
        {...outputData}
      />
    </div>
  );
};

export default RuntimeControls;
