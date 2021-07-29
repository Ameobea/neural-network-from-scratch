import React, { useMemo, useState } from 'react';
import ControlPanel from 'react-control-panel';
import { UnreachableException } from 'ameo-utils';

import { NNContext } from './NNContext';
import ResponseViz, { ResponseMatrix } from './ResponseViz';
import './RuntimeControls.css';

interface OutputData {
  responseMatrix: ResponseMatrix;
}

interface OutputDataDisplayProps extends OutputData {
  nnCtx: NNContext;
  sourceFn: (inputs: Float32Array) => Float32Array;
}

const OutputDataDisplay: React.FC<OutputDataDisplayProps> = ({
  nnCtx,
  responseMatrix,
  sourceFn,
}) => {
  return (
    <ResponseViz nnCtx={nnCtx} data={responseMatrix} sourceFn={sourceFn} inputRange={[0, 1]} />
  );
};

const computeResponseMatrix = (
  nnCtx: NNContext,
  steps: number,
  inputRange: [number, number]
): ResponseMatrix => {
  const responseMatrix: ResponseMatrix = [];

  if (
    nnCtx.definition.inputLayer.neuronCount !== 2 ||
    nnCtx.definition.outputLayer.neuronCount !== 1
  ) {
    throw new UnreachableException('Only 2-input, 1-output networks are supported');
  }
  if (!nnCtx.ctxPtr) {
    throw new UnreachableException();
  }

  const range = inputRange[1] - inputRange[0];
  const stepSize = range / steps;
  for (let aStepIx = 0; aStepIx < steps; aStepIx++) {
    const a = aStepIx * stepSize + inputRange[0];
    const example = new Float32Array([a, 0]);
    const outputs = nnCtx.engine.predict_batch(
      nnCtx.ctxPtr,
      example,
      1,
      inputRange[0],
      inputRange[1],
      steps
    );
    outputs.forEach((c, bStepIx) => {
      const b = bStepIx * stepSize + inputRange[0];
      responseMatrix.push([a, b, c]);
    });
  }

  return responseMatrix;
};

interface RuntimeControlsProps {
  nnCtx: NNContext;
}

enum SourceFnType {
  Ternary1,
  Multiply,
  Max,
  Min,
  ComplexFancy,
  Discretize,
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
    default:
      throw new UnreachableException();
  }
};

const buildSettings = (
  nnCtx: NNContext,
  sourceFn: (inputs: Float32Array) => Float32Array,
  setOutputData: (data: OutputData) => void
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
    },
    initial: SourceFnType.ComplexFancy,
  },
  {
    type: 'button',
    label: 'reset',
    action: () => {
      if (nnCtx.isRunning) {
        return;
      }

      nnCtx.ctxPtr = null;
      setOutputData({ responseMatrix: [] });
    },
  },
  {
    type: 'button',
    label: 'train 1 million examples',
    action: () => {
      if (nnCtx.isRunning) {
        return;
      }

      if (!nnCtx.ctxPtr) {
        nnCtx.init(nnCtx.definition);
      }
      nnCtx.isRunning = true;

      nnCtx.trainWithSourceFunction(sourceFn, 1_000, [0, 1]);

      const responseMatrix = computeResponseMatrix(nnCtx, 100, [0, 1]);
      setOutputData({ responseMatrix });

      let i = 0;
      const trainOneIteration = () => {
        nnCtx.trainWithSourceFunction(sourceFn, 50_000, [0, 1]);

        const responseMatrix = computeResponseMatrix(nnCtx, 100, [0, 1]);
        setOutputData({ responseMatrix });

        i += 1;
        if (i == 20) {
          nnCtx.isRunning = false;
          return;
        }

        setTimeout(() => trainOneIteration(), 50);
      };

      trainOneIteration();
    },
  },
  {
    type: 'button',
    label: 'train 1000 examples',
    action: () => {
      if (nnCtx.isRunning) {
        return;
      }

      if (!nnCtx.ctxPtr) {
        nnCtx.init(nnCtx.definition);
      }

      nnCtx.trainWithSourceFunction(sourceFn, 1_000, [0, 1]);

      const responseMatrix = computeResponseMatrix(nnCtx, 100, [0, 1]);
      setOutputData({ responseMatrix });
    },
  },
];

const RuntimeControls: React.FC<RuntimeControlsProps> = ({ nnCtx }) => {
  const [outputData, setOutputData] = useState<OutputData>({ responseMatrix: [] });
  const [{ sourceFn }, setSourceFn] = useState({
    sourceFn: buildSourceFn(SourceFnType.ComplexFancy),
  });
  const settings = useMemo(() => buildSettings(nnCtx, sourceFn, setOutputData), [nnCtx, sourceFn]);

  return (
    <div className='runtime-controls'>
      <ControlPanel
        style={{ width: 800 }}
        settings={settings}
        onChange={(_key: string, val: any) => {
          setSourceFn({ sourceFn: buildSourceFn(+val) });
        }}
      />
      {outputData ? <OutputDataDisplay nnCtx={nnCtx} sourceFn={sourceFn} {...outputData} /> : null}
    </div>
  );
};

export default RuntimeControls;
