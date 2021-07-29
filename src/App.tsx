import React, { useEffect, useState } from 'react';

import './App.css';
import Loading from './Loading';
import NetworkConfigurator from './NetworkConfigurator';
import { buildNNContext, NNContext } from './NNContext';
import RuntimeControls from './RuntimeControls';

const Content: React.FC<{ nnCtx: NNContext }> = ({ nnCtx }) => (
  <div className='content'>
    <NetworkConfigurator nnCtx={nnCtx} />
    <RuntimeControls nnCtx={nnCtx} />
  </div>
);

const App: React.FC = () => {
  const [nnCtx, setNNCtx] = useState<null | NNContext>(null);

  useEffect(() => {
    buildNNContext().then(ctx => {
      setNNCtx(ctx);
      (window as any).ctx = ctx;
    });
  }, []);

  return (
    <div className='app'>
      <h1>Neural Network Visualization</h1>
      {!nnCtx ? <Loading /> : <Content nnCtx={nnCtx} />}
    </div>
  );
};

export default App;
