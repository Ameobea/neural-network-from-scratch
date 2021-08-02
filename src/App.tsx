import React from 'react';

import './App.css';
import NetworkConfigurator from './NetworkConfigurator';
import { NNContext } from './NNContext';
import RuntimeControls from './RuntimeControls';

const Content: React.FC<{ nnCtx: NNContext }> = ({ nnCtx }) => (
  <div className='content'>
    <NetworkConfigurator nnCtx={nnCtx} />
    <RuntimeControls nnCtx={nnCtx} />
  </div>
);

const nnCtx = new NNContext();
(window as any).ctx = nnCtx;

const App: React.FC = () => {
  return (
    <div className='app'>
      {/* <h1>Neural Network Visualization</h1> */}
      <Content nnCtx={nnCtx} />
    </div>
  );
};

export default App;
