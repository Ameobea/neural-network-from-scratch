import React, { useState } from 'react';

import './App.css';
import NetworkConfigurator from './NetworkConfigurator';
import { NNContext } from './NNContext';
import RuntimeControls from './RuntimeControls';

const isConstrainedLayout = window.location.search.includes('constrainedLayout');
if (isConstrainedLayout) {
  const htmlElem = document.getElementsByTagName('html')[0] as HTMLHtmlElement;
  htmlElem.classList.add('constrained-layout');
}

const Content: React.FC<{ nnCtx: NNContext }> = ({ nnCtx }) => {
  const [expanded, setExpanded] = useState<'configurator' | 'runtime'>('configurator');

  return (
    <div className='content'>
      <NetworkConfigurator
        nnCtx={nnCtx}
        isConstrainedLayout={isConstrainedLayout}
        isExpanded={expanded === 'configurator'}
        setIsExpanded={isExpanded => setExpanded(isExpanded ? 'configurator' : 'runtime')}
      />
      <RuntimeControls
        nnCtx={nnCtx}
        isConstrainedLayout={isConstrainedLayout}
        isExpanded={expanded === 'runtime'}
        setExpanded={isExpanded => setExpanded(isExpanded ? 'runtime' : 'configurator')}
      />
    </div>
  );
};

const nnCtx = new NNContext();
(window as any).ctx = nnCtx;

const App: React.FC = () => {
  return (
    <div className='app'>
      <Content nnCtx={nnCtx} />
    </div>
  );
};

export default App;
