import { useWindowSize } from 'ameo-utils';
import React, { useLayoutEffect, useMemo, useState } from 'react';

import './App.css';
import NetworkConfigurator from './NetworkConfigurator';
import { NNContext } from './NNContext';
import { initPresetByID } from './presets';
import RuntimeControls from './RuntimeControls';
import { AppStyles, getAppStyles, MOBILE_CUTOFF_PX } from './sizing';

const getIsConstrainedLayout = (windowWidth: number) =>
  windowWidth < MOBILE_CUTOFF_PX || window.location.search.includes('constrainedLayout');

const setConstrainedLayout = (isConstrainedLayout: boolean) => {
  const htmlElem = document.getElementsByTagName('html')[0] as HTMLHtmlElement;
  if (isConstrainedLayout) {
    htmlElem.classList.add('constrained-layout');
  } else {
    htmlElem.classList.remove('constrained-layout');
  }
};

interface ContentProps {
  nnCtx: NNContext;
  appStyles: AppStyles;
  isConstrainedLayout: boolean;
}

const Content: React.FC<ContentProps> = ({ nnCtx, appStyles, isConstrainedLayout }) => {
  const [expanded, setExpanded] = useState<'configurator' | 'runtime'>('configurator');

  return (
    <div className='content' style={appStyles.content}>
      <NetworkConfigurator
        nnCtx={nnCtx}
        isConstrainedLayout={isConstrainedLayout}
        isExpanded={expanded === 'configurator'}
        setIsExpanded={isExpanded => setExpanded(isExpanded ? 'configurator' : 'runtime')}
        style={appStyles.networkConfigurator}
      />
      <RuntimeControls
        nnCtx={nnCtx}
        isConstrainedLayout={isConstrainedLayout}
        isExpanded={expanded === 'runtime'}
        setExpanded={isExpanded => setExpanded(isExpanded ? 'runtime' : 'configurator')}
        appStyles={appStyles}
      />
    </div>
  );
};

if (window.location.search.includes('preset')) {
  const searchParams = new URLSearchParams(window.location.search);
  const preset = searchParams.get('preset');
  if (preset) {
    initPresetByID(preset);
  }
}

if (getIsConstrainedLayout(window.innerWidth)) {
  const htmlElement = document.getElementsByTagName('html')[0];
  if (((window as any).defaultViz ?? 'response') === 'response') {
    htmlElement.style.overflowY = 'hidden';
  } else {
    htmlElement.style.overflowY = 'auto';
  }
}

const nnCtx = new NNContext(window.innerWidth < 850);
(window as any).ctx = nnCtx;

(window as any).serialize = () => console.log(JSON.stringify(nnCtx.definition));

const App: React.FC = () => {
  const windowSize = useWindowSize();
  const appStyles = useMemo(
    () => getAppStyles(windowSize.width, windowSize.height),
    [windowSize.height, windowSize.width]
  );
  const isConstrainedLayout = getIsConstrainedLayout(windowSize.width);
  useLayoutEffect(() => setConstrainedLayout(isConstrainedLayout), [isConstrainedLayout]);

  return (
    <div className='app'>
      <Content nnCtx={nnCtx} appStyles={appStyles} isConstrainedLayout={isConstrainedLayout} />
    </div>
  );
};

export default App;
