import React from 'react';

export const MOBILE_CUTOFF_PX = 850;

export const RESPONSE_VIZ_RESOLUTION = (window.innerWidth || 0) >= 800 ? 75 : 50;

export interface BottomVizsStyles {
  neuronResponsePlot: React.CSSProperties;
  coordPicker: React.CSSProperties;
}

export interface AppStyles {
  content: React.CSSProperties;
  networkConfigurator: React.CSSProperties;
  runtimeControls: React.CSSProperties;
  responseViz: React.CSSProperties;
  layersViz: React.CSSProperties;
  showSideBySizeResponseViz: boolean;
  bottomVizs: BottomVizsStyles;
}

const buildBottomVizsStyles = (layersVizWidth: number): BottomVizsStyles => ({
  neuronResponsePlot: {
    width: 250,
    height: 250,
    position: 'absolute',
    marginLeft: (layersVizWidth - 250) / 2,
    marginTop: 0,
  },
  coordPicker: {
    position: 'absolute',
    left: (layersVizWidth - 250) / 2,
    top: 20,
  },
});

const getMobileStyles = (windowWidth: number): AppStyles => {
  const content: React.CSSProperties = {
    flexDirection: 'column',
  };

  const networkConfigurator: React.CSSProperties = {};

  const runtimeControls: React.CSSProperties = {};

  const responseViz: React.CSSProperties = { height: 'max(270px, calc(100vh - 360px))' };

  const layersViz: React.CSSProperties = {};

  return {
    content,
    networkConfigurator,
    runtimeControls,
    responseViz,
    layersViz,
    showSideBySizeResponseViz: false,
    bottomVizs: buildBottomVizsStyles(windowWidth),
  };
};

const getDesktopStyles = (windowWidth: number): AppStyles => {
  const contentPadding = 19;
  const networkConfiguratorWidth = 400;
  const runtimeControlsPaddingLeft = 20;

  const content: React.CSSProperties = {
    padding: contentPadding,
    flexDirection: 'row',
  };

  const networkConfigurator: React.CSSProperties = {
    width: networkConfiguratorWidth,
  };

  const runtimeControlsWidth = windowWidth - contentPadding * 2 - networkConfiguratorWidth;

  const runtimeControls: React.CSSProperties = {
    boxSizing: 'border-box',
    width: runtimeControlsWidth,
    paddingLeft: runtimeControlsPaddingLeft,
  };

  const showSideBySizeResponseViz = runtimeControlsWidth >= 800;

  const responseViz: React.CSSProperties = {
    width: showSideBySizeResponseViz ? Math.floor(runtimeControlsWidth / 2) : runtimeControlsWidth,
    height: 'max(300px, calc(100vh - 335px))',
  };

  const layersVizWidth = showSideBySizeResponseViz
    ? Math.floor(runtimeControlsWidth / 2)
    : runtimeControlsWidth;
  const layersViz: React.CSSProperties = {
    width: layersVizWidth,
    height: 'max(300px, calc(100vh - 335px))',
  };

  return {
    content,
    networkConfigurator,
    runtimeControls,
    responseViz,
    layersViz,
    showSideBySizeResponseViz,
    bottomVizs: buildBottomVizsStyles(layersVizWidth),
  };
};

export const getAppStyles = (width: number, height: number): AppStyles => {
  const isMobile = width < MOBILE_CUTOFF_PX;

  return isMobile ? getMobileStyles(width) : getDesktopStyles(width);
};
