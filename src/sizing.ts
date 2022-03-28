import React from 'react';

export const MOBILE_CUTOFF_PX = 850;

export interface AppStyles {
  content: React.CSSProperties;
  networkConfigurator: React.CSSProperties;
  runtimeControls: React.CSSProperties;
  responseViz: React.CSSProperties;
  layersViz: React.CSSProperties;
  showSideBySizeResponseViz: boolean;
}

const getMobileStyles = (): AppStyles => {
  const content: React.CSSProperties = {
    flexDirection: 'column',
  };

  const networkConfigurator: React.CSSProperties = {};

  const runtimeControls: React.CSSProperties = {};

  const responseViz: React.CSSProperties = {};

  const layersViz: React.CSSProperties = {};

  return {
    content,
    networkConfigurator,
    runtimeControls,
    responseViz,
    layersViz,
    showSideBySizeResponseViz: false,
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
    height: '100%',
  };

  const layersViz: React.CSSProperties = {
    width: showSideBySizeResponseViz ? Math.floor(runtimeControlsWidth / 2) : runtimeControlsWidth,
    height: '100%',
  };

  return {
    content,
    networkConfigurator,
    runtimeControls,
    responseViz,
    layersViz,
    showSideBySizeResponseViz,
  };
};

export const getAppStyles = (width: number, height: number): AppStyles => {
  const isMobile = width < MOBILE_CUTOFF_PX;

  return isMobile ? getMobileStyles() : getDesktopStyles(width);
};
