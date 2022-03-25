import EChartsReactCore from 'echarts-for-react/lib/core';
import React from 'react';
import * as echarts from 'echarts/core';
import 'echarts-gl';

export type ResponseMatrix = [number, number, number][];

interface ResponseVizProps {
  data: ResponseMatrix;
  inputRange: [number, number];
  sourceFn: (inputs: Float32Array) => Float32Array;
  isConstrainedLayout: boolean;
}

const style = {
  height: 'max(300px, calc(100vh - 335px))',
};

const ResponseViz: React.FC<ResponseVizProps> = ({
  data,
  inputRange,
  sourceFn,
  isConstrainedLayout,
}) => {
  const chartConfig = {
    backgroundColor: '#101010',
    visualMap: {
      show: false,
      dimension: 2,
      min: 0,
      max: 1,
      seriesIndex: 0,
      inRange: {
        color: [
          '#313695',
          '#4575b4',
          '#74add1',
          '#abd9e9',
          '#e0f3f8',
          '#ffffbf',
          '#fee090',
          '#fdae61',
          '#f46d43',
          '#d73027',
          '#a50026',
        ],
      },
    },
    xAxis3D: {
      min: 0,
      max: 1,
      type: 'value',
    },
    yAxis3D: {
      min: 0,
      max: 1,
      type: 'value',
    },
    zAxis3D: {
      type: 'value',
      min: 0,
      max: 1,
    },
    grid3D: {
      axisPointer: {
        show: false,
      },
      top: isConstrainedLayout ? -50 : undefined,
      bottom: isConstrainedLayout ? 0 : undefined,
    },
    series: [
      {
        type: 'surface',
        silent: true,
        shading: 'color',
        wireframe: { show: true },
        data,
        dataShape: [80, 80],
      },
      {
        type: 'surface',
        silent: true,
        shading: 'color',
        wireframe: { show: true, opacity: 0 },
        itemStyle: {
          opacity: 0.2,
          color: '#ddd',
        },
        equation: {
          x: {
            step: (inputRange[1] - inputRange[0]) / 80,
            min: inputRange[0],
            max: inputRange[1],
          },
          y: {
            step: (inputRange[1] - inputRange[0]) / 80,
            min: inputRange[0],
            max: inputRange[1],
          },
          z: (x: number, y: number) => sourceFn(new Float32Array([x, y]))[0],
        },
      },
    ],
  };

  return <EChartsReactCore style={style} echarts={echarts} option={chartConfig} lazyUpdate />;
};

export default ResponseViz;
