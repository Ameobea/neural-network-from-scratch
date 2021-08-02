import React from 'react';
import * as echarts from 'echarts';
import EChartsReactCore from 'echarts-for-react/lib/core';
import * as R from 'ramda';

interface CostsPlotProps {
  costs: number[];
}

const CostsPlot: React.FC<CostsPlotProps> = ({ costs }) => {
  let averagedCosts = [];
  let indices: number[] = [];
  if (costs.length <= 1000) {
    averagedCosts = costs;
    indices = R.range(0, costs.length);
  } else {
    const stride = Math.floor(costs.length / 1000);

    let strideTotal = 0;
    let ixWithinStride = 0;
    for (let i = 0; i < costs.length; i++) {
      strideTotal += costs[i];
      ixWithinStride += 1;
      if (ixWithinStride > stride) {
        ixWithinStride = 0;
        averagedCosts.push(strideTotal / stride);
        indices.push(i + 1);
        strideTotal = 0;
      }
    }
    // Take the last `stride` elements in the array and average them for the final datapoint
    strideTotal = 0;
    for (let i = costs.length - stride; i < costs.length; i++) {
      strideTotal += costs[i];
    }
    averagedCosts.push(strideTotal / stride);
    indices.push(costs.length);
  }

  const style = { height: 200 };
  const chartConfig = {
    backgroundColor: '#101010',
    title: {
      show: true,
      text: 'Cost',
      textStyle: {
        color: '#dfdfdf',
        fontWeight: 'normal',
        fontSize: 14,
      },
    },
    xAxis: {
      type: 'category',
      data: indices,
    },
    yAxis: {
      type: 'log',
    },
    series: [
      {
        type: 'line',
        data: averagedCosts,
        itemStyle: {
          opacity: 0,
        },
      },
    ],
  };

  return <EChartsReactCore style={style} echarts={echarts} option={chartConfig} />;
};

export default CostsPlot;
