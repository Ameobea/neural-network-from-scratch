import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { VisualMapComponent, TitleComponent } from 'echarts/components';

import ResponseViz from './ResponseViz';
import CostsPlot from './CostsPlot';

echarts.use([CanvasRenderer, VisualMapComponent, TitleComponent]);

export { ResponseViz, CostsPlot };
