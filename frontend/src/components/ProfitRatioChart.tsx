import React, { useMemo, useRef, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';

export interface MASetting {
  period: number;
  color: string;
  enabled: boolean;
}

export interface ChartDataPoint extends Record<string, string | number | boolean | Date | undefined> {
  date: string;
  close_num: number;
  high_num: number;
  low_num: number;
  volume_num: number;
  profit_ratio: number;
  isUp: boolean;
}

export interface BacktestTrade {
  type: 'buy' | 'sell';
  date: string;
  price: number;
  ratio: number;
  amount: number;
  profit?: number;
  cumulative_yield?: number;
}

interface ProfitRatioChartProps {
  data: ChartDataPoint[];
  trades?: BacktestTrade[];
  hoveredDate: string | null;
  lockedDates?: string[];
  onHover: (date: string | null) => void;
  onClick: (date: string | null) => void;
  onDoubleClick?: () => void;
  isLocked: boolean;
  maSettings: MASetting[];
  showCloseLine: boolean;
  showIndicators?: {
    vma: boolean;
    macd: boolean;
    rsi: boolean;
  };
  height?: number;
}

const ProfitRatioChart: React.FC<ProfitRatioChartProps> = ({
  data,
  hoveredDate,
  lockedDates = [],
  onHover,
  onClick,
  onDoubleClick,
  isLocked,
  maSettings,
  showCloseLine,
  showIndicators = { vma: true, macd: false, rsi: false },
  height = 400,
  trades = [],
}) => {
  const echartsRef = useRef<ReactECharts>(null);
  const isZooming = useRef(false);
  const zoomTimer = useRef<number | null>(null);
  const [zoomRange, setZoomRange] = React.useState<{ start: number, end: number }>({ start: 0, end: 100 });

  // 监听外部 hoveredDate 变化，同步图表高亮
  useEffect(() => {
    let timer: number | undefined;
    if (echartsRef.current && hoveredDate) {
      // 如果正在缩放，跳过手动触发 Tooltip，防止 ECharts 内部 DOM 冲突
      if (isZooming.current) return;

      const chart = echartsRef.current.getEchartsInstance();
      // 检查图表实例是否已被销毁
      if (chart && !chart.isDisposed()) {
        const dataIndex = data.findIndex(d => d.date === hoveredDate);
        if (dataIndex !== -1) {
          // 使用 setTimeout 确保在 React 渲染周期之后执行，避免同步冲突
          timer = window.setTimeout(() => {
            try {
              if (!chart.isDisposed() && !isZooming.current) {
                chart.dispatchAction({
                  type: 'showTip',
                  seriesIndex: 0,
                  dataIndex: dataIndex
                });
              }
            } catch (e) {
              console.warn('ECharts showTip error:', e);
            }
          }, 16); // 延迟 16ms (约一帧)，等待 DOM 稳定
        }
      }
    }
    return () => {
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [hoveredDate, data]);

  const option = useMemo<echarts.EChartsOption>(() => {
    if (!data || data.length === 0) {
      return {
        series: []
      };
    }

    const dates = data.map(d => d.date);
    const profitRatios = data.map(d => d.profit_ratio);
    const closePrices = data.map(d => d.close_num);
    const volumes = data.map(d => d.volume_num);

    const series: echarts.SeriesOption[] = [
      // 获利比例面积图
      {
        name: '获利比例',
        type: 'line',
        data: profitRatios,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 1,
        lineStyle: { color: '#ef4444', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(239, 68, 68, 0.1)' },
            { offset: 1, color: 'rgba(239, 68, 68, 0.01)' }
          ])
        },
        zIndex: 1
      } as echarts.LineSeriesOption
    ];

    // 价格曲线
    if (showCloseLine) {
      series.push({
        name: '收盘价',
        type: 'line',
        data: closePrices,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 0,
        lineStyle: { color: '#3b82f6', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59, 130, 246, 0.15)' },
            { offset: 1, color: 'rgba(59, 130, 246, 0.02)' }
          ])
        },
        zIndex: 2
      } as echarts.LineSeriesOption);
    }

    // MA 均线
    maSettings.filter(ma => ma.enabled).forEach(ma => {
      const maData = data.map(d => d[`ma${ma.period}`] as number);
      series.push({
        name: `MA${ma.period}`,
        type: 'line',
        data: maData,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 0,
        lineStyle: { color: ma.color, width: 1.5, opacity: 0.8 },
        zIndex: 2
      } as echarts.LineSeriesOption);
    });

    // 成交量柱状图
    series.push({
      name: '成交量',
      type: 'bar',
      data: volumes.map((v, i) => ({
        value: v,
        itemStyle: {
          color: data[i].isUp ? '#ef4444' : '#22c55e',
          opacity: 0.3
        }
      })),
      yAxisIndex: 2,
      zIndex: 0
    } as echarts.BarSeriesOption);

    // VMA 均线
    if (showIndicators.vma) {
      ['vma5', 'vma10'].forEach((key, idx) => {
        const vmaData = data.map(d => d[key] as number);
        series.push({
          name: key.toUpperCase(),
          type: 'line',
          data: vmaData,
          smooth: true,
          showSymbol: false,
          yAxisIndex: 2,
          lineStyle: { color: idx === 0 ? '#94a3b8' : '#64748b', width: 1, opacity: 0.6 },
          zIndex: 0
        } as echarts.LineSeriesOption);
      });
    }

    // RSI 指标
    if (showIndicators.rsi) {
      const rsiData = data.map(d => d.rsi as number);
      series.push({
        name: 'RSI',
        type: 'line',
        data: rsiData,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 1, // 复用比例坐标轴 (0-100)
        lineStyle: { color: '#8b5cf6', width: 1.5, opacity: 0.8 },
        zIndex: 1
      } as echarts.LineSeriesOption);
    }

    // MACD 指标
    if (showIndicators.macd) {
      const macdHistData = data.map(d => d.macd_hist as number);
      const macdData = data.map(d => d.macd as number);
      const macdSignalData = data.map(d => d.macd_signal as number);

      series.push({
        name: 'MACD_HIST',
        type: 'bar',
        data: macdHistData.map((v) => ({
          value: v,
          itemStyle: { color: v >= 0 ? '#ef4444' : '#22c55e', opacity: 0.4 }
        })),
        yAxisIndex: 3,
        zIndex: 0
      } as echarts.BarSeriesOption);

      series.push({
        name: 'MACD',
        type: 'line',
        data: macdData,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 3,
        lineStyle: { color: '#3b82f6', width: 1 },
        zIndex: 1
      } as echarts.LineSeriesOption);

      series.push({
        name: 'MACD_SIGNAL',
        type: 'line',
        data: macdSignalData,
        smooth: true,
        showSymbol: false,
        yAxisIndex: 3,
        lineStyle: { color: '#f59e0b', width: 1 },
        zIndex: 1
      } as echarts.LineSeriesOption);
    }

    // 交易信号
    if (trades && trades.length > 0) {
      const markPoints: Record<string, unknown>[] = [];
      trades.forEach(trade => {
        const dataIndex = dates.indexOf(trade.date);
        if (dataIndex !== -1) {
          const isBuy = trade.type === 'buy';
          markPoints.push({
            name: isBuy ? '买入' : '卖出',
            value: `${isBuy ? 'B' : 'S'}\n${trade.price.toFixed(2)}`,
            coord: [dataIndex, trade.price],
            itemStyle: {
              color: isBuy ? '#ef4444' : '#22c55e'
            },
            label: {
              show: true,
              position: isBuy ? 'bottom' : 'top',
              fontSize: 10,
              fontWeight: 'bold',
              offset: [0, isBuy ? 5 : -5]
            },
            symbol: isBuy ? 'path://M-6,8 L0,0 L6,8 Z' : 'path://M-6,-8 L0,0 L6,-8 Z',
            symbolSize: 12,
            symbolOffset: [0, isBuy ? 15 : -15]
          });
        }
      });

      if (markPoints.length > 0) {
        // 将交易信号添加到价格曲线或创建一个透明曲线
        const targetSeries = (series.find(s => s.name === '收盘价') || series[0]) as { name?: string, markPoint?: { data: unknown[], silent: boolean } };
        if (targetSeries) {
          targetSeries.markPoint = {
            data: markPoints,
            silent: true // 防止 hover 时触发高亮状态导致位移或消失
          };
        }
      }
    }

    // 标记线 (锁定日期)
    const markLines: Record<string, unknown>[] = lockedDates.map(date => {
      const idx = dates.indexOf(date);
      return {
        xAxis: idx,
        lineStyle: {
          color: '#64748b',
          type: 'dashed',
          width: 1
        },
        label: {
          show: true,
          position: 'end',
          formatter: date
        }
      };
    });

    if (markLines.length > 0) {
      const targetSeries = (series.find(s => s.name === '收盘价') || series[0]) as { name?: string, markLine?: { symbol: string[], data: unknown[], silent: boolean } };
      if (targetSeries) {
        targetSeries.markLine = {
          symbol: ['none', 'none'],
          data: markLines,
          silent: true
        };
      }
    }

    return {
      backgroundColor: 'transparent',
      animation: false, // 关闭动画提升性能
      emphasis: {
        disabled: true // 禁用全局高亮，防止 hover 时组件闪烁或消失
      },
      tooltip: {
        trigger: 'axis',
        renderMode: 'html',
        confine: true, // 将 tooltip 限制在图表容器内，防止溢出导致的节点异常
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: '#6a7985'
          }
        },
        formatter: (params: unknown) => {
          const paramsList = params as { axisValue: string, seriesName: string, color: string, value: number }[];
          if (!paramsList || paramsList.length === 0) return '';
          
          const date = paramsList[0].axisValue;
          let html = `<div class="font-sans"><div class="font-bold mb-1 border-bottom pb-1">${date}</div>`;
          paramsList.forEach((p) => {
            if (p.seriesName === '成交量') {
              html += `<div class="flex justify-between items-center gap-4">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background-color:${p.color};margin-right:5px;"></span>
                <span>${p.seriesName}:</span>
                <span class="font-mono">${(p.value / 10000).toFixed(2)}万</span>
              </div>`;
            } else if (p.seriesName) {
              const val = typeof p.value === 'number' ? p.value.toFixed(2) : p.value;
              const unit = p.seriesName === '获利比例' ? '%' : '';
              html += `<div class="flex justify-between items-center gap-4">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background-color:${p.color};margin-right:5px;"></span>
                <span>${p.seriesName}:</span>
                <span class="font-mono">${val}${unit}</span>
              </div>`;
            }
          });
          html += '</div>';
          return html;
        }
      },
      legend: {
        data: ['收盘价', '获利比例', '成交量', ...maSettings.filter(m => m.enabled).map(m => `MA${m.period}`)],
        textStyle: { color: '#64748b', fontSize: 11 },
        top: 0
      },
      grid: [
        {
          left: '50',
          right: '60',
          top: '30',
          bottom: '80'
        }
      ],
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#64748b', fontSize: 11 },
        splitLine: { show: false }
      },
      yAxis: [
        {
          type: 'value',
          name: '价格',
          position: 'left',
          axisLine: { show: false },
          axisLabel: { color: '#3b82f6', fontSize: 11 },
          splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
          scale: true
        },
        {
          type: 'value',
          name: '比例',
          position: 'right',
          min: 0,
          max: 100,
          axisLine: { show: false },
          axisLabel: { color: '#ef4444', fontSize: 11, formatter: '{value}%' },
          splitLine: { show: false }
        },
        {
          type: 'value',
          name: '成交量',
          show: false,
          max: (v: { max: number }) => v.max * 4 // 成交量显示在底部 1/4 区域
        },
        {
          type: 'value',
          name: 'MACD',
          show: false,
          max: (v: { max: number, min: number }) => Math.max(Math.abs(v.max), Math.abs(v.min)) * 4
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          start: zoomRange.start,
          end: zoomRange.end
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          start: zoomRange.start,
          end: zoomRange.end,
          bottom: 10,
          height: 30,
          borderColor: 'transparent',
          backgroundColor: '#f1f5f9',
          fillerColor: 'rgba(59, 130, 246, 0.1)',
          handleIcon: 'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
          handleSize: '80%',
          handleStyle: {
            color: '#fff',
            shadowBlur: 3,
            shadowColor: 'rgba(0, 0, 0, 0.6)',
            shadowOffsetX: 2,
            shadowOffsetY: 2
          },
          textStyle: { color: '#64748b' }
        }
      ],
      series: series
    };
  }, [data, maSettings, showCloseLine, showIndicators, trades, lockedDates, zoomRange]);

  // 监听鼠标移动，实现外部联动
  const handleMouseMove = useCallback((params: { dataIndex?: number }) => {
    // 如果正在缩放或已锁定，不触发外部联动
    if (isZooming.current || isLocked) return;

    // 无论是否在 series 上，只要有 dataIndex 就触发（适用于 axis trigger）
    if (params && typeof params.dataIndex === 'number') {
      const date = data[params.dataIndex]?.date;
      if (date) onHover(date);
    }
  }, [data, isLocked, onHover]);

  // 使用 useEffect 绑定底层 zr 事件，确保“任何地方”点击都能触发
  useEffect(() => {
    const chart = echartsRef.current?.getEchartsInstance();
    if (!chart) return;

    const zr = chart.getZr();
    
    const handleZrClick = (params: { offsetX: number, offsetY: number }) => {
      // 如果正在缩放（drag 结束），不触发锁定
      if (isZooming.current) return;

      const pointInPixel = [params.offsetX, params.offsetY];
      // 将像素坐标转换为数据索引，即使在 grid 外部点击也能获取最近的索引
      const result = chart.convertFromPixel({ seriesIndex: 0 }, pointInPixel);
      
      if (result && typeof result[0] === 'number') {
        let xIndex = Math.round(result[0]);
        // 边界检查与限制
        if (xIndex < 0) xIndex = 0;
        if (xIndex >= data.length) xIndex = data.length - 1;
        
        const date = data[xIndex]?.date;
        if (date) {
          onClick(date);
        }
      }
    };

    zr.on('click', handleZrClick);
    return () => {
      zr.off('click', handleZrClick);
    };
  }, [data, onClick]);

  const onEvents = useMemo(() => ({
    'datazoom': (params: { start?: number, end?: number, batch?: { start: number, end: number }[] }) => {
      // 记录正在缩放状态，防止触发 Tooltip 冲突
      isZooming.current = true;
      if (zoomTimer.current !== null) {
        window.clearTimeout(zoomTimer.current);
      }
      zoomTimer.current = window.setTimeout(() => {
        isZooming.current = false;
        zoomTimer.current = null;
      }, 200);

      // 缩放时隐藏当前图表的 Tooltip，防止 DOM 冲突
      const chart = echartsRef.current?.getEchartsInstance();
      if (chart && !chart.isDisposed()) {
        chart.dispatchAction({ type: 'hideTip' });
      }

      // 记录缩放范围，防止重绘时丢失
      let start = 0;
      let end = 100;
      if (params.batch && params.batch.length > 0) {
        start = params.batch[0].start;
        end = params.batch[0].end;
      } else {
        start = params.start ?? 0;
        end = params.end ?? 100;
      }
      setZoomRange({ start, end });
    },
    'dblclick': () => {
      onDoubleClick?.();
    },
    'globalout': () => {
      if (!isLocked) onHover(null);
    },
    'updateAxisPointer': (params: { axesInfo?: { value: number }[] }) => {
      // 如果正在缩放或已锁定，不触发外部联动
      if (isZooming.current || isLocked) return;

      // updateAxisPointer 是最可靠的获取当前轴位置的事件
      if (params.axesInfo && params.axesInfo.length > 0) {
        const dataIndex = params.axesInfo[0].value;
        if (typeof dataIndex === 'number' && data[dataIndex]) {
          onHover(data[dataIndex].date);
        }
      }
    }
  }), [data, isLocked, onDoubleClick, onHover]);

  return (
    <div className="w-full bg-white rounded-lg p-2" style={{ height: height || '100%' }}>
      <ReactECharts
        ref={echartsRef}
        option={option}
        style={{ height: '100%', width: '100%' }}
        onEvents={{
          ...onEvents,
          'mousemove': handleMouseMove
        }}
        notMerge={true}
      />
    </div>
  );
};

export default ProfitRatioChart;
