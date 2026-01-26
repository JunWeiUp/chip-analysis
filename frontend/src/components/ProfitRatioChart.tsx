import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

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
  _date?: Date;
}

export interface BacktestTrade {
  type: 'buy' | 'sell';
  date: string;
  price: number;
  ratio: number;
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
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 格式化日期显示
  const formatDate = d3.timeFormat('%Y-%m-%d');
  const parseDate = d3.timeParse('%Y-%m-%d');

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0 || !containerRef.current) return;

    const margin = { top: 20, right: 60, bottom: 40, left: 50 };
    const width = containerRef.current.clientWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // 处理日期
    const processedData = data.map(d => ({
      ...d,
      _date: parseDate(d.date) || new Date(),
    }));

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => d._date) as [Date, Date])
      .range([0, width]);

    const yScalePrice = d3.scaleLinear()
      .domain([
        (d3.min(processedData, (d: ChartDataPoint) => {
          const vals = [d.low_num];
          maSettings.forEach(ma => {
            const val = d[`ma${ma.period}`];
            if (ma.enabled && typeof val === 'number') {
              vals.push(val);
            }
          });
          return Math.min(...vals);
        }) || 0) * 0.98,
        (d3.max(processedData, (d: ChartDataPoint) => {
          const vals = [d.high_num];
          maSettings.forEach(ma => {
            const val = d[`ma${ma.period}`];
            if (ma.enabled && typeof val === 'number') {
              vals.push(val);
            }
          });
          return Math.max(...vals);
        }) || 100) * 1.02
      ])
      .range([innerHeight, 0]);

    const yScaleProfit = d3.scaleLinear()
      .domain([0, 100])
      .range([innerHeight, 0]);

    const yScaleVolume = d3.scaleLinear()
      .domain([0, (d3.max(processedData, d => d.volume_num) || 0) * 4])
      .range([innerHeight, 0]);

    // 定义渐变
    const defs = svg.append('defs');

    // 价格背景渐变
    const priceGradient = defs.append('linearGradient')
      .attr('id', 'price-area-gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    priceGradient.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.15);
    priceGradient.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.02);

    // 价格线外发光
    const glowFilter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-20%').attr('y', '-20%')
      .attr('width', '140%').attr('height', '140%');
    glowFilter.append('feGaussianBlur').attr('stdDeviation', '1').attr('result', 'coloredBlur');
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // 获利比例渐变
    const profitGradient = defs.append('linearGradient')
      .attr('id', 'profit-area-gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    profitGradient.append('stop').attr('offset', '0%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.1);
    profitGradient.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.01);

    // 绘制坐标轴
    const xAxis = d3.axisBottom(xScale).ticks(width / 100).tickFormat(d => formatDate(d as Date));
    const yAxisPrice = d3.axisLeft(yScalePrice).ticks(6).tickFormat(d => d3.format('.2f')(d as number));
    const yAxisProfit = d3.axisRight(yScaleProfit).ticks(5).tickFormat(d => `${d}%`);

    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .style('color', '#64748b')
      .selectAll('text')
      .style('font-size', '11px');

    g.append('g')
      .attr('class', 'y-axis-price')
      .call(yAxisPrice)
      .style('color', '#3b82f6')
      .selectAll('text')
      .style('font-size', '11px');

    g.append('g')
      .attr('class', 'y-axis-profit')
      .attr('transform', `translate(${width},0)`)
      .call(yAxisProfit)
      .style('color', '#ef4444')
      .selectAll('text')
      .style('font-size', '11px');

    // 网格线
    g.append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.05)
      .call(d3.axisLeft(yScalePrice).ticks(6).tickSize(-width).tickFormat(() => ''))
      .selectAll('line')
      .style('stroke', '#64748b')
      .style('stroke-dasharray', '3,3');

    // 1. 绘制成交量柱状图
    g.selectAll('.volume-bar')
      .data(processedData)
      .enter()
      .append('rect')
      .attr('class', 'volume-bar')
      .attr('x', d => xScale(d._date) - (width / processedData.length) * 0.35)
      .attr('y', d => yScaleVolume(d.volume_num))
      .attr('width', (width / processedData.length) * 0.7)
      .attr('height', d => innerHeight - yScaleVolume(d.volume_num))
      .attr('fill', d => d.isUp ? '#ef4444' : '#22c55e')
      .attr('opacity', 0.2)
      .attr('rx', 1);

    // 绘制成交量均线
    if (showIndicators.vma) {
      const vmaLines = [
        { key: 'vma5', color: '#94a3b8' },
        { key: 'vma10', color: '#64748b' }
      ];

      vmaLines.forEach(line => {
        const vmaLine = d3.line<ChartDataPoint & { _date: Date }>()
          .x(d => xScale(d._date))
          .y(d => {
            const val = d[line.key];
            return yScaleVolume(typeof val === 'number' ? val : 0);
          })
          .defined(d => {
            const val = d[line.key];
            return typeof val === 'number';
          })
          .curve(d3.curveMonotoneX);

        g.append('path')
          .datum(processedData)
          .attr('class', `vma-line-${line.key}`)
          .attr('fill', 'none')
          .attr('stroke', line.color)
          .attr('stroke-width', 1)
          .attr('opacity', 0.6)
          .attr('d', vmaLine);
      });
    }

    // 2. 绘制获利比例面积图
    const profitArea = d3.area<ChartDataPoint & { _date: Date }>()
      .x(d => xScale(d._date))
      .y0(innerHeight)
      .y1(d => yScaleProfit(d.profit_ratio))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'url(#profit-area-gradient)')
      .attr('d', profitArea);

    const profitLine = d3.line<ChartDataPoint & { _date: Date }>()
      .x(d => xScale(d._date))
      .y(d => yScaleProfit(d.profit_ratio))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 1.2)
      .attr('stroke-dasharray', '3,2')
      .attr('opacity', 0.6)
      .attr('d', profitLine);

    // 3. 绘制价格面积图
    if (showCloseLine) {
      const priceArea = d3.area<ChartDataPoint & { _date: Date }>()
        .x(d => xScale(d._date))
        .y0(innerHeight)
        .y1(d => yScalePrice(d.close_num))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(processedData)
        .attr('fill', 'url(#price-area-gradient)')
        .attr('d', priceArea);

      const priceLine = d3.line<ChartDataPoint & { _date: Date }>()
        .x(d => xScale(d._date))
        .y(d => yScalePrice(d.close_num))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(processedData)
        .attr('fill', 'none')
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2)
        .attr('filter', 'url(#glow)')
        .attr('d', priceLine);
    }


    // 4. 绘制 MA 线
    maSettings.filter(ma => ma.enabled).forEach(ma => {
      const maKey = `ma${ma.period}`;
      const line = d3.line<ChartDataPoint & { _date: Date }>()
        .x(d => xScale(d._date))
        .y(d => {
          const val = d[maKey];
          return yScalePrice(typeof val === 'number' ? val : 0);
        })
        .curve(d3.curveMonotoneX)
        .defined(d => typeof d[maKey] === 'number');

      g.append('path')
        .datum(processedData)
        .attr('fill', 'none')
        .attr('stroke', ma.color)
        .attr('stroke-width', 1)
        .attr('opacity', 0.8)
        .attr('d', line);
    });

    // 5. 绘制回测交易信号
    if (trades && trades.length > 0) {
      const tradeMarkers = g.append('g').attr('class', 'trade-markers');
      
      trades.forEach((trade) => {
        const tradeDate = parseDate(trade.date);
        if (!tradeDate) return;
        
        const x = xScale(tradeDate);
        const y = yScalePrice(trade.price);
        
        const isBuy = trade.type === 'buy';
        
        // 绘制箭头/三角形
        tradeMarkers.append('path')
          .attr('d', isBuy ? 'M-6,8 L0,0 L6,8 Z' : 'M-6,-8 L0,0 L6,-8 Z')
          .attr('transform', `translate(${x}, ${isBuy ? y + 15 : y - 15})`)
          .attr('fill', isBuy ? '#ef4444' : '#22c55e')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);
          
        // 绘制价格文本
        tradeMarkers.append('text')
          .attr('x', x)
          .attr('y', isBuy ? y + 28 : y - 22)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('font-weight', 'bold')
          .attr('fill', isBuy ? '#ef4444' : '#22c55e')
          .text(`${isBuy ? 'B' : 'S'} ${trade.price.toFixed(2)}`);
      });
    }

    // 6. 交互层
    const interactionGroup = g.append('g').attr('class', 'interaction-layer');
    
    // 十字光标
    const crosshair = interactionGroup.append('g').style('display', 'none');
    
    // 锁定状态的垂直线 (支持多条)
    const lockLinesGroup = interactionGroup.append('g').attr('class', 'lock-lines');

    const verticalLine = crosshair.append('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('y1', 0)
      .attr('y2', innerHeight);

    const priceDot = crosshair.append('circle').attr('r', 4).attr('fill', '#3b82f6').attr('stroke', '#fff').attr('stroke-width', 2);
    const profitDot = crosshair.append('circle').attr('r', 4).attr('fill', '#ef4444').attr('stroke', '#fff').attr('stroke-width', 2);

    // 浮层提示
    const tooltip = d3.select(containerRef.current)
      .append('div')
      .style('position', 'absolute')
      .style('display', 'none')
      .style('background', 'rgba(255, 255, 255, 0.95)')
      .style('backdrop-filter', 'blur(4px)')
      .style('border', '1px solid #e2e8f0')
      .style('border-radius', '8px')
      .style('padding', '12px')
      .style('box-shadow', '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)')
      .style('pointer-events', 'none')
      .style('z-index', 1000)
      .style('min-width', '180px')
      .style('font-family', 'inherit');

    const bisect = d3.bisector((d: ChartDataPoint & { _date: Date }) => d._date).left;

    const updateHoverState = (mouseX: number) => {
      const x0 = xScale.invert(mouseX);
      const i = bisect(processedData, x0, 1);
      const d0 = processedData[i - 1];
      const d1 = processedData[i];
      if (!d0 || !d1) return;
      const d = (x0.getTime() - d0._date.getTime() > d1._date.getTime() - x0.getTime()) ? d1 : d0;

      onHover(d.date);

      crosshair.style('display', null);
      verticalLine.attr('x1', xScale(d._date)).attr('x2', xScale(d._date));
      priceDot.attr('cx', xScale(d._date)).attr('cy', yScalePrice(d.close_num));
      profitDot.attr('cx', xScale(d._date)).attr('cy', yScaleProfit(d.profit_ratio));

      // 更新内容并显示以进行测量
      tooltip
        .html(`
          <div style="font-weight: 600; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; color: #0f172a; font-size: 13px;">${d.date}</div>
          <div style="display: flex; justify-content: space-between; gap: 12px; margin-bottom: 4px;">
            <span style="color: #64748b; font-size: 12px;">收盘:</span>
            <span style="font-weight: 600; color: #3b82f6; font-size: 12px;">${d.close_num.toFixed(2)}</span>
          </div>
          <div style="display: flex; justify-content: space-between; gap: 12px; margin-bottom: 4px;">
            <span style="color: #64748b; font-size: 12px;">获利比例:</span>
            <span style="font-weight: 600; color: #ef4444; font-size: 12px;">${d.profit_ratio}%</span>
          </div>
          <div style="display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px;">
            <span style="color: #64748b; font-size: 12px;">成交量:</span>
            <span style="font-weight: 600; color: #1e293b; font-size: 12px;">${d.volume_num.toLocaleString()}</span>
          </div>
          ${maSettings.filter(ma => ma.enabled).map(ma => {
            const val = (d as ChartDataPoint)[`ma${ma.period}`];
            return `
              <div style="display: flex; justify-content: space-between; gap: 12px; margin-bottom: 2px;">
                <span style="color: ${ma.color}; font-size: 11px;">MA${ma.period}:</span>
                <span style="font-weight: 500; color: #475569; font-size: 11px;">${typeof val === 'number' ? val.toFixed(2) : '-'}</span>
              </div>
            `;
          }).join('')}
        `)
        .style('display', 'block')
        .style('visibility', 'hidden');

      // 计算位置以防裁剪
      const tooltipNode = tooltip.node() as HTMLElement;
      const tooltipWidth = tooltipNode.offsetWidth;
      const tooltipHeight = tooltipNode.offsetHeight;
      const containerWidth = containerRef.current!.clientWidth;
      const containerHeight = containerRef.current!.clientHeight;
      
      let left = xScale(d._date) + margin.left + 15;
      let top = yScalePrice(d.close_num) + margin.top;

      // 如果超出右侧，则移到左侧
      if (left + tooltipWidth > containerWidth) {
        left = xScale(d._date) + margin.left - tooltipWidth - 15;
      }

      // 如果超出底部，则上移
      if (top + tooltipHeight > containerHeight) {
        top = containerHeight - tooltipHeight - 10;
      }

      // 确保不会超出左侧或顶部
      left = Math.max(10, left);
      top = Math.max(10, top);

      tooltip
        .style('visibility', 'visible')
        .style('left', `${left}px`)
        .style('top', `${top}px`);
    };

    // 响应外部 hover 状态和锁定状态
    const renderActiveDate = (date: string | null, isLockLine: boolean = false, color: string = '#cbd5e1') => {
      const d = processedData.find(item => item.date === date);
      if (d) {
        if (isLockLine) {
          lockLinesGroup.append('line')
            .attr('stroke', color)
            .attr('stroke-width', 2)
            .attr('x1', xScale(d._date))
            .attr('x2', xScale(d._date))
            .attr('y1', 0)
            .attr('y2', innerHeight);
        } else {
          crosshair.style('display', null);
          verticalLine.attr('x1', xScale(d._date)).attr('x2', xScale(d._date));
          priceDot.attr('cx', xScale(d._date)).attr('cy', yScalePrice(d.close_num));
          profitDot.attr('cx', xScale(d._date)).attr('cy', yScaleProfit(d.profit_ratio));
        }
      }
    };

    if (hoveredDate) {
      renderActiveDate(hoveredDate);
    } else if (lockedDates.length > 0) {
      // 如果没有 hover，但有锁定，显示最后一个锁定的点（对应右侧筹码）
      renderActiveDate(lockedDates[lockedDates.length - 1]);
    } else {
      crosshair.style('display', 'none');
      tooltip.style('display', 'none');
    }

    // 处理锁定日期显示
    lockLinesGroup.selectAll('*').remove();
    lockedDates.forEach((date, i) => {
      renderActiveDate(date, true, i === 0 ? '#f59e0b' : '#3b82f6');
    });
    
    interactionGroup.append('rect')
      .attr('width', width)
      .attr('height', innerHeight)
      .attr('fill', 'transparent')
      .on('mousemove', (event) => {
        const [mouseX] = d3.pointer(event);
        updateHoverState(mouseX);
      })
      .on('mouseleave', () => {
        onHover(null);
        crosshair.style('display', 'none');
        tooltip.style('display', 'none');
      })
      .on('click', (event) => {
        const [mouseX] = d3.pointer(event);
        const x0 = xScale.invert(mouseX);
        const i = bisect(processedData, x0, 1);
        const d0 = processedData[i - 1];
        const d1 = processedData[i];
        if (!d0 || !d1) return;
        const d = (x0.getTime() - d0._date.getTime() > d1._date.getTime() - x0.getTime()) ? d1 : d0;
        onClick(d.date);
      })
      .on('dblclick', () => {
        onDoubleClick?.();
      });

    return () => {
      tooltip.remove();
    };
  }, [data, hoveredDate, lockedDates, isLocked, maSettings, showCloseLine, height, trades]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: height, position: 'relative' }}>
      <svg ref={svgRef} width="100%" height={height}></svg>
    </div>
  );
};

export default ProfitRatioChart;
