import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface YieldPoint {
  date: string;
  yield: number;
}

interface BacktestYieldChartProps {
  data: YieldPoint[];
  height?: number;
}

const BacktestYieldChart: React.FC<BacktestYieldChartProps> = ({ data, height = 200 }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0 || !containerRef.current) return;

    const margin = { top: 10, right: 40, bottom: 25, left: 40 };
    const width = containerRef.current.clientWidth - margin.left - margin.right;
    const currentHeight = height || containerRef.current.clientHeight || 200;
    const innerHeight = currentHeight - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    svg.attr('width', width + margin.left + margin.right)
       .attr('height', currentHeight);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const parseDate = d3.timeParse('%Y-%m-%d');
    const processedData = data.map(d => ({
      ...d,
      _date: parseDate(d.date) || new Date(),
    }));

    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => d._date) as [Date, Date])
      .range([0, width]);

    const yieldExtent = d3.extent(processedData, d => d.yield) as [number, number];
    const yScale = d3.scaleLinear()
      .domain([Math.min(0, yieldExtent[0] * 1.1), Math.max(0, yieldExtent[1] * 1.1)])
      .range([innerHeight, 0]);

    // Draw zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');

    // Draw area
    const area = d3.area<YieldPoint & { _date: Date }>()
      .x(d => xScale(d._date))
      .y0(yScale(0))
      .y1(d => yScale(d.yield))
      .curve(d3.curveMonotoneX);

    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'yield-gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');

    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.2);
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'url(#yield-gradient)')
      .attr('d', area);

    // Draw line
    const line = d3.line<YieldPoint & { _date: Date }>()
      .x(d => xScale(d._date))
      .y(d => yScale(d.yield))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(width / 100).tickFormat((d) => d3.timeFormat('%m-%d')(d as Date));
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}%`);

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .style('color', '#64748b')
      .selectAll('text')
      .style('font-size', '10px');

    g.append('g')
      .call(yAxis)
      .style('color', '#64748b')
      .selectAll('text')
      .style('font-size', '10px');

    // Tooltip and Hover
    const tooltip = d3.select(containerRef.current)
      .append('div')
      .style('position', 'absolute')
      .style('display', 'none')
      .style('background', 'rgba(255, 255, 255, 0.95)')
      .style('border', '1px solid #e2e8f0')
      .style('border-radius', '4px')
      .style('padding', '4px 8px')
      .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
      .style('pointer-events', 'none')
      .style('font-size', '11px')
      .style('z-index', 100);

    const crosshair = g.append('g').style('display', 'none');
    crosshair.append('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('y1', 0)
      .attr('y2', innerHeight);
    
    crosshair.append('circle')
      .attr('r', 4)
      .attr('fill', '#3b82f6')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    const bisect = d3.bisector((d: YieldPoint & { _date: Date }) => d._date).left;

    g.append('rect')
      .attr('width', width)
      .attr('height', innerHeight)
      .attr('fill', 'transparent')
      .on('mousemove', (event) => {
        const [mouseX] = d3.pointer(event);
        const x0 = xScale.invert(mouseX);
        const i = bisect(processedData, x0, 1);
        const d0 = processedData[i - 1];
        const d1 = processedData[i];
        if (!d0 || !d1) return;
        const d = (x0.getTime() - d0._date.getTime() > d1._date.getTime() - x0.getTime()) ? d1 : d0;

        crosshair.style('display', null);
        crosshair.select('line').attr('x1', xScale(d._date)).attr('x2', xScale(d._date));
        crosshair.select('circle').attr('cx', xScale(d._date)).attr('cy', yScale(d.yield));

        tooltip
          .style('display', 'block')
          .html(`
            <div style="font-weight: bold;">${d.date}</div>
            <div style="color: ${d.yield >= 0 ? '#ef4444' : '#22c55e'}">收益率: ${d.yield}%</div>
          `)
          .style('left', `${xScale(d._date) + margin.left + 10}px`)
          .style('top', `${yScale(d.yield) + margin.top - 30}px`);
      })
      .on('mouseleave', () => {
        crosshair.style('display', 'none');
        tooltip.style('display', 'none');
      });

    return () => {
      tooltip.remove();
    };
  }, [data, height]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: height || '100%' }}>
      <svg ref={svgRef} width="100%" height={height || '100%'}></svg>
    </div>
  );
};

export default BacktestYieldChart;
