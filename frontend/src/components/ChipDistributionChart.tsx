import React, { useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';

interface ChipDistribution {
  price: number;
  volume: number;
}

interface SummaryStats {
  avg_cost: number;
  conc_90: { low: number; high: number; concentration: number };
  conc_70: { low: number; high: number; concentration: number };
  peak_price: number;
  profit_ratio: number;
  asr: number;
  cost_20: number;
  cost_50: number;
  cost_90: number;
}

interface ChipDistributionChartProps {
  data: ChipDistribution[];
  currentClose: number;
  summaryStats?: SummaryStats | null;
  profitRatio?: number;
  height?: number;
  width?: number;
}

const ChipDistributionChart: React.FC<ChipDistributionChartProps> = ({ 
  data, 
  currentClose, 
  summaryStats,
  profitRatio,
  height = 400,
  width = 300 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const chartId = useId().replace(/:/g, ''); // Remove colons as they are not valid in IDs for all browsers
  const profitGradientId = `profit-gradient-${chartId}`;
  const lossGradientId = `loss-gradient-${chartId}`;

  useEffect(() => {
    if (!containerRef.current || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    
    // Clear SVG if no data
    if (!data || data.length === 0) {
      svg.selectAll('*').remove();
      return;
    }

    const margin = { top: 10, right: 30, bottom: 30, left: 50 };
    const chartWidth = (width || containerRef.current.clientWidth) - margin.left - margin.right;
    const chartHeight = (height || containerRef.current.clientHeight || 400) - margin.top - margin.bottom;

    svg.attr('width', chartWidth + margin.left + margin.right)
       .attr('height', chartHeight + margin.top + margin.bottom);

    let g = svg.select<SVGGElement>('g.main-group');
    if (g.empty()) {
      g = svg.append('g')
        .attr('class', 'main-group');
      
      // Add axes groups
      g.append('g').attr('class', 'x-axis');
      g.append('g').attr('class', 'y-axis');
      
      // Add price line group
      g.append('g').attr('class', 'price-line-group');
      
      // Add bars group
      g.append('g').attr('class', 'bars-group');

      // Add gradients
      const defs = svg.append('defs');
      
      const profitGradient = defs.append('linearGradient')
        .attr('id', profitGradientId)
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
      profitGradient.append('stop').attr('offset', '0%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.6);
      profitGradient.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.9);

      const lossGradient = defs.append('linearGradient')
        .attr('id', lossGradientId)
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
      lossGradient.append('stop').attr('offset', '0%').attr('stop-color', '#22c55e').attr('stop-opacity', 0.6);
      lossGradient.append('stop').attr('offset', '100%').attr('stop-color', '#22c55e').attr('stop-opacity', 0.9);
    }

    // Always update transforms in case width/height changed
    g.attr('transform', `translate(${margin.left},${margin.top})`);
    g.select('.x-axis').attr('transform', `translate(0,${chartHeight})`);

    // Scales
    const yScale = d3.scaleLinear()
      .domain([d3.min(data, d => d.price) || 0, d3.max(data, d => d.price) || 100])
      .range([chartHeight, 0])
      .nice();

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.volume) || 1])
      .range([0, chartWidth])
      .nice();

    // Axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(5)
      .tickFormat(d => {
        const v = d as number;
        return v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v.toString();
      });
    
    const yAxis = d3.axisLeft(yScale)
      .ticks(10)
      .tickFormat(d => `${(d as number).toFixed(2)}`);

    g.select<SVGGElement>('.x-axis')
      .transition().duration(300)
      .call(xAxis)
      .selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#64748b');

    g.select<SVGGElement>('.y-axis')
      .transition().duration(300)
      .call(yAxis)
      .selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#64748b');

    // Style axes lines
    g.selectAll('.domain').style('stroke', '#e2e8f0');
    g.selectAll('.tick line').style('stroke', '#f1f5f9');

    // Bars
    const barHeight = Math.max(1, chartHeight / data.length - 1);
    const bars = g.select('.bars-group').selectAll<SVGRectElement, ChipDistribution>('rect.chip-bar')
      .data(data, (d) => (d as ChipDistribution).price);

    // Exit
    bars.exit().remove();

    // Enter + Update
    bars.enter()
      .append('rect')
      .attr('class', 'chip-bar')
      .attr('y', d => yScale(d.price) - barHeight / 2)
      .attr('x', 0)
      .attr('height', barHeight)
      .attr('width', 0)
      .merge(bars)
      .transition().duration(200)
      .attr('y', d => yScale(d.price) - barHeight / 2)
      .attr('height', barHeight)
      .attr('width', d => xScale(d.volume))
      .attr('rx', 1)
      .attr('fill', d => d.price < currentClose ? `url(#${profitGradientId})` : `url(#${lossGradientId})`);

    // Price Line
    const priceLineGroup = g.select('.price-line-group');
    let priceLine = priceLineGroup.select<SVGLineElement>('line.current-price-line');
    let priceLabel = priceLineGroup.select<SVGTextElement>('text.current-price-label');
    let priceRect = priceLineGroup.select<SVGRectElement>('rect.current-price-bg');

    if (priceLine.empty()) {
      priceLine = priceLineGroup.append('line').attr('class', 'current-price-line');
      priceRect = priceLineGroup.append('rect').attr('class', 'current-price-bg');
      priceLabel = priceLineGroup.append('text').attr('class', 'current-price-label');
    }

    const yPos = yScale(currentClose);

    priceLine
      .transition().duration(200)
      .attr('x1', 0)
      .attr('x2', chartWidth)
      .attr('y1', yPos)
      .attr('y2', yPos)
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,2');

    const labelText = `${currentClose.toFixed(2)}`;
    const labelPadding = 6;
    
    priceLabel
      .text(labelText)
      .attr('x', chartWidth - 8)
      .attr('y', yPos)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .style('font-size', '11px')
      .style('font-weight', '600')
      .style('fill', '#fff')
      .style('pointer-events', 'none');

    const labelBBox = (priceLabel.node() as SVGTextElement).getBBox();
    
    priceRect
      .attr('x', labelBBox.x - labelPadding)
      .attr('y', labelBBox.y - 2)
      .attr('width', labelBBox.width + labelPadding * 2)
      .attr('height', labelBBox.height + 4)
      .attr('fill', '#3b82f6')
      .attr('rx', 4);
    
    // Bring label to front
    priceLabel.raise();

    // Summary Stats Overlay
    if (summaryStats) {
      let statsGroup = g.select<SVGGElement>('.stats-overlay');
      if (statsGroup.empty()) {
        statsGroup = g.append('g').attr('class', 'stats-overlay');
      }
      statsGroup.selectAll('*').remove();

      const stats = [
        { label: '平均成本', value: (summaryStats.avg_cost || 0).toFixed(2), color: '#0f172a' },
        { label: '90%筹码', value: `${(summaryStats.conc_90?.low || 0).toFixed(2)}-${(summaryStats.conc_90?.high || 0).toFixed(2)}`, color: '#0f172a' },
        { label: '集中度', value: `${(summaryStats.conc_90?.concentration || 0).toFixed(2)}%`, color: '#3b82f6' },
        { label: '获利比例', value: `${(profitRatio !== undefined ? profitRatio : (summaryStats.profit_ratio || 0)).toFixed(1)}%`, color: (profitRatio !== undefined ? profitRatio : (summaryStats.profit_ratio || 0)) > 80 ? '#ef4444' : '#22c55e' },
        { label: 'ASR(活跃)', value: `${(summaryStats.asr || 0).toFixed(1)}%`, color: '#0f172a' },
        { label: '90%成本', value: (summaryStats.cost_90 || 0).toFixed(2), color: '#0f172a' },
        { label: '50%成本', value: (summaryStats.cost_50 || 0).toFixed(2), color: '#0f172a' },
        { label: '20%成本', value: (summaryStats.cost_20 || 0).toFixed(2), color: '#0f172a' }
      ];

      stats.forEach((stat, i) => {
        const row = statsGroup.append('g').attr('transform', `translate(10, ${i * 20 + 10})`);
        row.append('text')
          .text(stat.label)
          .style('font-size', '11px')
          .style('fill', '#64748b');
        row.append('text')
          .text(stat.value)
          .attr('x', 55)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .style('fill', stat.color);
      });
    }

  }, [data, currentClose, height, width, summaryStats, profitRatio]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: height || '100%' }}>
      <svg ref={svgRef} style={{ overflow: 'visible' }} />
    </div>
  );
};

export default ChipDistributionChart;
