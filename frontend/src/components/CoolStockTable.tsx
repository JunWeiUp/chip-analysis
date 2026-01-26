import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Inbox } from 'lucide-react';

interface StockData {
  date: string;
  code: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  profit_ratio: number;
  turn?: number;
}

interface CoolStockTableProps {
  data: StockData[];
  onRowHover?: (date: string | null) => void;
  onRowClick?: (date: string | null) => void;
  lockedDates?: string[];
}

const CoolStockTable: React.FC<CoolStockTableProps> = ({ data, onRowHover, onRowClick, lockedDates = [] }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Clear previous content
    d3.select(containerRef.current).selectAll("*").remove();

    const table = d3.select(containerRef.current)
      .append("table")
      .attr("class", "w-full border-collapse text-left text-sm")
      .style("font-family", "inherit");

    // Header
    const headers = ["日期", "趋势", "行情 (开/收/高/低)", "成交量", "获利比例"];
    table.append("thead")
      .attr("class", "bg-slate-50/80 sticky top-0 z-10")
      .append("tr")
      .selectAll("th")
      .data(headers)
      .enter()
      .append("th")
      .text(d => d)
      .attr("class", "px-4 py-3 font-semibold text-slate-500 uppercase tracking-wider text-[11px] border-b border-slate-100");

    const tbody = table.append("tbody");

    // Rows
    const rows = tbody.selectAll("tr")
      .data(data.slice(-50).reverse()) // Show last 50 records
      .enter()
      .append("tr")
      .attr("class", d => {
        const isLocked = lockedDates.includes(d.date);
        const lockIndex = lockedDates.indexOf(d.date);
        const lockClass = lockIndex === 0 ? 'bg-amber-50/50 hover:bg-amber-50' : 'bg-blue-50/50 hover:bg-blue-50';
        return `border-b border-slate-50 transition-all cursor-default ${isLocked ? lockClass : 'hover:bg-slate-50/50'}`;
      })
      .on("mouseenter", function(_event, d) {
        onRowHover?.(d.date);
      })
      .on("mouseleave", function() {
        onRowHover?.(null);
      })
      .on("click", function(_event, d) {
        onRowClick?.(d.date);
      });

    // Date Cell
    rows.append("td")
      .attr("class", "px-4 py-3 text-slate-900 font-medium")
      .text(d => d.date);

    // Trend Cell (Sparkline)
    rows.append("td")
      .attr("class", "px-4 py-3")
      .each(function(d) {
        const currentIdx = data.findIndex(item => item.date === d.date);
        const sparkData = data.slice(Math.max(0, currentIdx - 10), currentIdx + 1)
          .map(item => parseFloat(item.close));

        if (sparkData.length < 2) return;

        const width = 80;
        const height = 32;
        const svg = d3.select(this).append("svg")
          .attr("width", width)
          .attr("height", height)
          .attr("class", "overflow-visible");

        const x = d3.scaleLinear()
          .domain([0, sparkData.length - 1])
          .range([0, width]);

        const y = d3.scaleLinear()
          .domain([d3.min(sparkData)!, d3.max(sparkData)!])
          .range([height - 4, 4]);

        const line = d3.line<number>()
          .x((_, i) => x(i))
          .y(d => y(d))
          .curve(d3.curveMonotoneX);

        const isUp = sparkData[sparkData.length - 1] >= sparkData[0];
        const color = isUp ? "#ef4444" : "#22c55e";

        // Background area for sparkline
        const area = d3.area<number>()
          .x((_, i) => x(i))
          .y0(height)
          .y1(d => y(d))
          .curve(d3.curveMonotoneX);

        const gradientId = `spark-gradient-${d.date.replace(/-/g, '')}`;
        const defs = svg.append("defs");
        const gradient = defs.append("linearGradient")
          .attr("id", gradientId)
          .attr("x1", "0%").attr("y1", "0%")
          .attr("x2", "0%").attr("y2", "100%");
        gradient.append("stop").attr("offset", "0%").attr("stop-color", color).attr("stop-opacity", 0.1);
        gradient.append("stop").attr("offset", "100%").attr("stop-color", color).attr("stop-opacity", 0);

        svg.append("path")
          .datum(sparkData)
          .attr("fill", `url(#${gradientId})`)
          .attr("d", area);

        svg.append("path")
          .datum(sparkData)
          .attr("fill", "none")
          .attr("stroke", color)
          .attr("stroke-width", 1.5)
          .attr("d", line);
          
        svg.append("circle")
          .attr("cx", x(sparkData.length - 1))
          .attr("cy", y(sparkData[sparkData.length - 1]))
          .attr("r", 2)
          .attr("fill", color);
      });

    // OHLC Cell
    rows.append("td")
      .attr("class", "px-4 py-3")
      .html(d => {
        const open = parseFloat(d.open);
        const close = parseFloat(d.close);
        const high = parseFloat(d.high);
        const low = parseFloat(d.low);
        const colorClass = close >= open ? "text-red-500" : "text-green-600";
        return `<div class="flex items-baseline gap-3">
          <span class="font-bold text-base ${colorClass}">${close.toFixed(2)}</span>
          <span class="text-[11px] text-slate-400">
            开 <span class="text-slate-600 font-medium">${open.toFixed(2)}</span> | 
            <span class="text-red-400">高 ${high.toFixed(2)}</span> / 
            <span class="text-green-500">低 ${low.toFixed(2)}</span>
          </span>
        </div>`;
      });

    // Volume Cell
    const maxVol = d3.max(data, d => parseFloat(d.volume)) || 1;
    rows.append("td")
      .attr("class", "px-4 py-3")
      .each(function(d) {
        const vol = parseFloat(d.volume);
        const width = 100;
        const height = 16;
        const svg = d3.select(this).append("svg")
          .attr("width", width)
          .attr("height", height);

        svg.append("rect")
          .attr("x", 0)
          .attr("y", 2)
          .attr("width", (vol / maxVol) * width)
          .attr("height", height - 4)
          .attr("class", "fill-slate-200")
          .attr("rx", 2);

        const volText = vol > 1000000 ? (vol / 1000000).toFixed(2) + "M" : vol > 1000 ? (vol / 1000).toFixed(1) + "K" : vol;
        
        d3.select(this).append("span")
          .text(volText)
          .attr("class", "ml-2 text-[11px] font-medium text-slate-500");
      });

    // Profit Ratio Cell
    rows.append("td")
      .attr("class", "px-4 py-3")
      .each(function(d) {
        const ratio = d.profit_ratio;
        const colorClass = ratio > 80 ? "bg-red-500" : ratio > 50 ? "bg-orange-500" : "bg-green-500";
        const textColorClass = ratio > 80 ? "text-red-600" : ratio > 50 ? "text-orange-600" : "text-green-600";
        
        const container = d3.select(this).append("div")
          .attr("class", "flex items-center gap-3");
          
        const track = container.append("div")
          .attr("class", "w-20 h-1.5 bg-slate-100 rounded-full overflow-hidden");
          
        track.append("div")
          .attr("class", `h-full ${colorClass}`)
          .style("width", `${ratio}%`);
          
        container.append("span")
          .text(`${ratio.toFixed(1)}%`)
          .attr("class", `text-[13px] font-bold ${textColorClass} min-w-[45px]`);
      });
      
    return () => {
       d3.select(containerRef.current).selectAll("*").remove();
     };
   }, [data, onRowHover, onRowClick, lockedDates]);

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400">
        <Inbox className="h-12 w-12 mb-2 opacity-20" />
        <p className="text-sm">暂无数据</p>
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto">
      <div ref={containerRef} />
    </div>
  );
};

export default CoolStockTable;
