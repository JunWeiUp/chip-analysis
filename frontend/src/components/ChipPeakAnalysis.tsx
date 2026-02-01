import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Activity, Target } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface StockData {
  date: string;
  close: string;
  [key: string]: string | number | boolean | undefined;
}

interface SummaryStats {
  avg_cost: number;
  peak_price: number;
  profit_ratio: number;
  conc_90: { low: number; high: number; concentration: number };
}

interface ChipPeakAnalysisProps {
  data: StockData[];
  allSummaryStats: Record<string, SummaryStats>;
}

interface PeakPoint {
  date: string;
  peakPrice: number;
  closePrice: number;
  avgCost: number;
  profitRatio: number;
  concentration: number;
}

const ChipPeakAnalysis: React.FC<ChipPeakAnalysisProps> = ({ data, allSummaryStats }) => {
  const peakAnalysis = useMemo(() => {
    if (!data || data.length === 0 || !allSummaryStats) return null;

    const peaks: PeakPoint[] = [];
    
    data.forEach((d) => {
      const stats = allSummaryStats[d.date];
      if (stats) {
        peaks.push({
          date: d.date,
          peakPrice: stats.peak_price,
          closePrice: parseFloat(d.close),
          avgCost: stats.avg_cost,
          profitRatio: stats.profit_ratio,
          concentration: stats.conc_90.concentration,
        });
      }
    });

    if (peaks.length === 0) return null;

    // 计算峰值迁移趋势
    const recentPeaks = peaks.slice(-30); // 最近30天
    const peakTrend = recentPeaks.map(p => p.peakPrice);
    const avgPeakChange = peakTrend.length > 1 
      ? ((peakTrend[peakTrend.length - 1] - peakTrend[0]) / peakTrend[0]) * 100
      : 0;

    // 找出关键峰值区域
    const currentPeak = peaks[peaks.length - 1];
    const peakDistanceFromPrice = ((currentPeak.peakPrice - currentPeak.closePrice) / currentPeak.closePrice) * 100;

    // 计算峰值密集度变化
    const concentrationTrend = recentPeaks.map(p => p.concentration);
    const avgConcChange = concentrationTrend.length > 1
      ? concentrationTrend[concentrationTrend.length - 1] - concentrationTrend[0]
      : 0;

    return {
      peaks,
      currentPeak,
      avgPeakChange,
      peakDistanceFromPrice,
      avgConcChange,
      recentPeaks,
    };
  }, [data, allSummaryStats]);

  if (!peakAnalysis) {
    return (
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3 border-b border-slate-50">
          <div className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            <CardTitle className="text-base font-semibold">筹码峰值追踪</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Activity className="h-12 w-12 mb-3 opacity-20" />
            <p className="text-sm">暂无峰值数据</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { currentPeak, avgPeakChange, peakDistanceFromPrice, avgConcChange, recentPeaks } = peakAnalysis;

  // 准备图表数据
  const chartData = {
    labels: recentPeaks.map(p => p.date),
    datasets: [
      {
        label: '收盘价',
        data: recentPeaks.map(p => p.closePrice),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: false,
      },
      {
        label: '筹码峰值',
        data: recentPeaks.map(p => p.peakPrice),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderDash: [5, 5],
        tension: 0.4,
        fill: false,
      },
      {
        label: '平均成本',
        data: recentPeaks.map(p => p.avgCost),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderDash: [2, 2],
        tension: 0.4,
        fill: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 12,
          font: { size: 11 },
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          font: { size: 9 },
        },
      },
    },
  };

  return (
    <Card className="border-none shadow-sm">
      <CardHeader className="pb-3 border-b border-slate-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            <CardTitle className="text-base font-semibold">筹码峰值追踪</CardTitle>
          </div>
          <Badge variant={avgPeakChange > 0 ? 'default' : 'secondary'} className="text-xs">
            {avgPeakChange > 0 ? '峰值上移' : '峰值下移'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        {/* 关键指标 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-slate-50">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-2 w-2 rounded-full bg-red-500" />
              <span className="text-xs font-medium text-slate-500">当前峰值价格</span>
            </div>
            <p className="text-2xl font-bold text-slate-900">
              ¥{currentPeak.peakPrice.toFixed(2)}
            </p>
            <div className="flex items-center gap-1 mt-1">
              {peakDistanceFromPrice > 0 ? (
                <>
                  <TrendingUp className="h-3 w-3 text-red-500" />
                  <span className="text-xs text-red-600">高于收盘 {peakDistanceFromPrice.toFixed(1)}%</span>
                </>
              ) : (
                <>
                  <TrendingDown className="h-3 w-3 text-green-500" />
                  <span className="text-xs text-green-600">低于收盘 {Math.abs(peakDistanceFromPrice).toFixed(1)}%</span>
                </>
              )}
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-50">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="h-3 w-3 text-primary" />
              <span className="text-xs font-medium text-slate-500">峰值迁移趋势</span>
            </div>
            <p className={`text-2xl font-bold ${avgPeakChange > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {avgPeakChange > 0 ? '+' : ''}{avgPeakChange.toFixed(2)}%
            </p>
            <p className="text-xs text-slate-400 mt-1">近30日变化</p>
          </div>
        </div>

        {/* 筹码集中度变化 */}
        <div className="p-4 rounded-lg border border-slate-100 bg-white">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-slate-700">筹码集中度变化</span>
            <Badge variant={avgConcChange < 0 ? 'default' : 'secondary'} className="text-xs">
              {avgConcChange < 0 ? '集中' : '分散'} {Math.abs(avgConcChange).toFixed(1)}%
            </Badge>
          </div>
          <div className="text-xs text-slate-500 space-y-1">
            <p>当前集中度: <span className="font-medium text-slate-700">{currentPeak.concentration.toFixed(2)}%</span></p>
            <p className="text-[11px] text-slate-400">
              {avgConcChange < 0 
                ? '筹码正在向峰值区域集中，可能表示主力吸筹或横盘整理' 
                : '筹码正在分散，可能表示获利盘派发或震荡加剧'}
            </p>
          </div>
        </div>

        {/* 峰值迁移图表 */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-700">近期峰值迁移轨迹</h4>
          <div className="h-[250px] w-full">
            <Line data={chartData} options={chartOptions} />
          </div>
        </div>

        {/* 分析提示 */}
        <div className="p-3 rounded-lg bg-blue-50 border border-blue-100">
          <div className="flex items-start gap-2">
            <div className="mt-0.5">
              <Activity className="h-4 w-4 text-blue-600" />
            </div>
            <div className="flex-1 text-xs text-blue-800 space-y-1">
              <p className="font-medium">峰值分析建议：</p>
              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-blue-700">
                {peakDistanceFromPrice > 5 && (
                  <li>当前价格低于峰值 {Math.abs(peakDistanceFromPrice).toFixed(1)}%，可能存在上行空间</li>
                )}
                {peakDistanceFromPrice < -5 && (
                  <li>当前价格高于峰值 {Math.abs(peakDistanceFromPrice).toFixed(1)}%，注意获利回吐风险</li>
                )}
                {avgPeakChange > 3 && (
                  <li>峰值持续上移，显示筹码成本抬升，可能处于上升趋势</li>
                )}
                {avgPeakChange < -3 && (
                  <li>峰值持续下移，显示筹码成本下降，可能处于下跌趋势</li>
                )}
                {avgConcChange < -5 && (
                  <li>筹码高度集中，关注突破方向</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ChipPeakAnalysis;
