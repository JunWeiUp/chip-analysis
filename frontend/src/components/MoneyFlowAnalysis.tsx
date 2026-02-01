import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpCircle, ArrowDownCircle, Activity, TrendingUp, TrendingDown, RefreshCw, DollarSign } from 'lucide-react';
import { toast } from "sonner";
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface DailyFlow {
  date: string;
  volume: number;
  amount: number;
  change_pct: number;
  turnover: number;
  net_inflow: number;
}

interface MoneyFlowSummary {
  total_inflow: number;
  total_outflow: number;
  net_flow: number;
  inflow_days: number;
  outflow_days: number;
}

interface LargeOrders {
  super_large: number;
  large: number;
  medium: number;
  small: number;
}

interface MoneyFlowData {
  daily_flow: DailyFlow[];
  summary: MoneyFlowSummary;
  large_orders: LargeOrders;
  error?: string;
}

interface MoneyFlowAnalysisProps {
  stockCode: string;
}

const MoneyFlowAnalysis: React.FC<MoneyFlowAnalysisProps> = ({ stockCode }) => {
  const [data, setData] = useState<MoneyFlowData | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchMoneyFlow = async () => {
    if (!stockCode) return;
    
    setLoading(true);
    try {
      const res = await axios.get<MoneyFlowData>(`http://localhost:8001/api/stock/${stockCode}/money-flow`);
      if (res.data.error) {
        toast.error('获取资金流向失败: ' + res.data.error);
        setData(null);
      } else {
        setData(res.data);
      }
    } catch (err) {
      console.error('Error fetching money flow:', err);
      toast.error('获取资金流向失败');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (stockCode) {
      fetchMoneyFlow();
    }
  }, [stockCode]);

  if (!stockCode) {
    return null;
  }

  if (loading) {
    return (
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3 border-b border-slate-50">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary animate-pulse" />
            <CardTitle className="text-base font-semibold">资金流向分析</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <RefreshCw className="h-12 w-12 mb-3 opacity-20 animate-spin" />
            <p className="text-sm">正在加载资金流向数据...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data || data.error) {
    return (
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3 border-b border-slate-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              <CardTitle className="text-base font-semibold">资金流向分析</CardTitle>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchMoneyFlow}
              className="h-8 px-2 text-slate-500 hover:text-primary"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Activity className="h-12 w-12 mb-3 opacity-20" />
            <p className="text-sm">暂无资金流向数据</p>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchMoneyFlow}
              className="mt-4"
            >
              重新加载
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { daily_flow, summary, large_orders } = data;

  // 准备图表数据
  const chartData = {
    labels: daily_flow.map(d => d.date.slice(5)), // 只显示月-日
    datasets: [
      {
        label: '资金净流入',
        data: daily_flow.map(d => d.net_inflow / 10000), // 转换为万元
        backgroundColor: daily_flow.map(d => 
          d.net_inflow > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)'
        ),
        borderColor: daily_flow.map(d => 
          d.net_inflow > 0 ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)'
        ),
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const value = context.parsed.y;
            return `净流入: ${value.toFixed(2)}万元`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          callback: function(value: any) {
            return value + '万';
          }
        }
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

  const isNetInflow = summary.net_flow > 0;
  const inflowRate = summary.inflow_days / (summary.inflow_days + summary.outflow_days) * 100;

  return (
    <Card className="border-none shadow-sm">
      <CardHeader className="pb-3 border-b border-slate-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <CardTitle className="text-base font-semibold">资金流向分析</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={isNetInflow ? 'default' : 'secondary'} className="text-xs">
              {isNetInflow ? '净流入' : '净流出'}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchMoneyFlow}
              disabled={loading}
              className="h-8 px-2 text-slate-500 hover:text-primary"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        {/* 汇总指标 */}
        <div className="grid grid-cols-2 gap-4">
          <div className={`p-4 rounded-lg ${isNetInflow ? 'bg-red-50' : 'bg-green-50'}`}>
            <div className="flex items-center gap-2 mb-2">
              {isNetInflow ? (
                <ArrowUpCircle className="h-4 w-4 text-red-500" />
              ) : (
                <ArrowDownCircle className="h-4 w-4 text-green-500" />
              )}
              <span className="text-xs font-medium text-slate-500">净流向（30日）</span>
            </div>
            <p className={`text-2xl font-bold ${isNetInflow ? 'text-red-600' : 'text-green-600'}`}>
              {isNetInflow ? '+' : ''}{(summary.net_flow / 10000).toFixed(2)}万
            </p>
            <p className="text-xs text-slate-400 mt-1">
              流入{summary.inflow_days}天 / 流出{summary.outflow_days}天
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-50">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-4 w-4 text-primary" />
              <span className="text-xs font-medium text-slate-500">资金活跃度</span>
            </div>
            <p className="text-2xl font-bold text-slate-900">
              {inflowRate.toFixed(0)}%
            </p>
            <p className="text-xs text-slate-400 mt-1">
              流入占比
            </p>
          </div>
        </div>

        {/* 大单资金分析 */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            大单资金流向（近期）
          </h4>
          <div className="space-y-2">
            {[
              { label: '超大单', value: large_orders.super_large, color: 'bg-red-500' },
              { label: '大单', value: large_orders.large, color: 'bg-orange-500' },
              { label: '中单', value: large_orders.medium, color: 'bg-blue-500' },
              { label: '小单', value: large_orders.small, color: 'bg-green-500' },
            ].map(item => {
              const maxValue = Math.max(
                large_orders.super_large,
                large_orders.large,
                large_orders.medium,
                large_orders.small
              );
              const percentage = maxValue > 0 ? (item.value / maxValue) * 100 : 0;
              
              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-600 font-medium">{item.label}</span>
                    <span className={`font-mono font-bold ${item.value > 0 ? 'text-red-600' : 'text-slate-400'}`}>
                      {item.value > 0 ? '+' : ''}{(item.value / 10000).toFixed(2)}万
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${item.color} transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 每日资金流向图表 */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            每日资金净流向趋势
          </h4>
          <div className="h-[200px] w-full">
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>

        {/* 分析提示 */}
        <div className={`p-3 rounded-lg border ${
          isNetInflow 
            ? 'bg-red-50 border-red-100' 
            : 'bg-green-50 border-green-100'
        }`}>
          <div className="flex items-start gap-2">
            <div className="mt-0.5">
              {isNetInflow ? (
                <TrendingUp className="h-4 w-4 text-red-600" />
              ) : (
                <TrendingDown className="h-4 w-4 text-green-600" />
              )}
            </div>
            <div className={`flex-1 text-xs space-y-1 ${
              isNetInflow ? 'text-red-800' : 'text-green-800'
            }`}>
              <p className="font-medium">资金流向分析：</p>
              <ul className={`list-disc list-inside space-y-0.5 text-[11px] ${
                isNetInflow ? 'text-red-700' : 'text-green-700'
              }`}>
                {isNetInflow ? (
                  <>
                    <li>近30日呈现资金净流入态势，市场关注度较高</li>
                    {large_orders.super_large > 0 && (
                      <li>超大单资金流入，可能存在主力资金参与</li>
                    )}
                    {inflowRate > 70 && (
                      <li>流入占比超过70%，资金情绪偏向乐观</li>
                    )}
                  </>
                ) : (
                  <>
                    <li>近30日呈现资金净流出态势，需关注市场情绪变化</li>
                    {Math.abs(summary.net_flow) > Math.abs(summary.total_inflow) * 0.5 && (
                      <li>流出幅度较大，建议谨慎操作</li>
                    )}
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MoneyFlowAnalysis;
