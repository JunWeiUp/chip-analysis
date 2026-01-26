import React, { useState, useEffect } from 'react';
import axios from 'axios';
import dayjs from 'dayjs';
import { 
  Calendar as CalendarIcon,
  Download,
  History,
  Info, 
  LayoutDashboard, 
  Loader2,
  Lock as LockIcon,
  RotateCcw,
  Search, 
  Settings,
  TrendingUp, 
  X as XIcon
} from 'lucide-react';
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { DatePickerWithRange } from "@/components/ui/date-range-picker";
import { Toaster } from "@/components/ui/sonner";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import ProfitRatioChart from './components/ProfitRatioChart';
import CoolStockTable from './components/CoolStockTable';
import ChipDistributionChart from './components/ChipDistributionChart';
import BacktestYieldChart from './components/BacktestYieldChart';
import './App.css';

interface ChipDistribution {
  price: number;
  volume: number;
}

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

interface ApiResponse {
  history: StockData[];
  distribution: ChipDistribution[];
  range_distribution: ChipDistribution[];
  all_distributions?: Record<string, ChipDistribution[]>;
  all_summary_stats?: Record<string, SummaryStats>;
  summary_stats?: SummaryStats;
}

interface StockFundamentals {
  info: Record<string, string | number>;
  groups: Record<string, string[]>;
  important_keys: string[];
  error?: string;
}

interface ChipSettings {
  algorithm: 'triangular' | 'average';
  decay: number;
  lookback: number;
  use_turnover: boolean;
  decay_factor: number;
  peakUpperPercent: number;
  peakLowerPercent: number;
  showPeakArea: boolean;
  longTermDays: number;
  mediumTermDays: number;
  shortTermDays: number;
  showCloseLine: boolean;
  maSettings: {
    period: number;
    enabled: boolean;
    color: string;
  }[];
  showIndicators: {
    vma: boolean;
    macd: boolean;
    rsi: boolean;
  };
}

const DEFAULT_SETTINGS: ChipSettings = {
  algorithm: 'triangular',
  decay: 1.0,
  lookback: 250,
  use_turnover: true,
  decay_factor: 1.0,
  peakUpperPercent: 10.0,
  peakLowerPercent: 10.0,
  showPeakArea: true,
  longTermDays: 100,
  mediumTermDays: 10,
  shortTermDays: 2,
  showCloseLine: true,
  maSettings: [
    { period: 5, enabled: true, color: '#3b82f6' },
    { period: 10, enabled: true, color: '#10b981' },
    { period: 20, enabled: true, color: '#f59e0b' },
    { period: 60, enabled: false, color: '#64748b' },
  ],
  showIndicators: {
    vma: true,
    macd: false,
    rsi: false,
  }
};

interface BacktestTrade {
  type: 'buy' | 'sell';
  date: string;
  price: number;
  ratio: number;
  profit?: number;
  cumulative_yield?: number;
}

interface BacktestResult {
  code: string;
  total_yield: number;
  trades: BacktestTrade[];
  yield_curve: { date: string; yield: number }[];
  max_drawdown: number;
  trade_count: number;
  win_rate: number;
}

const App: React.FC = () => {
  const [stockCode, setStockCode] = useState<string>('sh.600000');
  const [dataSource, setDataSource] = useState<string>('baostock');
  const [loading, setLoading] = useState<boolean>(false);
  const [data, setData] = useState<StockData[]>([]);
  const [summaryStats, setSummaryStats] = useState<SummaryStats | null>(null);

  const [distribution, setDistribution] = useState<ChipDistribution[]>([]);
  const [allDistributions, setAllDistributions] = useState<Record<string, ChipDistribution[]>>({});
  const [allSummaryStats, setAllSummaryStats] = useState<Record<string, SummaryStats>>({});
  const [hoveredDate, setHoveredDate] = useState<string | null>(null);
  const [lockedDates, setLockedDates] = useState<string[]>([]);

  const handleChartClick = (date: string | null) => {
    if (!date) return;
    
    setLockedDates(prev => {
      if (prev.includes(date)) {
        const next = prev.filter(d => d !== date);
        toast.info(`已取消锁定日期: ${date}`);
        return next;
      } else {
        if (prev.length >= 2) {
          toast.warning('最多只能锁定两个日期进行对比');
          return prev;
        }
        const next = [...prev, date].sort();
        toast.success(`已锁定日期: ${date}`);
        return next;
      }
    });
  };
  const [searchHistory, setSearchHistory] = useState<string[]>(() => {
    const saved = localStorage.getItem('stock_search_history');
    return saved ? JSON.parse(saved) : [];
  });

  const addToHistory = (code: string) => {
    setSearchHistory(prev => {
      const next = [code, ...prev.filter(c => c !== code)].slice(0, 10);
      localStorage.setItem('stock_search_history', JSON.stringify(next));
      return next;
    });
  };

  const exportData = () => {
    if (data.length === 0) return;
    
    const headers = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "获利比例", "平均成本", "90%成本", "50%成本", "20%成本", "ASR"];
    const rows = data.map(d => {
      const stats = allSummaryStats[d.date];
      return [
        d.date,
        d.open,
        d.high,
        d.low,
        d.close,
        d.volume,
        d.profit_ratio,
        stats?.avg_cost || '',
        stats?.cost_90 || '',
        stats?.cost_50 || '',
        stats?.cost_20 || '',
        stats?.asr || ''
      ].join(',');
    });
    
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob(["\ufeff" + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `stock_data_${stockCode}_${dayjs().format('YYYYMMDD')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success('数据导出成功');
  };
  const [fundamentals, setFundamentals] = useState<StockFundamentals | null>(null);
   const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChipSettings>(DEFAULT_SETTINGS);
  const [dateRange, setDateRange] = useState<{ from: Date; to: Date }>({
    from: dayjs().subtract(1, 'year').toDate(),
    to: dayjs().toDate()
  });
  const [drawerVisible, setDrawerVisible] = useState<boolean>(false);

  const [backtestLoading, setBacktestLoading] = useState<boolean>(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [buyThreshold, setBuyThreshold] = useState<number>(() => {
    const saved = localStorage.getItem('backtest_buy_threshold');
    return saved ? parseFloat(saved) : 5;
  });
  const [sellThreshold, setSellThreshold] = useState<number>(() => {
    const saved = localStorage.getItem('backtest_sell_threshold');
    return saved ? parseFloat(saved) : 50;
  });
  const [strategyType, setStrategyType] = useState<'breakout' | 'mean_reversion' | 'buy_and_hold'>(() => {
    const saved = localStorage.getItem('backtest_strategy_type');
    return (saved as 'breakout' | 'mean_reversion' | 'buy_and_hold') || 'mean_reversion';
  });

  useEffect(() => {
    localStorage.setItem('backtest_buy_threshold', buyThreshold.toString());
  }, [buyThreshold]);

  useEffect(() => {
    localStorage.setItem('backtest_sell_threshold', sellThreshold.toString());
  }, [sellThreshold]);

  useEffect(() => {
    localStorage.setItem('backtest_strategy_type', strategyType);
  }, [strategyType]);

  const handleBacktest = async () => {
    if (!stockCode) {
      toast.warning('请输入股票代码');
      return;
    }
    setBacktestLoading(true);
    try {
      const res = await axios.post<BacktestResult>('http://localhost:8001/api/backtest', {
        code: stockCode,
        buy_threshold: buyThreshold,
        sell_threshold: sellThreshold,
        source: dataSource,
        start_date: dayjs(dateRange.from).format('YYYY-MM-DD'),
        end_date: dayjs(dateRange.to).format('YYYY-MM-DD'),
        chip_settings: settings,
        strategy_type: strategyType
      });
      setBacktestResult(res.data);
      toast.success('回测完成');
    } catch (err: unknown) {
      let msg = '回测失败';
      if (axios.isAxiosError(err)) {
        msg = err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      toast.error('回测失败: ' + msg);
    } finally {
      setBacktestLoading(false);
    }
  };

  const handleSettingsSave = () => {
    // In a real app with Shadcn, we'd use react-hook-form here
    // For now, we'll assume settings are updated directly or via a simpler method
    setDrawerVisible(false);
    toast.success('设置已应用，请重新点击“分析数据”以生效');
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    toast.info('设置已重置为默认值');
  };

  const fetchData = async (targetCode?: string) => {
    const codeToUse = (typeof targetCode === 'string' ? targetCode : null) || stockCode;
    if (!codeToUse) {
      toast.warning('请输入股票代码');
      return;
    }

    setLoading(true);
    setError(null);
    addToHistory(codeToUse);
    try {
      const [historyRes, fundamentalsRes] = await Promise.all([
        axios.get<ApiResponse>(`http://localhost:8001/api/stock/${codeToUse}`, {
          params: {
            source: dataSource,
            algorithm: settings.algorithm,
            decay: settings.decay,
            lookback: settings.lookback,
            use_turnover: settings.use_turnover,
            decay_factor: settings.decay_factor,
            peakUpperPercent: settings.peakUpperPercent,
            peakLowerPercent: settings.peakLowerPercent,
            showPeakArea: settings.showPeakArea,
            longTermDays: settings.longTermDays,
            mediumTermDays: settings.mediumTermDays,
            shortTermDays: settings.shortTermDays,
            start_date: dayjs(dateRange.from).format('YYYY-MM-DD'),
            end_date: dayjs(dateRange.to).format('YYYY-MM-DD'),
            include_all_distributions: true,
          }
        }),
        axios.get<StockFundamentals>(`http://localhost:8001/api/stock/${codeToUse}/fundamentals`)
      ]);

      setData(historyRes.data.history || []);
      setDistribution(historyRes.data.distribution || []);
      setAllDistributions(historyRes.data.all_distributions || {});
      setAllSummaryStats(historyRes.data.all_summary_stats || {});
      setSummaryStats(historyRes.data.summary_stats || null);
      setFundamentals(fundamentalsRes.data);
      setHoveredDate(null);
      
      toast.success('数据获取成功');
    } catch (err: unknown) {
      let errorMsg = '获取数据失败，请检查股票代码是否正确';
      if (axios.isAxiosError(err)) {
        errorMsg = err.response?.data?.detail || err.message;
      }
      setError(errorMsg);
      toast.error(errorMsg);
      setData([]);
      setDistribution([]);
      setAllDistributions({});
      setFundamentals(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // 优先显示 hover 日期，其次显示最后一个锁定日期（对应右侧筹码）
    const activeDate = hoveredDate || (lockedDates.length > 0 ? lockedDates[lockedDates.length - 1] : null);
    
    if (activeDate && allDistributions[activeDate]) {
      setDistribution(allDistributions[activeDate]);
    } else if (!activeDate && data.length > 0) {
      const latestDate = data[data.length - 1].date;
      if (allDistributions[latestDate]) {
        setDistribution(allDistributions[latestDate]);
      }
    }
    
    if (activeDate && allSummaryStats[activeDate]) {
      setSummaryStats(allSummaryStats[activeDate]);
    } else if (!activeDate && data.length > 0) {
      const latestDate = data[data.length - 1].date;
      if (allSummaryStats[latestDate]) {
        setSummaryStats(allSummaryStats[latestDate]);
      }
    }
  }, [hoveredDate, lockedDates, allDistributions, allSummaryStats, data]);

  const activeDate = hoveredDate || (lockedDates.length > 0 ? lockedDates[lockedDates.length - 1] : null);

  const currentClose = activeDate 
    ? parseFloat(data.find(d => d.date === activeDate)?.close || '0')
    : (data.length > 0 ? parseFloat(data[data.length - 1].close) : 0);

  const currentProfitRatio = (activeDate
    ? data.find(d => d.date === activeDate)?.profit_ratio
    : (data.length > 0 ? data[data.length - 1].profit_ratio : 0)) || 0;

  const chartData = (data || []).map((d, i) => {
    const close_num = parseFloat(d.close) || 0;
    const volume_num = parseFloat(d.volume) || 0;
    const open_num = parseFloat(d.open) || 0;
    const high_num = parseFloat(d.high) || 0;
    const low_num = parseFloat(d.low) || 0;
    
    const mas: Record<string, number | null> = {};
    settings.maSettings.forEach(ma => {
      if (ma.enabled) {
        const period = ma.period;
        if (i >= period - 1) {
          const subset = data.slice(i - period + 1, i + 1);
          const sum = subset.reduce((acc, curr) => acc + parseFloat(curr.close), 0);
          mas[`ma${period}`] = parseFloat((sum / period).toFixed(2));
        } else {
          mas[`ma${period}`] = null;
        }
      }
    });

    // 计算成交量均线
    if (settings.showIndicators.vma) {
      const periods = [5, 10];
      periods.forEach(period => {
        if (i >= period - 1) {
          const subset = data.slice(i - period + 1, i + 1);
          const sum = subset.reduce((acc, curr) => acc + parseFloat(curr.volume), 0);
          mas[`vma${period}`] = parseFloat((sum / period).toFixed(0));
        } else {
          mas[`vma${period}`] = null;
        }
      });
    }

    // 计算 MACD (简易版)
    if (settings.showIndicators.macd) {
      // 这里只是简单的占位，完整的 MACD 计算通常需要之前的 EMA 值
      // 为了性能和简单，我们先只处理 VMA，如果用户需要再深入
    }

    return { 
      ...d, 
      open_num,
      high_num,
      low_num,
      close_num,
      volume_num,
      isUp: close_num >= open_num,
      ...mas,
    };
  });

  const renderFundamentals = () => {
    if (!fundamentals) return null;

    // 兼容旧格式
    const info = fundamentals.info || (fundamentals as unknown as Record<string, string | number>);
    const groups = fundamentals.groups || {
      "基本信息": Object.keys(info).filter(k => k !== 'error')
    };

    if (Object.keys(info).length === 0) {
      if (fundamentals.error) {
        return (
          <Alert variant="destructive" className="mb-6">
            <Info className="h-4 w-4" />
            <AlertTitle>基本面数据获取失败</AlertTitle>
            <AlertDescription>{fundamentals.error}</AlertDescription>
          </Alert>
        );
      }
      return null;
    }

    return (
      <Card className="border-none shadow-sm overflow-hidden">
        <CardHeader className="pb-3 border-b border-slate-50 bg-white/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Info className="h-5 w-5 text-primary" />
              <CardTitle className="text-base font-semibold text-slate-900">个股基本面概览</CardTitle>
            </div>
            {info['股票简称'] && (
              <div className="flex items-center gap-2 px-3 py-1 bg-primary/5 rounded-full border border-primary/10">
                <span className="text-sm font-bold text-primary">{String(info['股票简称'])}</span>
                <span className="text-xs text-slate-400">({stockCode})</span>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="divide-y divide-slate-100">
            {Object.entries(groups).map(([groupName, keys]) => {
              const validKeys = keys.filter(k => info[k] !== undefined && info[k] !== null);
              if (validKeys.length === 0) return null;

              return (
                <div key={groupName} className="flex flex-col md:flex-row md:items-center p-4 gap-4">
                  <div className="min-w-[80px]">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{groupName}</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 flex-1 gap-4">
                    {validKeys.map(key => {
                      const value = info[key];
                      
                      // 涨跌幅特殊处理
                      const isChange = key === '涨跌幅' || key === '振幅';
                      const isPrice = ['最新价', '最高', '最低', '今开', '昨收'].includes(key);
                      
                      const changeVal = isChange ? parseFloat(String(value)) : 0;
                      const colorClass = isChange 
                        ? (changeVal > 0 ? 'text-red-600' : changeVal < 0 ? 'text-green-600' : 'text-slate-600')
                        : (isPrice ? 'text-slate-900 font-mono' : 'text-slate-700');

                      return (
                        <div key={key} className="space-y-1">
                          <p className="text-[11px] text-slate-400 font-medium">{key}</p>
                          <p className={`text-sm font-bold ${colorClass} truncate`} title={String(value)}>
                            {String(value)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
            
            {/* 概念板块特殊显示 */}
            {info['所属概念'] && (
              <div className="p-4 bg-slate-50/30">
                <div className="flex flex-col md:flex-row md:items-start gap-4">
                  <div className="min-w-[80px] pt-1">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">所属概念</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {String(info['所属概念']).split('、').map(concept => (
                      <span key={concept} className="px-2 py-0.5 rounded-md bg-white border border-slate-200 text-[11px] text-slate-600 font-medium shadow-sm hover:border-primary/30 transition-colors">
                        {concept}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50/50">
      <Toaster position="top-center" />
      <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary shadow-lg shadow-primary/20">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">股票筹码分析系统</h1>
          </div>
          <div className="hidden items-center gap-4 md:flex">
            <div className="flex items-center gap-2 rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary">
              <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              Shadcn UI 活跃中
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-[1600px] p-4 md:p-6 space-y-6">
        <Card className="border-none shadow-sm overflow-hidden">
          <CardContent className="p-6">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end">
              <div className="grid flex-1 grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">数据源</Label>
                  <Select value={dataSource} onValueChange={setDataSource}>
                    <SelectTrigger className="h-10 bg-slate-50/50 border-slate-200">
                      <SelectValue placeholder="选择数据源" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="baostock">BaoStock</SelectItem>
                      <SelectItem value="ths">同花顺</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">股票代码</Label>
                  <div className="relative group">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                    <Input 
                      value={stockCode}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setStockCode(e.target.value.toUpperCase())}
                      placeholder={dataSource === 'baostock' ? "sh.600000" : "600000"}
                      className="h-10 pl-10 pr-10 bg-slate-50/50 border-slate-200 transition-all focus:ring-2 focus:ring-primary/20 hover:border-primary/30"
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && fetchData()}
                    />
                    {searchHistory.length > 0 && (
                       <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                         <History className="h-4 w-4 text-slate-300" />
                       </div>
                     )}
                   </div>
                   {searchHistory.length > 0 && (
                     <div className="flex flex-wrap gap-2 mt-2">
                       {searchHistory.map(code => (
                          <button
                            key={code}
                            onClick={() => {
                              setStockCode(code);
                              fetchData(code);
                            }}
                            className="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-500 hover:bg-primary/10 hover:text-primary transition-colors border border-slate-200"
                          >
                            {code}
                          </button>
                        ))}
                       <button
                         onClick={() => {
                           setSearchHistory([]);
                           localStorage.removeItem('stock_search_history');
                         }}
                         className="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-400 hover:bg-red-100 transition-colors border border-red-100"
                       >
                         清除历史
                       </button>
                     </div>
                   )}
                 </div>

                <div className="space-y-2 sm:col-span-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">时间范围</Label>
                    <div className="flex gap-2">
                      <button 
                        onClick={() => setDateRange({ from: dayjs().subtract(1, 'year').toDate(), to: dayjs().toDate() })}
                        className="text-[10px] text-primary hover:underline font-medium"
                      >
                        最近1年
                      </button>
                      <button 
                        onClick={() => setDateRange({ from: dayjs().subtract(2, 'year').toDate(), to: dayjs().toDate() })}
                        className="text-[10px] text-primary hover:underline font-medium"
                      >
                        最近2年
                      </button>
                      <button 
                        onClick={() => setDateRange({ from: dayjs('2000-01-01').toDate(), to: dayjs().toDate() })}
                        className="text-[10px] text-primary hover:underline font-medium"
                      >
                        所有时间
                      </button>
                    </div>
                  </div>
                  <DatePickerWithRange 
                    value={dateRange}
                    onChange={(range) => range && setDateRange(range)}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <Button 
                  onClick={() => { fetchData(); }} 
                  disabled={loading} 
                  className="h-11 px-8 rounded-xl shadow-lg shadow-primary/20 transition-all active:scale-95"
                >
                  {loading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <TrendingUp className="mr-2 h-4 w-4" />
                  )}
                  开始分析
                </Button>
                
                {data.length > 0 && (
                  <Button 
                    variant="outline" 
                    onClick={exportData} 
                    className="h-11 px-6 rounded-xl border-slate-200 hover:bg-slate-50 transition-all active:scale-95"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    导出数据
                  </Button>
                )}
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setStockCode('sh.600000');
                    setData([]);
                    setDistribution([]);
                    setSummaryStats(null);
                  }}
                  className="h-10 px-4 border-slate-200 hover:bg-slate-50"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            <div className="mt-4 flex items-center gap-2 text-[13px] text-slate-500">
              <Info className="h-4 w-4 text-primary" />
              {dataSource === 'baostock' 
                ? "提示: 上海股票使用 sh.xxxxxx，深圳股票使用 sz.xxxxxx" 
                : "提示: 直接输入 6 位股票代码即可"}
            </div>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive" className="border-red-100 bg-red-50">
            <Info className="h-4 w-4" />
            <AlertTitle>分析失败</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {renderFundamentals()}

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
          <div className="xl:col-span-8 space-y-6">
            <Card className="border-none shadow-sm overflow-hidden">
              <CardHeader className="pb-2 border-b border-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <LayoutDashboard className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base font-semibold">趋势走势分析</CardTitle>
                  </div>
                  <div className="flex items-center gap-3">
                    <Drawer open={drawerVisible} onOpenChange={setDrawerVisible}>
                      <DrawerTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 px-2 text-slate-500 hover:text-primary hover:bg-primary/5">
                          <Settings className="h-4 w-4 mr-1.5" />
                          <span className="text-xs font-medium">计算配置</span>
                        </Button>
                      </DrawerTrigger>
                      <DrawerContent>
                        <div className="mx-auto w-full max-w-2xl">
                          <DrawerHeader>
                            <DrawerTitle>筹码计算配置</DrawerTitle>
                            <DrawerDescription>调整筹码分布的计算参数和显示效果</DrawerDescription>
                          </DrawerHeader>
                          <div className="p-6 space-y-6 overflow-y-auto max-h-[60vh]">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                              <div className="space-y-4">
                                <h3 className="text-sm font-semibold text-slate-900 border-l-4 border-primary pl-2">算法与逻辑</h3>
                                <div className="flex items-center justify-between space-x-2 p-2 rounded-lg bg-slate-50">
                                  <div className="space-y-0.5">
                                    <Label htmlFor="algorithm">三角分布算法</Label>
                                    <p className="text-[11px] text-slate-500">更平滑的价格分布模型</p>
                                  </div>
                                  <Switch 
                                    id="algorithm" 
                                    checked={settings.algorithm === 'triangular'}
                                    onCheckedChange={(checked) => setSettings({...settings, algorithm: checked ? 'triangular' : 'average'})}
                                  />
                                </div>
                                <div className="flex items-center justify-between space-x-2 p-2 rounded-lg bg-slate-50">
                                  <div className="space-y-0.5">
                                    <Label htmlFor="turnover">考虑换手率</Label>
                                    <p className="text-[11px] text-slate-500">结合交易量进行权重衰减</p>
                                  </div>
                                  <Switch 
                                    id="turnover" 
                                    checked={settings.use_turnover}
                                    onCheckedChange={(checked) => setSettings({...settings, use_turnover: checked})}
                                  />
                                </div>
                                <div className="grid grid-cols-3 gap-4">
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">衰减系数: {settings.decay.toFixed(2)}</Label>
                                    <Input 
                                      type="number" 
                                      min="0.1" 
                                      max="2.0" 
                                      step="0.1" 
                                      value={settings.decay}
                                      onChange={(e) => setSettings({...settings, decay: parseFloat(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">衰减因子: {settings.decay_factor.toFixed(2)}</Label>
                                    <Input 
                                      type="number" 
                                      min="0.1" 
                                      max="2.0" 
                                      step="0.1" 
                                      value={settings.decay_factor}
                                      onChange={(e) => setSettings({...settings, decay_factor: parseFloat(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">回看天数: {settings.lookback}</Label>
                                    <Input 
                                      type="number" 
                                      min="10" 
                                      max="2000" 
                                      step="10" 
                                      value={settings.lookback}
                                      onChange={(e) => setSettings({...settings, lookback: parseInt(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                </div>
                                <div className="grid grid-cols-3 gap-4">
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">长期(天): {settings.longTermDays}</Label>
                                    <Input 
                                      type="number" 
                                      min="20" 
                                      max="500" 
                                      value={settings.longTermDays}
                                      onChange={(e) => setSettings({...settings, longTermDays: parseInt(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">中期(天): {settings.mediumTermDays}</Label>
                                    <Input 
                                      type="number" 
                                      min="5" 
                                      max="100" 
                                      value={settings.mediumTermDays}
                                      onChange={(e) => setSettings({...settings, mediumTermDays: parseInt(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">短期(天): {settings.shortTermDays}</Label>
                                    <Input 
                                      type="number" 
                                      min="1" 
                                      max="20" 
                                      value={settings.shortTermDays}
                                      onChange={(e) => setSettings({...settings, shortTermDays: parseInt(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">峰值提取(上)%: {settings.peakUpperPercent}</Label>
                                    <Input 
                                      type="number" 
                                      min="0" 
                                      max="50" 
                                      step="1" 
                                      value={settings.peakUpperPercent}
                                      onChange={(e) => setSettings({...settings, peakUpperPercent: parseFloat(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                  <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                    <Label className="text-xs">峰值提取(下)%: {settings.peakLowerPercent}</Label>
                                    <Input 
                                      type="number" 
                                      min="0" 
                                      max="50" 
                                      step="1" 
                                      value={settings.peakLowerPercent}
                                      onChange={(e) => setSettings({...settings, peakLowerPercent: parseFloat(e.target.value)})}
                                      className="h-8"
                                    />
                                  </div>
                                </div>
                              </div>

                              <div className="space-y-4">
                                <h3 className="text-sm font-semibold text-slate-900 border-l-4 border-primary pl-2">显示选项</h3>
                                <div className="grid grid-cols-2 gap-2">
                                  <div className="flex items-center justify-between space-x-2 p-2 rounded-lg bg-slate-50">
                                    <Label htmlFor="showCloseLine" className="text-xs">收盘价曲线</Label>
                                    <Switch 
                                      id="showCloseLine" 
                                      checked={settings.showCloseLine}
                                      onCheckedChange={(checked) => setSettings({...settings, showCloseLine: checked})}
                                    />
                                  </div>
                                  <div className="flex items-center justify-between space-x-2 p-2 rounded-lg bg-slate-50">
                                    <Label htmlFor="showPeakArea" className="text-xs">显示峰值区</Label>
                                    <Switch 
                                      id="showPeakArea" 
                                      checked={settings.showPeakArea}
                                      onCheckedChange={(checked) => setSettings({...settings, showPeakArea: checked})}
                                    />
                                  </div>
                                </div>
                                <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                  <Label className="text-xs">常用指标开关</Label>
                                  <div className="grid grid-cols-3 gap-2">
                                    <div className="flex items-center gap-2">
                                      <Switch 
                                        checked={settings.showIndicators.vma}
                                        onCheckedChange={(checked) => setSettings({
                                          ...settings, 
                                          showIndicators: {...settings.showIndicators, vma: checked}
                                        })}
                                      />
                                      <span className="text-[11px]">VMA</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Switch 
                                        checked={settings.showIndicators.macd}
                                        onCheckedChange={(checked) => setSettings({
                                          ...settings, 
                                          showIndicators: {...settings.showIndicators, macd: checked}
                                        })}
                                      />
                                      <span className="text-[11px]">MACD</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Switch 
                                        checked={settings.showIndicators.rsi}
                                        onCheckedChange={(checked) => setSettings({
                                          ...settings, 
                                          showIndicators: {...settings.showIndicators, rsi: checked}
                                        })}
                                      />
                                      <span className="text-[11px]">RSI</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                                  <Label className="text-xs">MA 均线周期设置</Label>
                                  <div className="grid grid-cols-2 gap-2">
                                    {settings.maSettings.map((ma, index) => (
                                      <div key={index} className="flex items-center gap-2 p-1.5 border rounded bg-white">
                                        <Switch 
                                          checked={ma.enabled}
                                          onCheckedChange={(checked) => {
                                            const newMa = [...settings.maSettings];
                                            newMa[index].enabled = checked;
                                            setSettings({...settings, maSettings: newMa});
                                          }}
                                        />
                                        <Input 
                                          type="number"
                                          value={ma.period}
                                          onChange={(e) => {
                                            const newMa = [...settings.maSettings];
                                            newMa[index].period = parseInt(e.target.value) || 1;
                                            setSettings({...settings, maSettings: newMa});
                                          }}
                                          className="h-7 w-12 px-1 text-xs"
                                        />
                                        <div className="h-3 w-3 rounded-full" style={{ backgroundColor: ma.color }} />
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                          <DrawerFooter className="flex-row justify-end gap-3">
                            <Button variant="outline" onClick={resetSettings}>重置默认</Button>
                            <Button onClick={handleSettingsSave}>保存并应用</Button>
                            <DrawerClose asChild>
                              <Button variant="ghost">取消</Button>
                            </DrawerClose>
                          </DrawerFooter>
                        </div>
                      </DrawerContent>
                    </Drawer>
                    {lockedDates.length > 0 && (
                      <div className="flex items-center gap-2">
                        {lockedDates.map(date => (
                          <div key={date} className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-50 text-amber-600 rounded-full border border-amber-100 text-xs font-medium animate-in fade-in zoom-in duration-300">
                            <LockIcon className="h-3 w-3" />
                            锁定: {date}
                            <button 
                              onClick={() => setLockedDates(prev => prev.filter(d => d !== date))}
                              className="ml-1 hover:text-amber-800 transition-colors"
                            >
                              <XIcon className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {(hoveredDate || lockedDates.length > 0) && (
                      <div className="flex items-center gap-2 text-sm font-medium text-slate-500 bg-slate-50 px-3 py-1 rounded-lg border border-slate-100">
                        <CalendarIcon className="h-4 w-4 text-slate-400" />
                        {hoveredDate || (lockedDates.length > 0 ? lockedDates[lockedDates.length - 1] : '')}
                      </div>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                <div className="h-[450px] w-full">
                  <ProfitRatioChart 
                    data={chartData} 
                    trades={backtestResult?.trades}
                    hoveredDate={hoveredDate}
                    lockedDates={lockedDates}
                    onHover={setHoveredDate}
                    onClick={handleChartClick}
                    onDoubleClick={() => setLockedDates([])}
                    isLocked={lockedDates.length > 0}
                    maSettings={settings.maSettings}
                    showIndicators={settings.showIndicators}
                    showCloseLine={settings.showCloseLine}
                    height={450}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-none shadow-sm overflow-hidden">
              <CardHeader className="pb-3 border-b border-slate-50">
                <div className="flex items-center gap-2">
                  <History className="h-5 w-5 text-primary" />
                  <CardTitle className="text-base font-semibold">策略回测</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold">买入获利比例阀值 (%)</Label>
                    <Input 
                      type="number" 
                      value={buyThreshold} 
                      onChange={(e) => setBuyThreshold(parseFloat(e.target.value))}
                      className="bg-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold">卖出获利比例阀值 (%)</Label>
                    <Input 
                      type="number" 
                      value={sellThreshold} 
                      onChange={(e) => setSellThreshold(parseFloat(e.target.value))}
                      className="bg-white"
                      disabled={strategyType === 'buy_and_hold'}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold">策略类型</Label>
                    <Select value={strategyType} onValueChange={(v: 'breakout' | 'mean_reversion' | 'buy_and_hold') => setStrategyType(v)}>
                      <SelectTrigger className="bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mean_reversion">低买高卖 (获利比例 {'<'}= 阀值买入)</SelectItem>
                        <SelectItem value="buy_and_hold">只买不卖 (持仓至回测结束)</SelectItem>
                        <SelectItem value="breakout">高买更高 (获利比例 {'>'}= 阀值买入)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button onClick={handleBacktest} disabled={backtestLoading} className="w-full">
                    {backtestLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    开始回测
                  </Button>
                </div>

                {backtestResult && (
                  <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <Card className="bg-primary/5 border-primary/10 shadow-none">
                        <CardContent className="pt-6">
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">累计收益率</p>
                          <p className={`text-2xl font-bold ${backtestResult.total_yield >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {backtestResult.total_yield > 0 ? '+' : ''}{backtestResult.total_yield}%
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="bg-slate-50 border-slate-100 shadow-none">
                        <CardContent className="pt-6">
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">最大回撤</p>
                          <p className="text-2xl font-bold text-green-600">
                            {backtestResult.max_drawdown}%
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="bg-slate-50 border-slate-100 shadow-none">
                        <CardContent className="pt-6">
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">交易次数</p>
                          <p className="text-2xl font-bold text-slate-900">{backtestResult.trade_count} 次</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-slate-50 border-slate-100 shadow-none">
                        <CardContent className="pt-6">
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">胜率</p>
                          <p className="text-2xl font-bold text-slate-900">{backtestResult.win_rate}%</p>
                        </CardContent>
                      </Card>
                    </div>

                    <Card className="bg-slate-50 border-slate-100 shadow-none">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-semibold text-slate-600">收益率走势</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[200px] w-full">
                          <BacktestYieldChart data={backtestResult.yield_curve} />
                        </div>
                      </CardContent>
                    </Card>

                    <div className="rounded-xl border border-slate-100 overflow-hidden">
                      <Table>
                        <TableHeader className="bg-slate-50">
                          <TableRow>
                            <TableHead className="w-[100px]">类型</TableHead>
                            <TableHead>日期</TableHead>
                            <TableHead>价格</TableHead>
                            <TableHead>获利比例</TableHead>
                            <TableHead className="text-right">盈亏 / 累计</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {backtestResult.trades.map((trade: BacktestTrade, idx: number) => (
                            <TableRow key={idx} className="hover:bg-slate-50/50">
                              <TableCell>
                                <Badge variant={trade.type === 'buy' ? 'default' : 'secondary'} className={trade.type === 'buy' ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600 text-white'}>
                                  {trade.type === 'buy' ? '买入' : '卖出'}
                                </Badge>
                              </TableCell>
                              <TableCell className="font-medium text-slate-600">{trade.date}</TableCell>
                              <TableCell className="font-mono">{trade.price.toFixed(2)}</TableCell>
                              <TableCell>{trade.ratio.toFixed(2)}%</TableCell>
                              <TableCell className="text-right">
                                {trade.type === 'sell' && trade.profit !== undefined && (
                                  <div className="space-y-1">
                                    <p className={`font-bold ${trade.profit >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                                      {trade.profit > 0 ? '+' : ''}{trade.profit}%
                                    </p>
                                    <p className="text-[10px] text-slate-400">
                                      累计: {trade.cumulative_yield}%
                                    </p>
                                  </div>
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                )}
                
                {!backtestResult && !backtestLoading && (
                  <div className="flex flex-col items-center justify-center py-20 text-slate-400 border-2 border-dashed border-slate-100 rounded-2xl">
                    <History className="h-12 w-12 mb-4 opacity-10" />
                    <p className="text-sm">设置参数并点击“开始回测”查看策略表现</p>
                    <p className="text-xs mt-2 opacity-60">建议: 突破模式可尝试 90%/70%，超跌模式可尝试 5%/50%</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <CardTitle className="text-base font-semibold">历史交易明细</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <CoolStockTable 
                  data={data} 
                  onRowHover={setHoveredDate} 
                  onRowClick={handleChartClick}
                  lockedDates={lockedDates}
                />
              </CardContent>
            </Card>
          </div>

          <div className="xl:col-span-4 space-y-6">
            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base font-semibold">筹码分布对比</CardTitle>
                  </div>
                  {lockedDates.length === 2 && (
                    <span className="text-[10px] font-bold text-blue-500 bg-blue-50 px-2 py-0.5 rounded uppercase">对比中</span>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pt-6">
                <div className={lockedDates.length === 2 ? "grid grid-cols-2 gap-2 h-[500px]" : "h-[500px]"}>
                  {lockedDates.length === 2 ? (
                    <>
                      <div className="flex flex-col">
                        <div className="text-[10px] font-medium text-slate-400 mb-2 text-center">日期: {lockedDates[0]}</div>
                        <ChipDistributionChart 
                          data={allDistributions[lockedDates[0]] || []} 
                          currentClose={parseFloat(data.find(d => d.date === lockedDates[0])?.close || '0')}
                          summaryStats={allSummaryStats[lockedDates[0]]}
                          profitRatio={data.find(d => d.date === lockedDates[0])?.profit_ratio}
                        />
                      </div>
                      <div className="flex flex-col border-l border-slate-100 pl-2">
                        <div className="text-[10px] font-medium text-slate-400 mb-2 text-center">日期: {lockedDates[1]}</div>
                        <ChipDistributionChart 
                          data={allDistributions[lockedDates[1]] || []} 
                          currentClose={parseFloat(data.find(d => d.date === lockedDates[1])?.close || '0')}
                          summaryStats={allSummaryStats[lockedDates[1]]}
                          profitRatio={data.find(d => d.date === lockedDates[1])?.profit_ratio}
                        />
                      </div>
                    </>
                  ) : (
                    <ChipDistributionChart 
                      data={distribution} 
                      currentClose={currentClose}
                      summaryStats={summaryStats}
                      profitRatio={currentProfitRatio}
                    />
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <CardTitle className="text-base font-semibold">筹码统计概览</CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                {summaryStats ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50">
                      <span className="text-sm text-slate-500">平均成本</span>
                      <span className="font-bold text-slate-900">{summaryStats.avg_cost.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50">
                      <span className="text-sm text-slate-500">获利比例</span>
                      <span className={`font-bold ${summaryStats.profit_ratio > 50 ? 'text-red-600' : 'text-green-600'}`}>
                        {summaryStats.profit_ratio.toFixed(2)}%
                      </span>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400 uppercase">90% 筹码集中度</span>
                        <span className="text-xs font-bold text-slate-600">{summaryStats.conc_90.concentration.toFixed(2)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary" 
                          style={{ width: `${Math.min(100, summaryStats.conc_90.concentration)}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-500 text-right">
                        范围: {summaryStats.conc_90.low.toFixed(2)} - {summaryStats.conc_90.high.toFixed(2)}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400 uppercase">70% 筹码集中度</span>
                        <span className="text-xs font-bold text-slate-600">{summaryStats.conc_70.concentration.toFixed(2)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary/70" 
                          style={{ width: `${Math.min(100, summaryStats.conc_70.concentration)}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-500 text-right">
                        范围: {summaryStats.conc_70.low.toFixed(2)} - {summaryStats.conc_70.high.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                    <Info className="h-12 w-12 mb-3 opacity-20" />
                    <p className="text-sm">暂无统计数据</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <footer className="border-t bg-white py-8">
        <div className="container px-4 text-center">
          <p className="text-sm text-slate-500">© 2026 股票筹码分析系统 · 基于 Shadcn UI 与 D3.js</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
