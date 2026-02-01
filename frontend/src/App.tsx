import React, { useState, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import type { AxiosResponse } from 'axios';
import dayjs from 'dayjs';
import { 
  Calendar as CalendarIcon,
  Download,
  History,
  Info, 
  LayoutDashboard, 
  Loader2,
  Lock as LockIcon,
  Maximize2,
  Minimize2,
  RotateCcw,
  Search, 
  Settings,
  Star,
  TrendingUp, 
  Activity,
  BarChart3,
  ShieldCheck,
  Zap,
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
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { DatePickerWithRange } from "@/components/ui/date-range-picker";
import { Toaster } from "@/components/ui/sonner";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import ProfitRatioChart from './components/ProfitRatioChart';
import ChipDistributionChart from './components/ChipDistributionChart';
import BacktestYieldChart from './components/BacktestYieldChart';
import FavoriteStocks from './components/FavoriteStocks';
import type { FavoriteStock } from './components/FavoriteStocks';
import ChipPeakAnalysis from './components/ChipPeakAnalysis';
import MoneyFlowAnalysis from './components/MoneyFlowAnalysis';
import MarketSentiment from './components/MarketSentiment';
import type { SentimentData } from './components/MarketSentiment';
import FinancialRadar from './components/FinancialRadar';
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
  [key: string]: string | number | boolean | undefined;
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
  amount: number;
  profit?: number;
  cumulative_yield?: number;
}

interface BacktestResult {
  code: string;
  total_yield: number;
  trades: BacktestTrade[];
  yield_curve: { date: string; yield: number }[];
  max_drawdown: number;
  buy_count: number;
  sell_count: number;
  trade_count: number;
  win_rate: number;
}

interface SearchRecord {
  code: string;
  name?: string;
}

interface ExtendedStockData extends StockData {
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  macd_hist?: number;
  [key: string]: string | number | boolean | undefined;
}

interface SectorComparisonItem {
  code: string;
  name: string;
  stats: {
    profit_ratio: number;
    avg_cost: number;
    conc_90: {
      concentration: number;
    };
  };
}

interface SectorComparison {
  comparison: SectorComparisonItem[];
  industry?: string;
}

interface StockFlowItem {
  日期: string;
  主力净流入: number;
  主力净流入占比: number;
}

interface SectorMoneyFlow {
  stock_flow: StockFlowItem[];
  sector_flow?: {
    名称: string;
    今日净额: number;
    今日涨跌幅: number;
    今日主力净占比: number;
  };
}

interface SectorRotationItem {
  "板块名称": string;
  "涨跌幅": number;
  "主力净额": number;
}

interface Diagnosis {
  risk_level: string;
  diagnosis: string[];
  suggestions: string;
  stats: {
    profit_ratio: number;
    conc_90: {
      concentration: number;
    };
    avg_cost: number;
    asr: number;
  };
}

interface FinancialRadarData {
  score: number;
  data: {
    subject: string;
    value: number;
    fullMark: number;
    original: string;
  }[];
  period: string;
}

const App: React.FC = () => {
  const [stockCode, setStockCode] = useState<string>('sh.600000');
  const [dataSource, setDataSource] = useState<string>('baostock');
  const [loading, setLoading] = useState<boolean>(false);
  const [suggestions, setSuggestions] = useState<{value: string, label: string}[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState<number>(-1);
  const suggestionRef = React.useRef<HTMLDivElement>(null);
  const [data, setData] = useState<StockData[]>([]);
  const [summaryStats, setSummaryStats] = useState<SummaryStats | null>(null);

  const [distribution, setDistribution] = useState<ChipDistribution[]>([]);
  const [allDistributions, setAllDistributions] = useState<Record<string, ChipDistribution[]>>({});
  const [allSummaryStats, setAllSummaryStats] = useState<Record<string, SummaryStats>>({});
  const [hoveredDate, setHoveredDate] = useState<string | null>(null);
  const [lockedDates, setLockedDates] = useState<string[]>([]);

  const handleChartClick = useCallback((date: string | null) => {
    if (!date) return;
    
    setLockedDates(prev => {
      // 如果点击的日期已经在锁定列表中，则移除它（解锁）
      if (prev.includes(date)) {
        const next = prev.filter(d => d !== date);
        toast.info(`已取消锁定日期: ${date}`);
        return next;
      } 
      
      // 如果是新日期
      if (prev.length >= 2) {
        // 最多锁定两个，采用先进先出策略，替换最早锁定的日期
        const next = [prev[1], date];
        toast.success(`已更新锁定日期: ${date}`);
        return next;
      }
      
      const next = [...prev, date];
      toast.success(`已锁定日期: ${date}`);
      return next;
    });
  }, []);

  const toggleFullscreen = (chartId: string) => {
    setFullscreenChart(prev => prev === chartId ? null : chartId);
  };

  const [searchHistory, setSearchHistory] = useState<SearchRecord[]>(() => {
    const saved = localStorage.getItem('stock_search_history');
    if (!saved) return [];
    try {
      const parsed = JSON.parse(saved);
      // 兼容旧格式 (string[])
      return parsed.map((item: string | SearchRecord) => 
        typeof item === 'string' ? { code: item } : item
      );
    } catch {
      return [];
    }
  });

  const [favorites, setFavorites] = useState<FavoriteStock[]>(() => {
    const saved = localStorage.getItem('favorite_stocks');
    return saved ? JSON.parse(saved) : [];
  });

  const isFavorite = useMemo(() => {
    return favorites.some(f => f.code === stockCode);
  }, [favorites, stockCode]);

  const toggleFavorite = () => {
    if (isFavorite) {
      const next = favorites.filter(f => f.code !== stockCode);
      setFavorites(next);
      localStorage.setItem('favorite_stocks', JSON.stringify(next));
      toast.info(`已从自选股移除 ${stockCode}`);
    } else {
      const info = fundamentals?.info || (fundamentals as unknown as Record<string, string | number>);
      const stockName = info?.['股票简称'] as string || stockCode;
      const newFav = {
        code: stockCode,
        name: stockName,
        addedAt: new Date().toISOString(),
      };
      const next = [...favorites, newFav];
      setFavorites(next);
      localStorage.setItem('favorite_stocks', JSON.stringify(next));
      toast.success(`已添加 ${stockName} (${stockCode}) 到自选股`);
    }
  };

  const addToHistory = (code: string, name?: string) => {
    setSearchHistory(prev => {
      const next = [{ code, name }, ...prev.filter(item => item.code !== code)].slice(0, 10);
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
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('analysis');

  const [sectorComparison, setSectorComparison] = useState<SectorComparison | null>(null);
  const [sectorMoneyFlow, setSectorMoneyFlow] = useState<SectorMoneyFlow | null>(null);
  const [sectorRotation, setSectorRotation] = useState<SectorRotationItem[]>([]);
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null);
  const [extraLoading, setExtraLoading] = useState<boolean>(false);

  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [financialRadarData, setFinancialRadarData] = useState<FinancialRadarData | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState<boolean>(false);
  const [radarLoading, setRadarLoading] = useState<boolean>(false);

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

  useEffect(() => {
    const timer = setTimeout(async () => {
      if (stockCode.length >= 2 && showSuggestions) {
        try {
          const res = await axios.get(`http://localhost:8001/api/search?q=${stockCode}`);
          setSuggestions(res.data);
          setActiveSuggestionIndex(-1);
        } catch (err) {
          console.error('Search error:', err);
        }
      } else {
        setSuggestions([]);
        setActiveSuggestionIndex(-1);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [stockCode, showSuggestions]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (suggestionRef.current && !suggestionRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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

    console.log('Fetching data for:', codeToUse);
    setLoading(true);
    setError(null);
    try {
      // 分开请求以提高鲁棒性
      const historyPromise = axios.get<ApiResponse>(`http://localhost:8001/api/stock/${codeToUse}`, {
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
      });

      const fundamentalsPromise = axios.get<StockFundamentals>(`http://localhost:8001/api/stock/${codeToUse}/fundamentals`);

      const [historyRes, fundamentalsRes] = await Promise.all([
        historyPromise,
        fundamentalsPromise.catch(err => {
            console.error('Fundamentals fetch failed:', err);
            return { 
              data: { 
                info: {}, 
                groups: {}, 
                important_keys: [], 
                error: '无法加载基本面数据: ' + err.message 
              } 
            } as unknown as AxiosResponse<StockFundamentals>;
          })
      ]);

      console.log('History data received:', historyRes.data.history?.length);
      console.log('Fundamentals response:', fundamentalsRes);

      setData(historyRes.data.history || []);
      setDistribution(historyRes.data.distribution || []);
      setAllDistributions(historyRes.data.all_distributions || {});
      setAllSummaryStats(historyRes.data.all_summary_stats || {});
      setSummaryStats(historyRes.data.summary_stats || null);
      
      const fundamentalsData = fundamentalsRes.data;
      setFundamentals(fundamentalsData);
      setHoveredDate(null);
      
      // 添加到搜索历史，包含股票名称
      const info = fundamentalsData?.info || (fundamentalsData as unknown as Record<string, string | number>);
      const stockName = info?.['股票简称'] as string | undefined;
      addToHistory(codeToUse, stockName);
      
      // 异步获取额外分析数据
      fetchExtraData(codeToUse);
      
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

  const fetchExtraData = async (code: string) => {
    setExtraLoading(true);
    setSentimentLoading(true);
    setRadarLoading(true);
    try {
      const [compRes, flowRes, rotRes, diagRes, sentimentRes, radarRes] = await Promise.all([
        axios.get(`http://localhost:8001/api/stock/${code}/sector/comparison`),
        axios.get(`http://localhost:8001/api/stock/${code}/sector/money-flow`),
        axios.get(`http://localhost:8001/api/sector/rotation`),
        axios.get(`http://localhost:8001/api/stock/${code}/diagnosis`),
        axios.get(`http://localhost:8001/api/market/sentiment`),
        axios.get(`http://localhost:8001/api/stock/${code}/financial-radar`)
      ]);
      setSectorComparison(compRes.data);
      setSectorMoneyFlow(flowRes.data);
      setSectorRotation(rotRes.data);
      setDiagnosis(diagRes.data);
      setSentimentData(sentimentRes.data);
      setFinancialRadarData(radarRes.data);
    } catch (err) {
      console.error('Failed to fetch extra data:', err);
    } finally {
      setExtraLoading(false);
      setSentimentLoading(false);
      setRadarLoading(false);
    }
  };

  const renderSectorAnalysis = () => {
    if (!sectorComparison && !sectorMoneyFlow && !sectorRotation.length) {
      return (
        <div className="p-8 text-center text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p>暂无板块分析数据，请先搜索股票并点击分析</p>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 同板块筹码对比 */}
          <Card className="border-none shadow-sm">
            <CardHeader className="pb-3 border-b border-slate-50">
              <div className="flex items-center gap-2">
                <LayoutDashboard className="h-5 w-5 text-primary" />
                <CardTitle className="text-base font-semibold">同板块股票筹码对比 ({sectorComparison?.industry})</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>股票</TableHead>
                    <TableHead className="text-right">获利比例</TableHead>
                    <TableHead className="text-right">平均成本</TableHead>
                    <TableHead className="text-right">集中度(90)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sectorComparison?.comparison?.map((item: SectorComparisonItem) => (
                    <TableRow key={item.code} className="cursor-pointer hover:bg-slate-50" onClick={() => {
                      setStockCode(item.code);
                      fetchData(item.code);
                    }}>
                      <TableCell>
                        <div className="font-medium">{item.name}</div>
                        <div className="text-xs text-slate-400">{item.code}</div>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant={item.stats.profit_ratio > 50 ? "destructive" : "secondary"}>
                          {item.stats.profit_ratio.toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right font-mono">¥{item.stats.avg_cost.toFixed(2)}</TableCell>
                      <TableCell className="text-right">{item.stats.conc_90.concentration.toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* 资金流向分析 */}
          <Card className="border-none shadow-sm">
            <CardHeader className="pb-3 border-b border-slate-50">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-500" />
                <CardTitle className="text-base font-semibold">板块与个股资金流向</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              {sectorMoneyFlow?.sector_flow && (
                <div className="mb-6 p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <div className="text-sm font-medium text-slate-500 mb-3">板块资金流向: {sectorMoneyFlow.sector_flow.名称}</div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-xs text-slate-400 mb-1">今日净流入</div>
                      <div className={`text-sm font-bold ${sectorMoneyFlow.sector_flow['今日净额'] > 0 ? 'text-red-500' : 'text-green-500'}`}>
                        {(sectorMoneyFlow.sector_flow['今日净额'] / 10000).toFixed(2)}万
                      </div>
                    </div>
                    <div className="text-center border-x border-slate-200">
                      <div className="text-xs text-slate-400 mb-1">今日涨跌幅</div>
                      <div className={`text-sm font-bold ${sectorMoneyFlow.sector_flow['今日涨跌幅'] > 0 ? 'text-red-500' : 'text-green-500'}`}>
                        {sectorMoneyFlow.sector_flow['今日涨跌幅']}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-slate-400 mb-1">主力净占比</div>
                      <div className="text-sm font-bold text-slate-700">
                        {sectorMoneyFlow.sector_flow['今日主力净占比']}%
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div className="text-sm font-medium text-slate-500 mb-2">个股近5日资金趋势:</div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>日期</TableHead>
                    <TableHead className="text-right">主力净流入</TableHead>
                    <TableHead className="text-right">主力占比</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sectorMoneyFlow?.stock_flow?.map((item: StockFlowItem, idx: number) => (
                    <TableRow key={idx}>
                      <TableCell className="text-xs">{item.日期}</TableCell>
                      <TableCell className={`text-right text-xs ${item.主力净流入 > 0 ? 'text-red-500' : 'text-green-500'}`}>
                        {(item.主力净流入 / 10000).toFixed(2)}万
                      </TableCell>
                      <TableCell className="text-right text-xs">{item.主力净流入占比}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        {/* 板块轮动监控 */}
        <Card className="border-none shadow-sm">
          <CardHeader className="pb-3 border-b border-slate-50">
            <div className="flex items-center gap-2">
              <RotateCcw className="h-5 w-5 text-orange-500" />
              <CardTitle className="text-base font-semibold">行业板块轮动监控 (今日涨幅榜)</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {Array.isArray(sectorRotation) && sectorRotation?.map((sector: SectorRotationItem, idx: number) => (
                <div key={idx} className="p-3 rounded-lg bg-slate-50 border border-slate-100 hover:border-primary/30 transition-colors">
                  <div className="text-xs font-bold text-slate-700 truncate mb-1">{sector.板块名称}</div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-400">涨跌幅</span>
                    <span className={`text-xs font-bold ${sector.涨跌幅 > 0 ? 'text-red-500' : 'text-green-500'}`}>
                      {sector.涨跌幅}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] text-slate-400">主力净入</span>
                    <span className="text-[10px] font-medium">{((sector.主力净额 || 0) / 100000000).toFixed(2)}亿</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderSmartDiagnosis = () => {
    if (!diagnosis) {
      return (
        <div className="p-8 text-center text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">
          <Zap className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p>暂无智能诊断报告，请先搜索股票并点击分析</p>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* AI 核心诊断 */}
          <Card className="lg:col-span-2 border-none shadow-sm overflow-hidden">
            <div className={`h-1.5 w-full ${diagnosis.risk_level === '高' ? 'bg-red-500' : diagnosis.risk_level === '低' ? 'bg-green-500' : 'bg-orange-400'}`} />
            <CardHeader className="pb-3 border-b border-slate-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <CardTitle className="text-base font-semibold">AI 筹码智能诊断</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">风险评级:</span>
                  <Badge className={diagnosis.risk_level === '高' ? 'bg-red-500' : diagnosis.risk_level === '低' ? 'bg-green-500' : 'bg-orange-400'}>
                    {diagnosis.risk_level}风险
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-4">
                {diagnosis.diagnosis.map((text: string, idx: number) => (
                  <div key={idx} className="flex gap-3 items-start p-3 rounded-lg bg-slate-50 border-l-4 border-primary/20">
                    <div className="mt-0.5 h-4 w-4 rounded-full bg-primary/10 flex items-center justify-center text-[10px] font-bold text-primary shrink-0">
                      {idx + 1}
                    </div>
                    <p className="text-sm text-slate-700 leading-relaxed">{text}</p>
                  </div>
                ))}
              </div>
              
              <div className="mt-8 p-4 rounded-xl bg-primary/5 border border-primary/10">
                <div className="flex items-center gap-2 mb-2 text-primary">
                  <ShieldCheck className="h-4 w-4" />
                  <span className="text-sm font-bold">操盘建议 (仅供参考)</span>
                </div>
                <p className="text-sm font-medium text-slate-900 leading-relaxed">
                  {diagnosis.suggestions}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* 筹码健康度 */}
          <div className="space-y-6">
            <FinancialRadar 
              score={financialRadarData?.score || 0}
              data={financialRadarData?.data || []}
              period={financialRadarData?.period || ''}
              loading={radarLoading}
            />

            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-blue-500" />
                  <CardTitle className="text-base font-semibold">筹码健康度指标</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="pt-6 space-y-6">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-500">获利占比健康度</span>
                  <span className="font-bold">{diagnosis.stats.profit_ratio.toFixed(1)}%</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${diagnosis.stats.profit_ratio > 80 ? 'bg-red-500' : diagnosis.stats.profit_ratio < 20 ? 'bg-green-500' : 'bg-blue-500'}`} 
                    style={{ width: `${diagnosis.stats.profit_ratio}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-500">筹码集中度健康度</span>
                  <span className="font-bold">{(100 - diagnosis.stats.conc_90.concentration).toFixed(1)}%</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500 rounded-full" 
                    style={{ width: `${100 - diagnosis.stats.conc_90.concentration}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-500">平均持仓盈亏</span>
                  <span className={`font-bold ${((parseFloat(data[data.length-1]?.close || '0') / diagnosis.stats.avg_cost - 1) * 100) > 0 ? 'text-red-500' : 'text-green-600'}`}>
                    {((parseFloat(data[data.length-1]?.close || '0') / diagnosis.stats.avg_cost - 1) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="p-3 rounded-lg bg-slate-50 text-center">
                    <div className="text-[10px] text-slate-400 mb-1">价格/成本比</div>
                    <div className="text-sm font-bold">{(parseFloat(data[data.length-1]?.close || '0') / diagnosis.stats.avg_cost).toFixed(2)}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50 text-center">
                    <div className="text-[10px] text-slate-400 mb-1">ASR值</div>
                    <div className="text-sm font-bold">{diagnosis.stats.asr.toFixed(1)}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
    );
  };

  const activeDate = hoveredDate || (lockedDates.length > 0 ? lockedDates[lockedDates.length - 1] : null);

  const currentDistribution = useMemo(() => {
    if (activeDate) {
      // 如果当前日期有数据，直接返回
      if (allDistributions[activeDate]) return allDistributions[activeDate];
      
      // 如果当前日期没数据，寻找最近的有数据的日期（向前查找）
      const dates = data.map(d => d.date);
      const currentIndex = dates.indexOf(activeDate);
      if (currentIndex !== -1) {
        for (let i = currentIndex; i >= 0; i--) {
          const d = dates[i];
          if (allDistributions[d]) return allDistributions[d];
        }
      }
    }
    
    if (!activeDate && data.length > 0) {
      return allDistributions[data[data.length - 1].date] || [];
    }
    return distribution; // Fallback to initial distribution
  }, [activeDate, allDistributions, data, distribution]);

  const currentSummaryStats = useMemo(() => {
    if (activeDate) {
      if (allSummaryStats[activeDate]) return allSummaryStats[activeDate];
      
      const dates = data.map(d => d.date);
      const currentIndex = dates.indexOf(activeDate);
      if (currentIndex !== -1) {
        for (let i = currentIndex; i >= 0; i--) {
          const d = dates[i];
          if (allSummaryStats[d]) return allSummaryStats[d];
        }
      }
    }

    if (!activeDate && data.length > 0) {
      return allSummaryStats[data[data.length - 1].date] || null;
    }
    return summaryStats; // Fallback to initial stats
  }, [activeDate, allSummaryStats, data, summaryStats]);

  const currentClose = useMemo(() => {
    const dates = data.map(d => d.date);
    let targetDate = activeDate;
    
    // 如果 activeDate 没数据，找到最近的有筹码数据的日期
    if (activeDate && !allDistributions[activeDate]) {
      const currentIndex = dates.indexOf(activeDate);
      if (currentIndex !== -1) {
        for (let i = currentIndex; i >= 0; i--) {
          if (allDistributions[dates[i]]) {
            targetDate = dates[i];
            break;
          }
        }
      }
    }

    const d = targetDate 
      ? data.find(item => item.date === targetDate)
      : (data.length > 0 ? data[data.length - 1] : null);
    return d ? parseFloat(d.close) : 0;
  }, [activeDate, data, allDistributions]);

  const currentProfitRatio = useMemo(() => {
    const dates = data.map(d => d.date);
    let targetDate = activeDate;
    
    if (activeDate && !allDistributions[activeDate]) {
      const currentIndex = dates.indexOf(activeDate);
      if (currentIndex !== -1) {
        for (let i = currentIndex; i >= 0; i--) {
          if (allDistributions[dates[i]]) {
            targetDate = dates[i];
            break;
          }
        }
      }
    }

    const d = targetDate
      ? data.find(item => item.date === targetDate)
      : (data.length > 0 ? data[data.length - 1] : null);
    return d?.profit_ratio || 0;
  }, [activeDate, data, allDistributions]);

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

    // 使用后端计算的指标
    const rsi = (d as ExtendedStockData).rsi;
    const macd = (d as ExtendedStockData).macd;
    const macd_signal = (d as ExtendedStockData).macd_signal;
    const macd_hist = (d as ExtendedStockData).macd_hist;

    return { 
      ...d, 
      open_num,
      high_num,
      low_num,
      close_num,
      volume_num,
      isUp: close_num >= open_num,
      rsi,
      macd,
      macd_signal,
      macd_hist,
      ...mas,
    };
  });


  const renderFundamentals = () => {
    if (!fundamentals) {
      return (
        <div className="mb-6 p-4 bg-slate-50 rounded-lg border border-dashed border-slate-200 text-center text-slate-400 text-sm">
          正在加载或暂无基本面数据...
        </div>
      );
    }

    // 处理错误情况
    if (fundamentals.error) {
      return (
        <Alert variant="destructive" className="mb-6">
          <Info className="h-4 w-4" />
          <AlertTitle>基本面数据获取失败</AlertTitle>
          <AlertDescription>{fundamentals.error}</AlertDescription>
        </Alert>
      );
    }

    // 兼容逻辑：后端可能直接返回 info 内容，也可能返回包含 info 的对象
    const info = fundamentals.info || (fundamentals as unknown as Record<string, string | number>);
    
    // 检查是否有实际内容 (排除掉包装字段)
    const infoKeys = Object.keys(info).filter(k => !['info', 'groups', 'important_keys', 'error'].includes(k));
    const hasValidInfo = infoKeys.length > 0 && infoKeys.some(k => 
      info[k] !== undefined && 
      info[k] !== null &&
      info[k] !== ''
    );

    if (!hasValidInfo) {
      return (
        <div className="mb-6 p-4 bg-slate-50 rounded-lg border border-dashed border-slate-200 text-center text-slate-400 text-sm">
          未查询到该股票的详细基本面信息
        </div>
      );
    }

    const groups = fundamentals.groups || {
      "基本信息": infoKeys
    };

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
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="h-9 gap-2 border-slate-200 hover:bg-slate-50 transition-all active:scale-95">
                  <Star className="h-4 w-4 text-amber-500 fill-amber-500" />
                  <span className="font-medium">我的自选</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[400px] sm:w-[540px] p-0 overflow-hidden flex flex-col border-l-0 shadow-2xl">
                <SheetHeader className="p-6 border-b bg-slate-50/50">
                  <SheetTitle className="flex items-center gap-2 text-xl">
                    <Star className="h-5 w-5 text-amber-500 fill-amber-500" />
                    我的自选股
                  </SheetTitle>
                </SheetHeader>
                <div className="flex-1 overflow-hidden p-6">
                  <FavoriteStocks 
                    favorites={favorites}
                    setFavorites={setFavorites}
                    onSelectStock={(code) => {
                      setStockCode(code);
                      fetchData(code);
                    }} 
                  />
                </div>
              </SheetContent>
            </Sheet>
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
                  <div className="relative group" ref={suggestionRef}>
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                    <Input 
                      value={stockCode}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        setStockCode(e.target.value.toUpperCase());
                        setShowSuggestions(true);
                      }}
                      onFocus={() => setShowSuggestions(true)}
                      placeholder={dataSource === 'baostock' ? "sh.600000" : "600000"}
                      className="h-10 pl-10 pr-10 bg-slate-50/50 border-slate-200 transition-all focus:ring-2 focus:ring-primary/20 hover:border-primary/30"
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === 'Enter') {
                          if (activeSuggestionIndex >= 0 && suggestions[activeSuggestionIndex]) {
                            const selected = suggestions[activeSuggestionIndex];
                            setStockCode(selected.value);
                            setShowSuggestions(false);
                            fetchData(selected.value);
                          } else {
                            fetchData();
                            setShowSuggestions(false);
                          }
                        } else if (e.key === 'ArrowDown') {
                          e.preventDefault();
                          setActiveSuggestionIndex(prev => {
                            const next = prev < suggestions.length - 1 ? prev + 1 : prev;
                            // 确保选中的元素在视图中
                            const element = document.getElementById(`suggestion-${next}`);
                            element?.scrollIntoView({ block: 'nearest' });
                            return next;
                          });
                        } else if (e.key === 'ArrowUp') {
                          e.preventDefault();
                          setActiveSuggestionIndex(prev => {
                            const next = prev > 0 ? prev - 1 : -1;
                            // 确保选中的元素在视图中
                            const element = document.getElementById(`suggestion-${next}`);
                            element?.scrollIntoView({ block: 'nearest' });
                            return next;
                          });
                        } else if (e.key === 'Escape') {
                          setShowSuggestions(false);
                        }
                      }}
                    />
                    {showSuggestions && suggestions.length > 0 && (
                      <div className="absolute z-50 w-full mt-1 bg-white border border-slate-200 rounded-md shadow-lg max-h-60 overflow-y-auto overflow-x-hidden">
                        {suggestions.map((s, index) => (
                          <div
                            key={s.value}
                            id={`suggestion-${index}`}
                            className={`px-4 py-2 cursor-pointer text-sm flex justify-between items-center transition-colors ${
                              index === activeSuggestionIndex ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-slate-50 text-slate-700'
                            }`}
                            onClick={() => {
                              setStockCode(s.value);
                              setShowSuggestions(false);
                              fetchData(s.value);
                            }}
                            onMouseEnter={() => setActiveSuggestionIndex(index)}
                          >
                            <div className="flex items-center gap-2 overflow-hidden">
                              <span className="truncate">{s.label}</span>
                            </div>
                            <span className="text-[10px] opacity-40 font-mono ml-4 flex-shrink-0">{s.value}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {searchHistory.length > 0 && (
                       <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                         <History className="h-4 w-4 text-slate-300" />
                       </div>
                     )}
                   </div>
                   {searchHistory.length > 0 && (
                     <div className="flex flex-wrap gap-2 mt-2">
                       {searchHistory.map(item => (
                          <button
                            key={item.code}
                            onClick={() => {
                              setStockCode(item.code);
                              fetchData(item.code);
                            }}
                            className="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-500 hover:bg-primary/10 hover:text-primary transition-colors border border-slate-200 flex items-center gap-1"
                          >
                            <span>{item.code}</span>
                            {item.name && <span className="text-slate-400 font-normal">| {item.name}</span>}
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
                  onClick={toggleFavorite} 
                  className="h-11 px-6 rounded-xl border-slate-200 hover:bg-slate-50 transition-all active:scale-95"
                >
                  <Star className={`mr-2 h-4 w-4 ${isFavorite ? 'fill-amber-500 text-amber-500' : 'text-slate-500'}`} />
                  {isFavorite ? '移除自选' : '加入自选'}
                </Button>
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
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-white/50 border border-slate-100 p-1">
            <TabsTrigger value="analysis" className="data-[state=active]:bg-primary data-[state=active]:text-white">
              <LayoutDashboard className="h-4 w-4 mr-2" />
              筹码分析
            </TabsTrigger>
            <TabsTrigger value="sector" className="data-[state=active]:bg-primary data-[state=active]:text-white">
              <BarChart3 className="h-4 w-4 mr-2" />
              板块分析
            </TabsTrigger>
            <TabsTrigger value="diagnosis" className="data-[state=active]:bg-primary data-[state=active]:text-white">
              <Zap className="h-4 w-4 mr-2" />
              智能诊断
            </TabsTrigger>
          </TabsList>

          <TabsContent value="analysis" className="mt-0 space-y-6">
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
          <div className="xl:col-span-8 space-y-6">
            <Card className={`border-none shadow-sm overflow-hidden ${fullscreenChart === 'trend' ? 'fixed inset-0 z-[100] m-0 rounded-none' : ''}`}>
              <CardHeader className="pb-2 border-b border-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <LayoutDashboard className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base font-semibold">趋势走势分析</CardTitle>
                  </div>
                  <div className="flex items-center gap-3">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-8 w-8 p-0 text-slate-500 hover:text-primary"
                      onClick={() => toggleFullscreen('trend')}
                    >
                      {fullscreenChart === 'trend' ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </Button>
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
              <CardContent className={`p-6 ${fullscreenChart === 'trend' ? 'h-[calc(100vh-64px)] overflow-hidden' : ''}`}>
                <div className={fullscreenChart === 'trend' ? 'h-full w-full' : 'h-[450px] w-full'}>
                  <ProfitRatioChart 
                    data={chartData} 
                    trades={backtestResult?.trades}
                    hoveredDate={hoveredDate}
                    lockedDates={lockedDates}
                    onHover={setHoveredDate}
                    onClick={handleChartClick}
                    onDoubleClick={() => setLockedDates([])}
                    isLocked={lockedDates.length >= 2}
                    maSettings={settings.maSettings}
                    showIndicators={settings.showIndicators}
                    showCloseLine={settings.showCloseLine}
                    height={fullscreenChart === 'trend' ? undefined : 450}
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
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">买入 / 卖出</p>
                          <p className="text-2xl font-bold text-slate-900">
                            <span className="text-red-600">{backtestResult.buy_count}</span>
                            <span className="mx-2 text-slate-300">/</span>
                            <span className="text-green-600">{backtestResult.sell_count}</span>
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="bg-slate-50 border-slate-100 shadow-none">
                        <CardContent className="pt-6">
                          <p className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider">胜率</p>
                          <p className="text-2xl font-bold text-slate-900">{backtestResult.win_rate}%</p>
                        </CardContent>
                      </Card>
                    </div>

                    <Card className={`bg-slate-50 border-slate-100 shadow-none ${fullscreenChart === 'yield' ? 'fixed inset-0 z-[100] m-0 rounded-none bg-white' : ''}`}>
                      <CardHeader className="pb-2 flex flex-row items-center justify-between">
                        <CardTitle className="text-sm font-semibold text-slate-600">收益率走势</CardTitle>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-8 w-8 p-0 text-slate-500 hover:text-primary"
                          onClick={() => toggleFullscreen('yield')}
                        >
                          {fullscreenChart === 'yield' ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                        </Button>
                      </CardHeader>
                      <CardContent className={fullscreenChart === 'yield' ? 'h-[calc(100vh-64px)] overflow-hidden' : ''}>
                        <div className={fullscreenChart === 'yield' ? 'h-full w-full' : 'h-[200px] w-full'}>
                          <BacktestYieldChart 
                            data={backtestResult.yield_curve} 
                            height={fullscreenChart === 'yield' ? undefined : 200}
                          />
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-slate-50 border-slate-100 shadow-none">
                      <CardHeader className="pb-2 flex flex-row items-center justify-between">
                        <CardTitle className="text-sm font-semibold text-slate-600">交易明细</CardTitle>
                        <Badge variant="outline" className="text-[10px]">{backtestResult.trades.length} 条记录</Badge>
                      </CardHeader>
                      <CardContent className="p-0">
                        <div className="max-h-[400px] overflow-y-auto">
                          <Table>
                            <TableHeader className="bg-slate-100/50 sticky top-0 z-10 backdrop-blur-sm">
                              <TableRow>
                                <TableHead className="w-[80px]">类型</TableHead>
                                <TableHead>日期</TableHead>
                                <TableHead>价格</TableHead>
                                <TableHead>获利比例</TableHead>
                                <TableHead className="text-right">金额 / 收益</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {backtestResult.trades.map((trade: BacktestTrade, idx: number) => (
                                <TableRow key={idx} className="hover:bg-white/50 transition-colors">
                                  <TableCell>
                                    <Badge 
                                      variant={trade.type === 'buy' ? 'default' : 'secondary'} 
                                      className={trade.type === 'buy' 
                                        ? 'bg-red-500 hover:bg-red-600 border-none' 
                                        : 'bg-green-500 hover:bg-green-600 text-white border-none'}
                                    >
                                      {trade.type === 'buy' ? '买入' : '卖出'}
                                    </Badge>
                                  </TableCell>
                                  <TableCell className="font-medium text-slate-600">{trade.date}</TableCell>
                                  <TableCell className="font-mono text-slate-700">{trade.price.toFixed(2)}</TableCell>
                                  <TableCell className="text-slate-600">{trade.ratio.toFixed(2)}%</TableCell>
                                  <TableCell className="text-right">
                                    <div className="space-y-0.5">
                                      <p className="text-xs font-medium text-slate-700">
                                        ¥{trade.amount.toLocaleString()}
                                      </p>
                                      {trade.profit !== undefined && (
                                        <p className={`text-[10px] font-bold ${trade.profit >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                                          {trade.profit > 0 ? '+' : ''}{trade.profit}%
                                          {trade.type === 'sell' && trade.cumulative_yield !== undefined && (
                                            <span className="ml-1 text-slate-400 font-normal">
                                              (累计: {trade.cumulative_yield}%)
                                            </span>
                                          )}
                                        </p>
                                      )}
                                    </div>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </CardContent>
                    </Card>
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
          </div>

          <div className="xl:col-span-4 space-y-6">
            <Card className={`border-none shadow-sm ${fullscreenChart === 'chips' ? 'fixed inset-0 z-[100] m-0 rounded-none bg-white' : ''}`}>
              <CardHeader className="pb-3 border-b border-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base font-semibold">筹码分布对比</CardTitle>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-8 w-8 p-0 text-slate-500 hover:text-primary"
                      onClick={() => toggleFullscreen('chips')}
                    >
                      {fullscreenChart === 'chips' ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </Button>
                    {lockedDates.length === 2 && (
                      <span className="text-[10px] font-bold text-blue-500 bg-blue-50 px-2 py-0.5 rounded uppercase">对比中</span>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className={`pt-6 ${fullscreenChart === 'chips' ? 'h-[calc(100vh-64px)] overflow-hidden' : ''}`}>
                <div className={fullscreenChart === 'chips' ? 'h-full w-full' : (lockedDates.length === 2 ? "grid grid-cols-2 gap-2 h-[500px]" : "h-[500px]")}>
                  {lockedDates.length === 2 ? (
                    <>
                      <div className="flex flex-col">
                        <div className="text-[10px] font-medium text-slate-400 mb-2 text-center">日期: {lockedDates[0]}</div>
                        <ChipDistributionChart 
                          key={`locked-0-${lockedDates[0]}`}
                          data={allDistributions[lockedDates[0]] || []} 
                          currentClose={parseFloat(data.find(d => d.date === lockedDates[0])?.close || '0')}
                          summaryStats={allSummaryStats[lockedDates[0]]}
                          profitRatio={data.find(d => d.date === lockedDates[0])?.profit_ratio}
                          height={fullscreenChart === 'chips' ? undefined : 400}
                        />
                      </div>
                      <div className="flex flex-col border-l border-slate-100 pl-2">
                        <div className="text-[10px] font-medium text-slate-400 mb-2 text-center">日期: {lockedDates[1]}</div>
                        <ChipDistributionChart 
                          key={`locked-1-${lockedDates[1]}`}
                          data={allDistributions[lockedDates[1]] || []} 
                          currentClose={parseFloat(data.find(d => d.date === lockedDates[1])?.close || '0')}
                          summaryStats={allSummaryStats[lockedDates[1]]}
                          profitRatio={data.find(d => d.date === lockedDates[1])?.profit_ratio}
                          height={fullscreenChart === 'chips' ? undefined : 400}
                        />
                      </div>
                    </>
                  ) : (
                    <ChipDistributionChart 
                      data={currentDistribution} 
                      currentClose={currentClose}
                      summaryStats={currentSummaryStats}
                      profitRatio={currentProfitRatio}
                      height={fullscreenChart === 'chips' ? undefined : 400}
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
                {currentSummaryStats ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50">
                      <span className="text-sm text-slate-500">平均成本</span>
                      <span className="font-bold text-slate-900">{currentSummaryStats.avg_cost.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50">
                      <span className="text-sm text-slate-500">获利比例</span>
                      <span className={`font-bold ${currentSummaryStats.profit_ratio > 50 ? 'text-red-600' : 'text-green-600'}`}>
                        {currentSummaryStats.profit_ratio.toFixed(2)}%
                      </span>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400 uppercase">90% 筹码集中度</span>
                        <span className="text-xs font-bold text-slate-600">{currentSummaryStats.conc_90.concentration.toFixed(2)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary" 
                          style={{ width: `${Math.min(100, currentSummaryStats.conc_90.concentration)}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-500 text-right">
                        范围: {currentSummaryStats.conc_90.low.toFixed(2)} - {currentSummaryStats.conc_90.high.toFixed(2)}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400 uppercase">70% 筹码集中度</span>
                        <span className="text-xs font-bold text-slate-600">{currentSummaryStats.conc_70.concentration.toFixed(2)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary/70" 
                          style={{ width: `${Math.min(100, currentSummaryStats.conc_70.concentration)}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-500 text-right">
                        范围: {currentSummaryStats.conc_70.low.toFixed(2)} - {currentSummaryStats.conc_70.high.toFixed(2)}
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

            {/* 筹码峰值追踪 */}
            {data.length > 0 && (
              <ChipPeakAnalysis 
                data={data}
                allSummaryStats={allSummaryStats}
              />
            )}

            {/* 资金流向分析 */}
            {stockCode && data.length > 0 && (
              <MoneyFlowAnalysis stockCode={stockCode} />
            )}

            {/* 大盘情绪指数 */}
            <MarketSentiment data={sentimentData} loading={sentimentLoading} />
          </div>
        </div>
        </TabsContent>

        <TabsContent value="sector" className="mt-0">
          {extraLoading ? (
            <div className="flex flex-col items-center justify-center p-20 bg-white rounded-xl shadow-sm border border-slate-100">
              <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
              <p className="text-slate-500">正在进行板块深度分析...</p>
            </div>
          ) : renderSectorAnalysis()}
        </TabsContent>

        <TabsContent value="diagnosis" className="mt-0">
          {extraLoading ? (
            <div className="flex flex-col items-center justify-center p-20 bg-white rounded-xl shadow-sm border border-slate-100">
              <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
              <p className="text-slate-500">AI 正在生成智能诊断报告...</p>
            </div>
          ) : renderSmartDiagnosis()}
        </TabsContent>
      </Tabs>
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