import React, { useState, useEffect } from 'react';
import axios from 'axios';
import dayjs from 'dayjs';
import { 
  Search, 
  TrendingUp, 
  RotateCcw, 
  Info, 
  Settings,
  Calendar as CalendarIcon,
  Loader2,
  LayoutDashboard
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

import ProfitRatioChart from './components/ProfitRatioChart';
import CoolStockTable from './components/CoolStockTable';
import ChipDistributionChart from './components/ChipDistributionChart';
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
  [key: string]: string;
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
  ]
};

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
  const [fundamentals, setFundamentals] = useState<StockFundamentals | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChipSettings>(DEFAULT_SETTINGS);
  const [dateRange, setDateRange] = useState<{ from: Date; to: Date }>({
    from: dayjs().subtract(1, 'year').toDate(),
    to: dayjs().toDate()
  });
  const [drawerVisible, setDrawerVisible] = useState<boolean>(false);

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

  const fetchData = async () => {
    if (!stockCode) {
      toast.warning('请输入股票代码');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [historyRes, fundamentalsRes] = await Promise.all([
        axios.get<ApiResponse>(`http://localhost:8000/api/stock/${stockCode}`, {
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
        axios.get<StockFundamentals>(`http://localhost:8000/api/stock/${stockCode}/fundamentals`)
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
    if (hoveredDate && allDistributions[hoveredDate]) {
      setDistribution(allDistributions[hoveredDate]);
    } else if (!hoveredDate && data.length > 0) {
      const latestDate = data[data.length - 1].date;
      if (allDistributions[latestDate]) {
        setDistribution(allDistributions[latestDate]);
      }
    }
    
    if (hoveredDate && allSummaryStats[hoveredDate]) {
      setSummaryStats(allSummaryStats[hoveredDate]);
    } else if (!hoveredDate && data.length > 0) {
      const latestDate = data[data.length - 1].date;
      if (allSummaryStats[latestDate]) {
        setSummaryStats(allSummaryStats[latestDate]);
      }
    }
  }, [hoveredDate, allDistributions, allSummaryStats, data]);

  const currentClose = hoveredDate 
    ? parseFloat(data.find(d => d.date === hoveredDate)?.close || '0')
    : (data.length > 0 ? parseFloat(data[data.length - 1].close) : 0);

  const currentProfitRatio = (hoveredDate
    ? data.find(d => d.date === hoveredDate)?.profit_ratio
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
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                    <Input 
                      value={stockCode}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setStockCode(e.target.value)}
                      placeholder={dataSource === 'baostock' ? "sh.600000" : "600000"}
                      className="h-10 pl-10 bg-slate-50/50 border-slate-200"
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && fetchData()}
                    />
                  </div>
                </div>

                <div className="space-y-2 sm:col-span-2">
                  <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">时间范围</Label>
                  <DatePickerWithRange 
                    value={dateRange}
                    onChange={(range) => range && setDateRange(range)}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Button 
                  onClick={fetchData} 
                  disabled={loading}
                  className="h-10 px-6 rounded-lg font-medium shadow-sm"
                >
                  {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                  分析数据
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
                
                <Drawer open={drawerVisible} onOpenChange={setDrawerVisible}>
                  <DrawerTrigger asChild>
                    <Button variant="secondary" className="h-10 px-4 bg-slate-100 hover:bg-slate-200 text-slate-700">
                      <Settings className="mr-2 h-4 w-4" />
                      计算配置
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
                            <div className="space-y-2 p-2 rounded-lg bg-slate-50">
                              <Label className="text-xs">衰减系数: {settings.decay.toFixed(2)}</Label>
                              <Input 
                                type="range" 
                                min="0.1" 
                                max="2.0" 
                                step="0.1" 
                                value={settings.decay}
                                onChange={(e) => setSettings({...settings, decay: parseFloat(e.target.value)})}
                                className="h-6"
                              />
                            </div>
                          </div>

                          <div className="space-y-4">
                            <h3 className="text-sm font-semibold text-slate-900 border-l-4 border-primary pl-2">显示选项</h3>
                            <div className="flex items-center justify-between space-x-2 p-2 rounded-lg bg-slate-50">
                              <Label htmlFor="showCloseLine">显示收盘价曲线</Label>
                              <Switch 
                                id="showCloseLine" 
                                checked={settings.showCloseLine}
                                onCheckedChange={(checked) => setSettings({...settings, showCloseLine: checked})}
                              />
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

        {fundamentals && Object.keys(fundamentals).length > 0 && (
          <Card className="border-none shadow-sm">
            <CardHeader className="pb-3 border-b border-slate-50">
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-primary" />
                <CardTitle className="text-base font-semibold">基本面信息</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
                {Object.entries(fundamentals).map(([key, value]) => (
                  <div key={key} className="group rounded-xl bg-slate-50/80 p-3 transition-colors hover:bg-slate-100/80">
                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">{key}</p>
                    <p className="text-sm font-semibold text-slate-700 truncate">{value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
          <div className="xl:col-span-8 space-y-6">
            <Card className="border-none shadow-sm overflow-hidden">
              <CardHeader className="pb-2 border-b border-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <LayoutDashboard className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base font-semibold">趋势分析与获利比例</CardTitle>
                  </div>
                  {hoveredDate && (
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                      <CalendarIcon className="h-4 w-4" />
                      {hoveredDate}
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-[450px] w-full">
                  <ProfitRatioChart 
                    data={chartData} 
                    hoveredDate={hoveredDate}
                    onHover={setHoveredDate}
                    onClick={() => {}}
                    isLocked={false}
                    maSettings={settings.maSettings}
                    showCloseLine={settings.showCloseLine}
                    height={450}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <CardTitle className="text-base font-semibold">历史交易明细</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <CoolStockTable data={data} onRowHover={setHoveredDate} />
              </CardContent>
            </Card>
          </div>

          <div className="xl:col-span-4 space-y-6">
            <Card className="border-none shadow-sm">
              <CardHeader className="pb-3 border-b border-slate-50">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  <CardTitle className="text-base font-semibold">当前筹码分布</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="h-[500px]">
                  <ChipDistributionChart 
                    data={distribution} 
                    currentClose={currentClose}
                    summaryStats={summaryStats}
                    profitRatio={currentProfitRatio}
                  />
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
