import React, { useState, useEffect } from 'react';
import { Star, Trash2, TrendingUp, TrendingDown, RefreshCw, Plus } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import axios from 'axios';

export interface FavoriteStock {
  code: string;
  name: string;
  addedAt: string;
  currentPrice?: number;
  changePercent?: number;
  profitRatio?: number;
  chipConcentration?: number;
}

interface FavoriteStocksProps {
  onSelectStock: (code: string) => void;
  favorites: FavoriteStock[];
  setFavorites: React.Dispatch<React.SetStateAction<FavoriteStock[]>>;
}

const FavoriteStocks: React.FC<FavoriteStocksProps> = ({ onSelectStock, favorites, setFavorites }) => {
  const [loading, setLoading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newStockCode, setNewStockCode] = useState('');
  const [newStockName, setNewStockName] = useState('');

  useEffect(() => {
    localStorage.setItem('favorite_stocks', JSON.stringify(favorites));
  }, [favorites]);

  const addFavorite = () => {
    if (!newStockCode) {
      toast.warning('请输入股票代码');
      return;
    }

    const exists = favorites.find(f => f.code === newStockCode);
    if (exists) {
      toast.warning('该股票已在自选列表中');
      return;
    }

    const newFav: FavoriteStock = {
      code: newStockCode,
      name: newStockName || newStockCode,
      addedAt: new Date().toISOString(),
    };

    setFavorites(prev => [...prev, newFav]);
    toast.success(`已添加 ${newStockCode} 到自选股`);
    setNewStockCode('');
    setNewStockName('');
    setDialogOpen(false);
  };

  const removeFavorite = (code: string) => {
    setFavorites(prev => prev.filter(f => f.code !== code));
    toast.info(`已从自选股移除 ${code}`);
  };

  const refreshData = async () => {
    if (favorites.length === 0) {
      toast.info('自选股列表为空');
      return;
    }

    setLoading(true);
    try {
      const promises = favorites.map(async (stock) => {
        try {
          // 获取基本面数据
          const fundamentalsRes = await axios.get(`http://localhost:8001/api/stock/${stock.code}/fundamentals`);
          const info = fundamentalsRes.data?.info || fundamentalsRes.data;
          
          // 获取筹码数据
          const stockRes = await axios.get(`http://localhost:8001/api/stock/${stock.code}`, {
            params: {
              source: 'baostock',
              lookback: 250,
            }
          });

          const history = stockRes.data.history || [];
          const latestData = history[history.length - 1];
          const summaryStats = stockRes.data.summary_stats;

          return {
            ...stock,
            name: info['股票简称'] || stock.name,
            currentPrice: info['最新价'] || parseFloat(latestData?.close || '0'),
            changePercent: parseFloat(String(info['涨跌幅'] || '0').replace('%', '')),
            profitRatio: latestData?.profit_ratio || 0,
            chipConcentration: summaryStats?.conc_90?.concentration || 0,
          };
        } catch (err) {
          console.error(`Error fetching data for ${stock.code}:`, err);
          return stock;
        }
      });

      const updated = await Promise.all(promises);
      setFavorites(updated);
      toast.success('自选股数据已更新');
    } catch (err) {
      toast.error('更新失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs px-2 py-0.5">{favorites.length} 个股票</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={refreshData}
            disabled={loading}
            className="h-9 px-3 text-slate-500 hover:text-primary hover:bg-primary/5 rounded-lg"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            <span className="text-xs font-medium">刷新数据</span>
          </Button>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="h-9 px-4 rounded-lg shadow-sm">
                <Plus className="h-4 w-4 mr-1.5" />
                添加
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>添加自选股</DialogTitle>
                <DialogDescription>
                  输入股票代码和名称，添加到您的个人自选列表
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">股票代码</label>
                  <Input
                    placeholder="如: sh.600000 或 600000"
                    value={newStockCode}
                    onChange={(e) => setNewStockCode(e.target.value.toUpperCase())}
                    className="bg-slate-50/50 border-slate-200"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">股票名称（可选）</label>
                  <Input
                    placeholder="如: 浦发银行"
                    value={newStockName}
                    onChange={(e) => setNewStockName(e.target.value)}
                    className="bg-slate-50/50 border-slate-200"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setDialogOpen(false)}>
                  取消
                </Button>
                <Button onClick={addFavorite}>确认添加</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto -mx-6 px-6">
        {favorites.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-slate-400 border-2 border-dashed border-slate-100 rounded-2xl">
            <Star className="h-12 w-12 mb-4 opacity-10" />
            <p className="text-sm font-medium">暂无自选股</p>
            <p className="text-xs mt-2 opacity-60">点击上方“添加”按钮开始管理</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 pb-6">
            {favorites.map((stock) => (
              <div
                key={stock.code}
                className="group relative bg-white border border-slate-100 rounded-xl p-4 hover:border-primary/30 hover:shadow-md hover:shadow-primary/5 transition-all cursor-pointer"
                onClick={() => onSelectStock(stock.code)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <h4 className="text-sm font-bold text-slate-900 truncate group-hover:text-primary transition-colors">
                        {stock.name}
                      </h4>
                      <span className="text-[10px] px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded font-mono font-medium">{stock.code}</span>
                    </div>
                    
                    <div className="flex items-center gap-4 text-xs">
                      {stock.currentPrice !== undefined && (
                        <div className="flex items-center gap-1.5">
                          <span className="text-slate-400">最新价</span>
                          <span className="font-mono font-bold text-slate-700">
                            ¥{stock.currentPrice.toFixed(2)}
                          </span>
                        </div>
                      )}
                      
                      {stock.changePercent !== undefined && (
                        <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded ${
                          stock.changePercent > 0 ? 'bg-red-50 text-red-600' : 
                          stock.changePercent < 0 ? 'bg-green-50 text-green-600' : 
                          'bg-slate-50 text-slate-600'
                        }`}>
                          {stock.changePercent > 0 ? (
                            <TrendingUp className="h-3 w-3" />
                          ) : stock.changePercent < 0 ? (
                            <TrendingDown className="h-3 w-3" />
                          ) : null}
                          <span className="font-mono font-bold">
                            {stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-slate-50">
                      {stock.profitRatio !== undefined && (
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-400 uppercase tracking-wider">获利比例</span>
                          <span className={`text-xs font-bold ${
                            stock.profitRatio > 70 ? 'text-red-600' :
                            stock.profitRatio < 30 ? 'text-green-600' :
                            'text-slate-700'
                          }`}>
                            {stock.profitRatio.toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {stock.chipConcentration !== undefined && (
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-400 uppercase tracking-wider">筹码集中度</span>
                          <span className={`text-xs font-bold ${
                            stock.chipConcentration < 10 ? 'text-blue-600' :
                            stock.chipConcentration > 20 ? 'text-amber-600' :
                            'text-slate-700'
                          }`}>
                            {stock.chipConcentration.toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-slate-300 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all rounded-lg"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFavorite(stock.code);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default FavoriteStocks;
