import React from 'react';

export interface SentimentData {
  index: number;
  sentiment: string;
  description: string;
  timestamp: string;
}

interface MarketSentimentProps {
  data: SentimentData | null;
  loading: boolean;
}

const MarketSentiment: React.FC<MarketSentimentProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg animate-pulse">
        <span className="text-gray-400">正在计算大盘情绪...</span>
      </div>
    );
  }

  if (!data || typeof data.index !== 'number') return null;

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case '极度贪婪': return 'text-red-600 bg-red-50 border-red-200';
      case '贪婪': return 'text-orange-600 bg-orange-50 border-orange-200';
      case '极度恐惧': return 'text-green-600 bg-green-50 border-green-200';
      case '恐惧': return 'text-emerald-600 bg-emerald-50 border-emerald-200';
      default: return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  const getProgressColor = (index: number) => {
    if (index >= 75) return 'bg-red-500';
    if (index >= 60) return 'bg-orange-500';
    if (index <= 25) return 'bg-green-500';
    if (index <= 40) return 'bg-emerald-500';
    return 'bg-blue-500';
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
            📊 大盘“恐惧与贪婪”指数
            <span className={`text-xs px-2 py-0.5 rounded-full border ${getSentimentColor(data.sentiment)}`}>
              {data.sentiment}
            </span>
          </h3>
          <p className="text-sm text-gray-500 mt-1">基于上证指数全市场获利比例计算</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-black text-gray-900">{(data.index || 0).toFixed(1)}%</div>
          <div className="text-[10px] text-gray-400">更新于 {data.timestamp ? (data.timestamp.split(' ')[1] || data.timestamp) : '--:--:--'}</div>
        </div>
      </div>

      <div className="relative pt-1">
        <div className="flex mb-2 items-center justify-between">
          <div className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-green-600 bg-green-50">
            极度恐惧 (0%)
          </div>
          <div className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-red-600 bg-red-50">
            极度贪婪 (100%)
          </div>
        </div>
        <div className="overflow-hidden h-3 mb-4 text-xs flex rounded-full bg-gray-100">
          <div 
            style={{ width: `${data.index}%` }}
            className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-1000 ${getProgressColor(data.index)}`}
          ></div>
        </div>
      </div>

      <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <p className="text-sm text-gray-700 leading-relaxed italic">
          "{data.description}"
        </p>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-center">
        <div className="p-2 rounded-lg bg-red-50 border border-red-100">
          <div className="text-xs text-red-600 mb-1">风险提示</div>
          <div className="text-sm font-medium text-red-800">
            {data.index > 75 ? '高位拥挤，注意减仓' : '暂无系统性风险'}
          </div>
        </div>
        <div className="p-2 rounded-lg bg-green-50 border border-green-100">
          <div className="text-xs text-green-600 mb-1">机会提示</div>
          <div className="text-sm font-medium text-green-800">
            {data.index < 25 ? '超跌严重，分批布局' : '等待择时机会'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketSentiment;
