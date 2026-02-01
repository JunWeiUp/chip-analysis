import React from 'react';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface RadarDataItem {
  subject: string;
  value: number;
  fullMark: number;
  original: string;
}

interface FinancialRadarProps {
  score: number;
  data: RadarDataItem[];
  period: string;
  loading: boolean;
}

const FinancialRadar: React.FC<FinancialRadarProps> = ({ score, data, period, loading }) => {
  if (loading) {
    return (
      <div className="flex justify-center items-center h-64 bg-gray-50 rounded-lg animate-pulse">
        <span className="text-gray-400">正在生成财务雷达图...</span>
      </div>
    );
  }

  if (!data || data.length === 0) return null;

  const chartData = {
    labels: data.map(d => d.subject),
    datasets: [
      {
        label: '基本面评分',
        data: data.map(d => d.value),
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(59, 130, 246, 1)',
      },
    ],
  };

  const options = {
    scales: {
      r: {
        angleLines: {
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
        },
        suggestedMin: 0,
        suggestedMax: 100,
        ticks: {
          stepSize: 20,
          display: false,
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        pointLabels: {
          font: {
            size: 12,
            weight: 'bold' as const,
          },
          color: '#4B5563',
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const item = data[context.dataIndex];
            return ` 分数: ${item.value} (${item.original})`;
          },
        },
      },
    },
    maintainAspectRatio: false,
  };

  const getScoreColor = (s: number) => {
    if (s >= 80) return 'text-green-600';
    if (s >= 60) return 'text-blue-600';
    if (s >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-bold text-gray-800">💎 基本面评分雷达</h3>
          <p className="text-xs text-gray-400 mt-1">数据期: {period}</p>
        </div>
        <div className="text-center">
          <div className={`text-3xl font-black ${getScoreColor(score)}`}>{score.toFixed(1)}</div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider font-bold">综合得分</div>
        </div>
      </div>

      <div className="h-64 mb-6">
        <Radar data={chartData} options={options} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {data.map((item, idx) => (
          <div key={idx} className="flex flex-col p-2 bg-gray-50 rounded-lg border border-gray-100">
            <span className="text-[10px] text-gray-400 font-medium">{item.subject}</span>
            <div className="flex justify-between items-baseline mt-1">
              <span className="text-xs font-bold text-gray-700">{item.original.split(': ')[1]}</span>
              <span className={`text-[10px] font-bold ${item.value >= 60 ? 'text-green-500' : 'text-orange-500'}`}>
                {item.value.toFixed(0)}分
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FinancialRadar;
