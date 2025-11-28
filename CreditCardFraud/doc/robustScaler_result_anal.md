import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ModelMetricsHistogram = () => {
  const results = {
    'XGB_robustScaler_HO': {'AUC': 0.9645, '정확도': 0.9996, '정밀도': 0.9310, '재현율': 0.8265, 'F1': 0.8757},
    'XGB_robustScaler_HO_smote': {'AUC': 0.9757, '정확도': 0.9991, '정밀도': 0.6983, '재현율': 0.8265, 'F1': 0.7570},
    'XGB_robustScaler_HO_under': {'AUC': 0.9725, '정확도': 0.9402, '정밀도': 0.0255, '재현율': 0.9082, 'F1': 0.0497},
    'XGB_robustScaler_HO_combined': {'AUC': 0.9757, '정확도': 0.9991, '정밀도': 0.6983, '재현율': 0.8265, 'F1': 0.7570},
    'GB_robustScaler_HO': {'AUC': 0.2959, '정확도': 0.9985, '정밀도': 0.7308, '재현율': 0.1939, 'F1': 0.3065},
    'GB_robustScaler_HO_smote': {'AUC': 0.9750, '정확도': 0.9874, '정밀도': 0.1094, '재현율': 0.8878, 'F1': 0.1948},
    'GB_robustScaler_HO_under': {'AUC': 0.9715, '정확도': 0.9597, '정밀도': 0.0367, '재현율': 0.8878, 'F1': 0.0705},
    'GB_robustScaler_HO_combined': {'AUC': 0.9750, '정확도': 0.9874, '정밀도': 0.1094, '재현율': 0.8878, 'F1': 0.1948},
    'DT_robustScaler_HO': {'AUC': 0.8926, '정확도': 0.9992, '정밀도': 0.7549, '재현율': 0.7857, 'F1': 0.7700},
    'DT_robustScaler_HO_smote': {'AUC': 0.8714, '정확도': 0.9974, '정밀도': 0.3744, '재현율': 0.7449, 'F1': 0.4983},
    'DT_robustScaler_HO_under': {'AUC': 0.9125, '정확도': 0.9169, '정밀도': 0.0185, '재현율': 0.9082, 'F1': 0.0362},
    'DT_robustScaler_HO_combined': {'AUC': 0.8714, '정확도': 0.9974, '정밀도': 0.3744, '재현율': 0.7449, 'F1': 0.4983},
    'mlp_robustScaler_HO': {'AUC': 0.9652, '정확도': 0.9995, '정밀도': 0.9080, '재현율': 0.8061, 'F1': 0.8541},
    'mlp_robustScaler_HO_smote': {'AUC': 0.9663, '정확도': 0.9991, '정밀도': 0.7182, '재현율': 0.8061, 'F1': 0.7596},
    'mlp_robustScaler_HO_under': {'AUC': 0.9719, '정확도': 0.9682, '정밀도': 0.0461, '재현율': 0.8878, 'F1': 0.0877},
    'mlp_robustScaler_HO_combined': {'AUC': 0.9663, '정확도': 0.9991, '정밀도': 0.7182, '재현율': 0.8061, 'F1': 0.7596},
    'lr_robustScaler_HO': {'AUC': 0.9605, '정확도': 0.9764, '정밀도': 0.0613, '재현율': 0.8878, 'F1': 0.1147},
    'lr_robustScaler_HO_smote': {'AUC': 0.9601, '정확도': 0.9754, '정밀도': 0.0583, '재현율': 0.8776, 'F1': 0.1093},
    'lr_robustScaler_HO_under': {'AUC': 0.9689, '정확도': 0.9838, '정밀도': 0.0862, '재현율': 0.8776, 'F1': 0.1569},
    'lr_robustScaler_HO_combined': {'AUC': 0.9601, '정확도': 0.9754, '정밀도': 0.0583, '재현율': 0.8776, 'F1': 0.1093}
  };

  // 데이터를 차트 형식으로 변환
  const chartData = Object.entries(results).map(([name, metrics]) => ({
    name: name,
    AUC: metrics.AUC,
    정밀도: metrics.정밀도,
    재현율: metrics.재현율,
    F1: metrics.F1
  }));

  const colors = {
    AUC: '#8884d8',
    정밀도: '#82ca9d',
    재현율: '#ffc658',
    F1: '#ff7c7c'
  };

  return (
    <div className="w-full h-screen bg-gray-50 p-6 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          모델별 성능 지표 비교
        </h1>
        
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <ResponsiveContainer width="100%" height={600}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 120 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end"
                height={150}
                interval={0}
                tick={{ fontSize: 11 }}
              />
              <YAxis 
                domain={[0, 1]}
                label={{ value: '점수', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                formatter={(value) => value.toFixed(4)}
              />
              <Legend 
                verticalAlign="top"
                height={36}
              />
              <Bar dataKey="AUC" fill={colors.AUC} />
              <Bar dataKey="정밀도" fill={colors.정밀도} />
              <Bar dataKey="재현율" fill={colors.재현율} />
              <Bar dataKey="F1" fill={colors.F1} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(colors).map(([metric, color]) => (
            <div key={metric} className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center mb-2">
                <div 
                  className="w-4 h-4 rounded mr-2" 
                  style={{ backgroundColor: color }}
                />
                <h3 className="font-semibold text-lg">{metric}</h3>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-gray-600">
                  최고: {Math.max(...chartData.map(d => d[metric])).toFixed(4)}
                </p>
                <p className="text-sm text-gray-600">
                  최저: {Math.min(...chartData.map(d => d[metric])).toFixed(4)}
                </p>
                <p className="text-sm text-gray-600">
                  평균: {(chartData.reduce((sum, d) => sum + d[metric], 0) / chartData.length).toFixed(4)}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">주요 관찰 사항</h2>
          <ul className="list-disc list-inside space-y-2 text-gray-700">
            <li>XGB 모델이 전반적으로 가장 균형잡힌 성능을 보입니다.</li>
            <li>Under-sampling을 적용한 모델들은 재현율은 높지만 정밀도가 매우 낮습니다.</li>
            <li>GB_robustScaler_HO 모델은 AUC가 특이하게 낮게 나타났습니다.</li>
            <li>정확도는 대부분의 모델에서 높게 나타나지만, 불균형 데이터에서는 신뢰도가 낮을 수 있습니다.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ModelMetricsHistogram;