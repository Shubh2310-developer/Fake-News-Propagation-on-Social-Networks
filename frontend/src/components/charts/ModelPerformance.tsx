import React from 'react';

export interface ModelPerformanceProps {
  data?: {
    epoch: number;
    loss: number;
    valLoss: number;
    accuracy: number;
    valAccuracy: number;
  }[];
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

export const ModelPerformance: React.FC<ModelPerformanceProps> = ({ data, metrics }) => {
  return (
    <div className="p-4 border rounded-lg">
      <h3 className="text-lg font-semibold mb-4">Model Performance</h3>
      {metrics && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-600">Accuracy</div>
            <div className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Precision</div>
            <div className="text-2xl font-bold">{(metrics.precision * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Recall</div>
            <div className="text-2xl font-bold">{(metrics.recall * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">F1 Score</div>
            <div className="text-2xl font-bold">{(metrics.f1Score * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
      {data && (
        <div className="text-gray-600">
          Training data visualization would go here (epochs: {data.length})
        </div>
      )}
    </div>
  );
};