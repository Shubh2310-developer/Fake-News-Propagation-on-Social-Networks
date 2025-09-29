import React from 'react';

export interface ConfusionMatrixProps {
  data?: {
    matrix: number[][];
    labels: string[];
  };
  matrix?: number[][];
  labels?: string[];
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ data, matrix, labels }) => {
  const actualMatrix = data?.matrix || matrix || [];
  const actualLabels = data?.labels || labels || [];

  return (
    <div className="p-4 border rounded-lg">
      <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
      {actualMatrix.length > 0 && actualLabels.length > 0 ? (
        <div className="grid grid-cols-3 gap-2 max-w-xs">
          <div></div>
          {actualLabels.map(label => (
            <div key={label} className="text-center font-semibold">{label}</div>
          ))}
          {actualMatrix.map((row, i) => (
            <React.Fragment key={i}>
              <div className="font-semibold">{actualLabels[i]}</div>
              {row.map((value, j) => (
                <div key={j} className="text-center p-2 border bg-gray-50">
                  {value}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      ) : (
        <div className="text-gray-600">No confusion matrix data available</div>
      )}
    </div>
  );
};