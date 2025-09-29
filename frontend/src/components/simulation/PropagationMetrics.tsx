import React from 'react';

export interface PropagationMetricsProps {
  data?: any;
}

export const PropagationMetrics: React.FC<PropagationMetricsProps> = ({ data }) => {
  return (
    <div className="p-4 border rounded-lg">
      <h3 className="text-lg font-semibold mb-4">Propagation Metrics</h3>
      <div className="text-gray-600">
        Propagation metrics will be displayed here.
      </div>
    </div>
  );
};