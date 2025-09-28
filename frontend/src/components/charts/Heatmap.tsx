// frontend/src/components/charts/Heatmap.tsx

"use client";

import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface HeatmapData {
  x: string;
  y: string;
  value: number;
}

interface HeatmapProps {
  data: HeatmapData[];
  colorScale?: (value: number) => string;
  minValue?: number;
  maxValue?: number;
}

const defaultColorScale = (value: number, min: number = 0, max: number = 1) => {
  const intensity = (value - min) / (max - min);
  const hue = (1 - intensity) * 240; // Blue to red
  return `hsl(${hue}, 70%, 50%)`;
};

export function Heatmap({
  data,
  colorScale = defaultColorScale,
  minValue,
  maxValue
}: HeatmapProps) {
  const min = minValue ?? Math.min(...data.map(d => d.value));
  const max = maxValue ?? Math.max(...data.map(d => d.value));

  const processedData = data.map(item => ({
    ...item,
    color: colorScale(item.value, min, max),
    size: 400, // Fixed size for squares
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart>
        <XAxis
          type="category"
          dataKey="x"
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="y"
          axisLine={false}
          tickLine={false}
        />
        <ZAxis
          dataKey="size"
          range={[400, 400]}
        />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              const data = payload[0].payload;
              return (
                <div className="bg-background border border-border rounded-lg p-2 shadow-lg">
                  <p className="text-sm">{`x: ${data.x}`}</p>
                  <p className="text-sm">{`y: ${data.y}`}</p>
                  <p className="text-sm font-semibold">{`Value: ${data.value}`}</p>
                </div>
              );
            }
            return null;
          }}
        />
        <Scatter data={processedData} shape="square">
          {processedData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}