// frontend/src/components/charts/BarChart.tsx

"use client";

import {
  BarChart as RechartsBarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
  CartesianGrid,
} from 'recharts';

interface ChartSeries {
  dataKey: string;
  name: string;
  color: string;
}

interface BarChartProps {
  data: any[];
  series: ChartSeries[];
  xAxisKey: string;
}

export function BarChart({ data, series, xAxisKey }: BarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RechartsBarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--background))',
            borderColor: 'hsl(var(--border))',
          }}
        />
        <Legend />
        {series.map((s) => (
          <Bar key={s.dataKey} dataKey={s.dataKey} name={s.name} fill={s.color} />
        ))}
      </RechartsBarChart>
    </ResponsiveContainer>
  );
}