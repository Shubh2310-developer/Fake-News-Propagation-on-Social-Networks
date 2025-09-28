// frontend/src/components/charts/LineChart.tsx

"use client";

import {
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
  CartesianGrid,
} from 'recharts';

interface ChartSeries {
  dataKey: string;
  name: string;
  color: string;
}

interface LineChartProps {
  data: any[];
  series: ChartSeries[];
  xAxisKey: string;
}

export function LineChart({ data, series, xAxisKey }: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RechartsLineChart data={data}>
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
          <Line
            key={s.dataKey}
            type="monotone"
            dataKey={s.dataKey}
            name={s.name}
            stroke={s.color}
            dot={false}
          />
        ))}
      </RechartsLineChart>
    </ResponsiveContainer>
  );
}