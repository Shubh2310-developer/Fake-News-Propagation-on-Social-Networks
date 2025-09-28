// frontend/src/app/(dashboard)/analytics/page.tsx

"use client";

import React from 'react';
import { TrendingUp, TrendingDown, Users, Activity, Target, Shield } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart } from '@/components/charts/LineChart';
import { BarChart } from '@/components/charts/BarChart';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';

// Mock data - in real app this would come from API
const mockMetrics = {
  totalSimulations: 1247,
  totalSimulationsTrend: 12.5,
  activeUsers: 89,
  activeUsersTrend: -3.2,
  detectionAccuracy: 94.7,
  detectionAccuracyTrend: 2.1,
  misinformationBlocked: 2340,
  misinformationBlockedTrend: 18.3,
};

const mockChartData = [
  { month: 'Jan', simulations: 45, accuracy: 91.2, blocked: 180 },
  { month: 'Feb', simulations: 67, accuracy: 92.1, blocked: 220 },
  { month: 'Mar', simulations: 89, accuracy: 93.5, blocked: 195 },
  { month: 'Apr', simulations: 102, accuracy: 94.2, blocked: 240 },
  { month: 'May', simulations: 115, accuracy: 94.7, blocked: 310 },
  { month: 'Jun', simulations: 98, accuracy: 93.8, blocked: 285 },
];

const mockRecentActivity = [
  { id: 1, type: 'Simulation', name: 'Network Topology Study', status: 'completed', timestamp: '2 hours ago' },
  { id: 2, type: 'Classification', name: 'News Article Batch #47', status: 'running', timestamp: '4 hours ago' },
  { id: 3, type: 'Training', name: 'BERT Fine-tuning v2.1', status: 'completed', timestamp: '6 hours ago' },
  { id: 4, type: 'Simulation', name: 'Equilibrium Analysis', status: 'failed', timestamp: '8 hours ago' },
  { id: 5, type: 'Classification', name: 'Social Media Posts', status: 'completed', timestamp: '12 hours ago' },
];

const StatCard: React.FC<{
  title: string;
  value: string | number;
  trend: number;
  icon: React.ReactNode;
  description: string;
}> = ({ title, value, trend, icon, description }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">{value}</div>
      <div className="flex items-center space-x-2 text-xs text-slate-600 dark:text-slate-400">
        {trend > 0 ? (
          <TrendingUp className="w-4 h-4 text-green-600" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-600" />
        )}
        <span className={trend > 0 ? 'text-green-600' : 'text-red-600'}>
          {Math.abs(trend)}%
        </span>
        <span>{description}</span>
      </div>
    </CardContent>
  </Card>
);

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'completed':
      return <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Completed</Badge>;
    case 'running':
      return <Badge variant="default" className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">Running</Badge>;
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
};

export default function AnalyticsPage() {
  const simulationSeries = [
    { dataKey: 'simulations', name: 'Simulations Run', color: '#3b82f6' }
  ];

  const accuracySeries = [
    { dataKey: 'accuracy', name: 'Detection Accuracy (%)', color: '#10b981' }
  ];

  const blockedSeries = [
    { dataKey: 'blocked', name: 'Content Blocked', color: '#ef4444' }
  ];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50">Analytics Overview</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Monitor platform performance, user activity, and detection metrics.
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Simulations"
          value={mockMetrics.totalSimulations.toLocaleString()}
          trend={mockMetrics.totalSimulationsTrend}
          icon={<Activity className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
          description="from last month"
        />
        <StatCard
          title="Active Users"
          value={mockMetrics.activeUsers}
          trend={mockMetrics.activeUsersTrend}
          icon={<Users className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
          description="from last week"
        />
        <StatCard
          title="Detection Accuracy"
          value={`${mockMetrics.detectionAccuracy}%`}
          trend={mockMetrics.detectionAccuracyTrend}
          icon={<Target className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
          description="from last model"
        />
        <StatCard
          title="Content Blocked"
          value={mockMetrics.misinformationBlocked.toLocaleString()}
          trend={mockMetrics.misinformationBlockedTrend}
          icon={<Shield className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
          description="from last month"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Simulation Activity</CardTitle>
            <CardDescription>Number of simulations run per month</CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart
              data={mockChartData}
              series={simulationSeries}
              xAxisKey="month"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Detection Performance</CardTitle>
            <CardDescription>Model accuracy trends over time</CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart
              data={mockChartData}
              series={accuracySeries}
              xAxisKey="month"
            />
          </CardContent>
        </Card>
      </div>

      {/* Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Content Moderation</CardTitle>
          <CardDescription>Misinformation content blocked per month</CardDescription>
        </CardHeader>
        <CardContent>
          <BarChart
            data={mockChartData}
            series={blockedSeries}
            xAxisKey="month"
          />
        </CardContent>
      </Card>

      {/* Recent Activity Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Latest simulations, classifications, and training jobs</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Type</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Timestamp</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockRecentActivity.map((activity) => (
                <TableRow key={activity.id}>
                  <TableCell className="font-medium">{activity.type}</TableCell>
                  <TableCell>{activity.name}</TableCell>
                  <TableCell>{getStatusBadge(activity.status)}</TableCell>
                  <TableCell className="text-slate-600 dark:text-slate-400">
                    {activity.timestamp}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}