// frontend/src/app/(dashboard)/classifier/page.tsx

"use client";

import React, { useState } from 'react';
import { Download, Play, Brain, Gauge, Settings, FileText, BarChart3, Database, CheckCircle, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ClassifierConfigForm } from '@/components/forms/ClassifierConfigForm';
import { DataUploadForm } from '@/components/forms/DataUploadForm';
import { ModelPerformance } from '@/components/charts/ModelPerformance';
import { ConfusionMatrix } from '@/components/charts/ConfusionMatrix';

// Mock data for demonstration
const mockModelMetrics = {
  accuracy: 0.947,
  precision: 0.923,
  recall: 0.891,
  f1Score: 0.907,
  auc: 0.962,
  training: {
    epochs: 25,
    currentEpoch: 25,
    loss: 0.023,
    valLoss: 0.045,
    isTraining: false,
  }
};

const mockTrainingHistory = [
  { epoch: 1, loss: 0.95, valLoss: 0.89, accuracy: 0.62, valAccuracy: 0.65 },
  { epoch: 5, loss: 0.45, valLoss: 0.52, accuracy: 0.78, valAccuracy: 0.76 },
  { epoch: 10, loss: 0.22, valLoss: 0.31, accuracy: 0.87, valAccuracy: 0.85 },
  { epoch: 15, loss: 0.12, valLoss: 0.24, accuracy: 0.92, valAccuracy: 0.89 },
  { epoch: 20, loss: 0.06, valLoss: 0.18, accuracy: 0.95, valAccuracy: 0.91 },
  { epoch: 25, loss: 0.023, valLoss: 0.045, accuracy: 0.947, valAccuracy: 0.934 },
];

const mockConfusionMatrix = {
  matrix: [
    [450, 23],  // True Negative, False Positive
    [18, 387],  // False Negative, True Positive
  ],
  labels: ['Legitimate', 'Misinformation']
};

const mockClassificationResults = [
  { id: 1, text: "Breaking: Scientists discover new planet...", prediction: "legitimate", confidence: 0.94, status: "completed" },
  { id: 2, text: "Shocking: Miracle cure for all diseases found...", prediction: "misinformation", confidence: 0.89, status: "completed" },
  { id: 3, text: "Local elections show increased voter turnout...", prediction: "legitimate", confidence: 0.92, status: "completed" },
  { id: 4, text: "Government secretly controls weather patterns...", prediction: "misinformation", confidence: 0.97, status: "completed" },
  { id: 5, text: "Economic indicators suggest market recovery...", prediction: "legitimate", confidence: 0.85, status: "processing" },
];

const MetricCard: React.FC<{
  title: string;
  value: number;
  format?: 'percentage' | 'decimal';
  icon: React.ReactNode;
  description: string;
}> = ({ title, value, format = 'percentage', icon, description }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">
        {format === 'percentage' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400">{description}</p>
    </CardContent>
  </Card>
);

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'completed':
      return <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Completed</Badge>;
    case 'processing':
      return <Badge variant="default" className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">Processing</Badge>;
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
};

const getPredictionBadge = (prediction: string, confidence: number) => {
  const isHighConfidence = confidence > 0.9;

  if (prediction === 'legitimate') {
    return (
      <Badge
        variant="default"
        className={`${isHighConfidence ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300'}`}
      >
        ✓ Legitimate
      </Badge>
    );
  } else {
    return (
      <Badge
        variant="default"
        className={`${isHighConfidence ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 'bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300'}`}
      >
        ⚠ Misinformation
      </Badge>
    );
  }
};

export default function ClassifierPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isTraining, setIsTraining] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);

  const handleStartTraining = () => {
    setIsTraining(true);
    // Simulate training process
    setTimeout(() => setIsTraining(false), 5000);
  };

  const handleClassifyData = () => {
    setIsClassifying(true);
    // Simulate classification process
    setTimeout(() => setIsClassifying(false), 3000);
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50">Fake News Classifier</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Train, evaluate, and deploy machine learning models for misinformation detection.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
          <TabsTrigger value="classification">Classification</TabsTrigger>
          <TabsTrigger value="data">Data Management</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Model Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            <MetricCard
              title="Accuracy"
              value={mockModelMetrics.accuracy}
              icon={<Gauge className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
              description="Overall model accuracy"
            />
            <MetricCard
              title="Precision"
              value={mockModelMetrics.precision}
              icon={<CheckCircle className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
              description="True positive rate"
            />
            <MetricCard
              title="Recall"
              value={mockModelMetrics.recall}
              icon={<BarChart3 className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
              description="Sensitivity measure"
            />
            <MetricCard
              title="F1-Score"
              value={mockModelMetrics.f1Score}
              icon={<Brain className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
              description="Harmonic mean of precision/recall"
            />
            <MetricCard
              title="AUC-ROC"
              value={mockModelMetrics.auc}
              icon={<BarChart3 className="h-4 w-4 text-slate-600 dark:text-slate-400" />}
              description="Area under ROC curve"
            />
          </div>

          {/* Training Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Current Model Status
              </CardTitle>
              <CardDescription>
                Latest training session and model deployment status
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">BERT-Large Fine-tuned v2.1</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Last trained: 2 hours ago • Deployed: Active
                  </p>
                </div>
                <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                  Production Ready
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Training Progress</span>
                  <span>{mockModelMetrics.training.currentEpoch}/{mockModelMetrics.training.epochs} epochs</span>
                </div>
                <Progress value={(mockModelMetrics.training.currentEpoch / mockModelMetrics.training.epochs) * 100} />
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-slate-600 dark:text-slate-400">Training Loss:</span>
                  <span className="ml-2 font-mono">{mockModelMetrics.training.loss}</span>
                </div>
                <div>
                  <span className="text-slate-600 dark:text-slate-400">Validation Loss:</span>
                  <span className="ml-2 font-mono">{mockModelMetrics.training.valLoss}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Train New Model</CardTitle>
                <CardDescription>Start training with your datasets</CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={handleStartTraining}
                  disabled={isTraining}
                  className="w-full"
                >
                  <Brain className="w-4 h-4 mr-2" />
                  {isTraining ? 'Training...' : 'Start Training'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Classify Content</CardTitle>
                <CardDescription>Analyze text for misinformation</CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={handleClassifyData}
                  disabled={isClassifying}
                  variant="outline"
                  className="w-full"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  {isClassifying ? 'Processing...' : 'Classify Data'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Model Settings</CardTitle>
                <CardDescription>Configure model parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  <Settings className="w-4 h-4 mr-2" />
                  Configure Model
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="training" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Training Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
                <CardDescription>Configure model architecture and training parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <ClassifierConfigForm
                  onSubmit={(config) => {
                    console.log('Training with config:', config);
                    handleStartTraining();
                  }}
                  isLoading={isTraining}
                />
              </CardContent>
            </Card>

            {/* Training Progress */}
            <Card>
              <CardHeader>
                <CardTitle>Training Progress</CardTitle>
                <CardDescription>Real-time training metrics and progress</CardDescription>
              </CardHeader>
              <CardContent>
                <ModelPerformance data={mockTrainingHistory} />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="evaluation" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Confusion Matrix */}
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <CardDescription>Detailed classification performance breakdown</CardDescription>
              </CardHeader>
              <CardContent>
                <ConfusionMatrix data={mockConfusionMatrix} />
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Detailed Metrics</CardTitle>
                <CardDescription>Comprehensive model evaluation results</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Accuracy</span>
                      <span className="text-sm font-mono">{(mockModelMetrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={mockModelMetrics.accuracy * 100} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Precision</span>
                      <span className="text-sm font-mono">{(mockModelMetrics.precision * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={mockModelMetrics.precision * 100} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Recall</span>
                      <span className="text-sm font-mono">{(mockModelMetrics.recall * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={mockModelMetrics.recall * 100} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">F1-Score</span>
                      <span className="text-sm font-mono">{(mockModelMetrics.f1Score * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={mockModelMetrics.f1Score * 100} />
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h4 className="font-medium mb-2">Classification Report</h4>
                  <div className="text-sm space-y-1 font-mono">
                    <div className="grid grid-cols-4 gap-4 font-semibold border-b pb-1">
                      <span>Class</span>
                      <span>Precision</span>
                      <span>Recall</span>
                      <span>F1-Score</span>
                    </div>
                    <div className="grid grid-cols-4 gap-4">
                      <span>Legitimate</span>
                      <span>0.951</span>
                      <span>0.923</span>
                      <span>0.937</span>
                    </div>
                    <div className="grid grid-cols-4 gap-4">
                      <span>Misinformation</span>
                      <span>0.894</span>
                      <span>0.915</span>
                      <span>0.904</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="classification" className="space-y-6">
          {/* Classification Results */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Classifications</CardTitle>
              <CardDescription>Latest content analysis results</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Content Preview</TableHead>
                    <TableHead>Prediction</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mockClassificationResults.map((result) => (
                    <TableRow key={result.id}>
                      <TableCell className="max-w-md">
                        <div className="truncate text-sm">
                          {result.text}
                        </div>
                      </TableCell>
                      <TableCell>
                        {getPredictionBadge(result.prediction, result.confidence)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-mono">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                          <Progress value={result.confidence * 100} className="w-16" />
                        </div>
                      </TableCell>
                      <TableCell>
                        {getStatusBadge(result.status)}
                      </TableCell>
                      <TableCell>
                        <Button variant="ghost" size="sm">
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Batch Classification */}
          <Card>
            <CardHeader>
              <CardTitle>Batch Classification</CardTitle>
              <CardDescription>Upload and classify multiple documents at once</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <DataUploadForm
                    onUpload={(files) => {
                      console.log('Uploaded files for classification:', files);
                      handleClassifyData();
                    }}
                    acceptedTypes={['.txt', '.csv', '.json']}
                    maxFileSize={10 * 1024 * 1024} // 10MB
                    title="Upload Content for Classification"
                    description="Supported formats: TXT, CSV, JSON"
                  />
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium">Classification Options</h4>
                  <div className="space-y-3">
                    <Button className="w-full" disabled={isClassifying}>
                      <Play className="w-4 h-4 mr-2" />
                      {isClassifying ? 'Classifying...' : 'Start Batch Classification'}
                    </Button>
                    <Button variant="outline" className="w-full">
                      <Download className="w-4 h-4 mr-2" />
                      Download Results
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          {/* Data Upload */}
          <Card>
            <CardHeader>
              <CardTitle>Training Data Management</CardTitle>
              <CardDescription>Upload and manage datasets for model training</CardDescription>
            </CardHeader>
            <CardContent>
              <DataUploadForm
                onUpload={(files) => {
                  console.log('Uploaded training data:', files);
                }}
                acceptedTypes={['.csv', '.json', '.parquet']}
                maxFileSize={100 * 1024 * 1024} // 100MB
                title="Upload Training Dataset"
                description="Supported formats: CSV, JSON, Parquet (max 100MB)"
              />
            </CardContent>
          </Card>

          {/* Dataset Statistics */}
          <Card>
            <CardHeader>
              <CardTitle>Dataset Statistics</CardTitle>
              <CardDescription>Overview of available training data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 border rounded-lg">
                  <Database className="w-8 h-8 mx-auto mb-2 text-slate-600 dark:text-slate-400" />
                  <div className="text-2xl font-bold">24,847</div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Total Samples</div>
                </div>

                <div className="text-center p-4 border rounded-lg">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-600" />
                  <div className="text-2xl font-bold">12,423</div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Legitimate News</div>
                </div>

                <div className="text-center p-4 border rounded-lg">
                  <XCircle className="w-8 h-8 mx-auto mb-2 text-red-600" />
                  <div className="text-2xl font-bold">12,424</div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Misinformation</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}