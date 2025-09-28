// frontend/src/components/forms/ClassifierConfigForm.tsx

"use client";

import React, { useState } from 'react';
import { Info, RotateCcw, Brain, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ClassifierConfig } from '@/types/classifier';
import { DEFAULT_CLASSIFIER_CONFIG } from '@/lib/constants';

interface ClassifierConfigFormProps {
  initialValues?: Partial<ClassifierConfig>;
  onSubmit: (config: ClassifierConfig) => void;
  onReset?: () => void;
  isLoading?: boolean;
}

// Reusable form section component
const FormSection: React.FC<{
  title: string;
  description?: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
}> = ({ title, description, children, icon }) => (
  <Card className="mb-6">
    <CardHeader>
      <CardTitle className="text-lg flex items-center gap-2">
        {icon}
        {title}
      </CardTitle>
      {description && <CardDescription>{description}</CardDescription>}
    </CardHeader>
    <CardContent className="space-y-6">
      {children}
    </CardContent>
  </Card>
);

// Reusable form field with tooltip
const FormField: React.FC<{
  label: string;
  tooltip?: string;
  children: React.ReactNode;
  className?: string;
  badge?: string;
}> = ({ label, tooltip, children, className = "grid grid-cols-1 md:grid-cols-2 items-center gap-4", badge }) => (
  <div className={className}>
    <Label className="flex items-center text-sm font-medium">
      <span className="flex items-center gap-2">
        {label}
        {badge && <Badge variant="secondary" className="text-xs">{badge}</Badge>}
      </span>
      {tooltip && (
        <div className="relative group ml-2">
          <Info className="w-4 h-4 text-slate-400 hover:text-slate-600 cursor-help" />
          <div className="absolute left-0 top-6 z-10 w-64 p-2 bg-slate-900 text-white text-xs rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            {tooltip}
          </div>
        </div>
      )}
    </Label>
    <div>{children}</div>
  </div>
);

export const ClassifierConfigForm: React.FC<ClassifierConfigFormProps> = ({
  initialValues = {},
  onSubmit,
  onReset,
  isLoading = false,
}) => {
  const [config, setConfig] = useState<ClassifierConfig>({
    ...DEFAULT_CLASSIFIER_CONFIG,
    ...initialValues,
  });

  const updateConfig = <K extends keyof ClassifierConfig>(
    key: K,
    value: ClassifierConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(config);
  };

  const handleReset = () => {
    setConfig({ ...DEFAULT_CLASSIFIER_CONFIG, ...initialValues });
    if (onReset) onReset();
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">
          Classifier Configuration
        </h2>
        <p className="text-slate-600 dark:text-slate-400">
          Configure machine learning models for misinformation detection and content classification.
        </p>
      </div>

      <Tabs defaultValue="model" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="model">Model</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
        </TabsList>

        <TabsContent value="model" className="space-y-6">
          <FormSection
            title="Model Architecture"
            description="Select and configure the machine learning model architecture."
            icon={<Brain className="w-5 h-5" />}
          >
            <FormField
              label="Model Type"
              tooltip="The type of machine learning model to use for classification."
            >
              <Select
                value={config.modelType}
                onValueChange={(value) => updateConfig('modelType', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="transformer">Transformer (BERT/RoBERTa)</SelectItem>
                  <SelectItem value="lstm">LSTM Neural Network</SelectItem>
                  <SelectItem value="cnn">Convolutional Neural Network</SelectItem>
                  <SelectItem value="random-forest">Random Forest</SelectItem>
                  <SelectItem value="svm">Support Vector Machine</SelectItem>
                  <SelectItem value="ensemble">Ensemble Model</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Pre-trained Model"
              tooltip="Use a pre-trained model as the starting point."
            >
              <Select
                value={config.pretrainedModel}
                onValueChange={(value) => updateConfig('pretrainedModel', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bert-base">BERT Base</SelectItem>
                  <SelectItem value="roberta-base">RoBERTa Base</SelectItem>
                  <SelectItem value="distilbert">DistilBERT</SelectItem>
                  <SelectItem value="albert">ALBERT</SelectItem>
                  <SelectItem value="custom">Custom Model</SelectItem>
                  <SelectItem value="none">Train from Scratch</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Max Sequence Length"
              tooltip="Maximum number of tokens in input sequences."
            >
              <Input
                type="number"
                min={64}
                max={2048}
                step={64}
                value={config.maxSequenceLength}
                onChange={(e) => updateConfig('maxSequenceLength', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Hidden Dimensions"
              tooltip="Size of hidden layers in the neural network."
            >
              <Input
                type="number"
                min={64}
                max={2048}
                step={64}
                value={config.hiddenDim}
                onChange={(e) => updateConfig('hiddenDim', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Number of Classes"
              tooltip="Number of output classes for classification."
            >
              <Input
                type="number"
                min={2}
                max={10}
                value={config.numClasses}
                onChange={(e) => updateConfig('numClasses', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Dropout Rate"
              tooltip="Dropout probability for regularization during training."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.dropoutRate]}
                  onValueChange={([value]) => updateConfig('dropoutRate', value)}
                  min={0}
                  max={0.8}
                  step={0.1}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.dropoutRate * 100).toFixed(0)}%
                </div>
              </div>
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="training" className="space-y-6">
          <FormSection
            title="Training Configuration"
            description="Configure training parameters and optimization settings."
            icon={<Zap className="w-5 h-5" />}
          >
            <FormField
              label="Batch Size"
              tooltip="Number of samples processed in each training batch."
              badge="Performance"
            >
              <Select
                value={config.batchSize.toString()}
                onValueChange={(value) => updateConfig('batchSize', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="8">8 (Small GPU)</SelectItem>
                  <SelectItem value="16">16 (Recommended)</SelectItem>
                  <SelectItem value="32">32 (High Memory)</SelectItem>
                  <SelectItem value="64">64 (Large GPU)</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Learning Rate"
              tooltip="Step size for gradient descent optimization."
            >
              <div className="space-y-2">
                <Slider
                  value={[Math.log10(config.learningRate) + 6]}
                  onValueChange={([value]) => updateConfig('learningRate', Math.pow(10, value - 6))}
                  min={2}
                  max={6}
                  step={0.1}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {config.learningRate.toExponential(1)}
                </div>
              </div>
            </FormField>

            <FormField
              label="Number of Epochs"
              tooltip="Number of complete passes through the training dataset."
            >
              <Input
                type="number"
                min={1}
                max={100}
                value={config.epochs}
                onChange={(e) => updateConfig('epochs', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Optimizer"
              tooltip="Optimization algorithm for training the model."
            >
              <Select
                value={config.optimizer}
                onValueChange={(value) => updateConfig('optimizer', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="adam">Adam</SelectItem>
                  <SelectItem value="adamw">AdamW</SelectItem>
                  <SelectItem value="sgd">SGD with Momentum</SelectItem>
                  <SelectItem value="rmsprop">RMSprop</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Weight Decay"
              tooltip="L2 regularization strength for preventing overfitting."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.weightDecay]}
                  onValueChange={([value]) => updateConfig('weightDecay', value)}
                  min={0}
                  max={0.1}
                  step={0.001}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {config.weightDecay.toFixed(3)}
                </div>
              </div>
            </FormField>

            <FormField
              label="Early Stopping"
              tooltip="Stop training early if validation performance stops improving."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.enableEarlyStopping}
                  onCheckedChange={(checked) => updateConfig('enableEarlyStopping', checked)}
                />
              </div>
            </FormField>

            {config.enableEarlyStopping && (
              <FormField
                label="Patience"
                tooltip="Number of epochs to wait before stopping if no improvement."
              >
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={config.patience}
                  onChange={(e) => updateConfig('patience', parseInt(e.target.value))}
                />
              </FormField>
            )}
          </FormSection>
        </TabsContent>

        <TabsContent value="features" className="space-y-6">
          <FormSection
            title="Feature Engineering"
            description="Configure input features and preprocessing options."
          >
            <FormField
              label="Text Features"
              tooltip="Enable text-based features for classification."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.useTextFeatures}
                  onCheckedChange={(checked) => updateConfig('useTextFeatures', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Metadata Features"
              tooltip="Include metadata features like author, timestamp, platform."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.useMetadataFeatures}
                  onCheckedChange={(checked) => updateConfig('useMetadataFeatures', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Network Features"
              tooltip="Include social network features like connectivity patterns."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.useNetworkFeatures}
                  onCheckedChange={(checked) => updateConfig('useNetworkFeatures', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Text Preprocessing"
              tooltip="Text cleaning and normalization options."
              className="grid grid-cols-1 gap-4"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Lowercase normalization</span>
                  <Switch
                    checked={config.preprocessing.lowercase}
                    onCheckedChange={(checked) =>
                      updateConfig('preprocessing', { ...config.preprocessing, lowercase: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Remove punctuation</span>
                  <Switch
                    checked={config.preprocessing.removePunctuation}
                    onCheckedChange={(checked) =>
                      updateConfig('preprocessing', { ...config.preprocessing, removePunctuation: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Remove stopwords</span>
                  <Switch
                    checked={config.preprocessing.removeStopwords}
                    onCheckedChange={(checked) =>
                      updateConfig('preprocessing', { ...config.preprocessing, removeStopwords: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Stemming</span>
                  <Switch
                    checked={config.preprocessing.stemming}
                    onCheckedChange={(checked) =>
                      updateConfig('preprocessing', { ...config.preprocessing, stemming: checked })}
                  />
                </div>
              </div>
            </FormField>

            <FormField
              label="Vocabulary Size"
              tooltip="Maximum number of unique tokens in the vocabulary."
            >
              <Input
                type="number"
                min={1000}
                max={100000}
                step={1000}
                value={config.vocabularySize}
                onChange={(e) => updateConfig('vocabularySize', parseInt(e.target.value))}
              />
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="evaluation" className="space-y-6">
          <FormSection
            title="Evaluation Settings"
            description="Configure model evaluation and validation parameters."
          >
            <FormField
              label="Validation Split"
              tooltip="Percentage of training data to use for validation."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.validationSplit]}
                  onValueChange={([value]) => updateConfig('validationSplit', value)}
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.validationSplit * 100).toFixed(0)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Cross Validation Folds"
              tooltip="Number of folds for cross-validation evaluation."
            >
              <Select
                value={config.crossValidationFolds.toString()}
                onValueChange={(value) => updateConfig('crossValidationFolds', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3">3-fold</SelectItem>
                  <SelectItem value="5">5-fold</SelectItem>
                  <SelectItem value="10">10-fold</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Evaluation Metrics"
              tooltip="Metrics to compute during model evaluation."
              className="grid grid-cols-1 gap-4"
            >
              <div className="space-y-3">
                {['accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix'].map((metric) => (
                  <div key={metric} className="flex items-center justify-between">
                    <span className="text-sm capitalize">{metric.replace('_', ' ')}</span>
                    <Switch
                      checked={config.evaluationMetrics.includes(metric)}
                      onCheckedChange={(checked) => {
                        const metrics = checked
                          ? [...config.evaluationMetrics, metric]
                          : config.evaluationMetrics.filter(m => m !== metric);
                        updateConfig('evaluationMetrics', metrics);
                      }}
                    />
                  </div>
                ))}
              </div>
            </FormField>

            <FormField
              label="Save Model Checkpoints"
              tooltip="Save model state during training for recovery."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.saveCheckpoints}
                  onCheckedChange={(checked) => updateConfig('saveCheckpoints', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Model Export Format"
              tooltip="Format for exporting the trained model."
            >
              <Select
                value={config.exportFormat}
                onValueChange={(value) => updateConfig('exportFormat', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pytorch">PyTorch (.pt)</SelectItem>
                  <SelectItem value="onnx">ONNX (.onnx)</SelectItem>
                  <SelectItem value="tensorflow">TensorFlow (.pb)</SelectItem>
                  <SelectItem value="pickle">Pickle (.pkl)</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
          </FormSection>
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <div className="flex justify-between items-center pt-6 border-t">
        <Button
          type="button"
          variant="outline"
          onClick={handleReset}
          disabled={isLoading}
          className="flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset to Defaults
        </Button>

        <div className="flex gap-3">
          <Button
            type="button"
            variant="outline"
            disabled={isLoading}
          >
            Validate Config
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={isLoading}
          >
            Save Template
          </Button>
          <Button
            type="submit"
            size="lg"
            disabled={isLoading}
            className="min-w-[140px] flex items-center gap-2"
          >
            <Brain className="w-4 h-4" />
            {isLoading ? 'Training...' : 'Start Training'}
          </Button>
        </div>
      </div>
    </form>
  );
};