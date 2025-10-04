// frontend/src/app/(dashboard)/classifier/page.tsx

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  FileText,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Brain,
  Loader2,
  BarChart2,
  Info
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { BarChart } from '@/components/charts/BarChart';
import { useClassifier } from '@/hooks/useClassifier';
import { cn } from '@/lib/utils';
import { ClassifierModelType } from '@/types/classifier';

// ================================================================
// Animation Variants
// ================================================================

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const scaleIn = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.95 },
};

// ================================================================
// Components
// ================================================================

// Animated Progress Bar Component
interface AnimatedProgressProps {
  value: number;
  className?: string;
}

const AnimatedProgress: React.FC<AnimatedProgressProps> = ({ value, className }) => {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 100);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div className="relative">
      <Progress
        value={animatedValue}
        className={cn("h-3", className)}
        style={{
          transition: 'all 1s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
      />
    </div>
  );
};

// Model Selection Info - Updated with actual trained model performance
const MODEL_INFO: Record<ClassifierModelType, { label: string; description: string }> = {
  ensemble: {
    label: 'Ensemble (Recommended)',
    description: 'Weighted voting ensemble combining all models - 99.86% accuracy',
  },
  gradient_boosting: {
    label: 'Gradient Boosting',
    description: 'Best single model with 99.95% accuracy on 8K test samples',
  },
  random_forest: {
    label: 'Random Forest',
    description: 'High-performance tree ensemble with 99.89% accuracy',
  },
  naive_bayes: {
    label: 'Naive Bayes',
    description: 'Fast probabilistic classifier with 94.83% accuracy',
  },
  logistic_regression: {
    label: 'Logistic Regression',
    description: 'Lightweight baseline model with 66.71% accuracy',
  },
  bert: {
    label: 'DistilBERT Transformer',
    description: 'Deep learning transformer model (coming soon - in development)',
  },
  lstm: {
    label: 'LSTM Neural Network',
    description: 'Recurrent neural network model (coming soon - in development)',
  },
};

// ================================================================
// Main Classifier Page Component
// ================================================================

export default function ClassifierPage() {
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState<ClassifierModelType>('ensemble');
  const [showResults, setShowResults] = useState(false);

  const {
    classificationResult,
    isLoading,
    error,
    classifyText,
  } = useClassifier();

  // Handle text classification
  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      return;
    }

    setShowResults(false);
    await classifyText(inputText, {
      model_type: selectedModel,
      explain: true,
      confidence_threshold: 0.5,
    });
    setShowResults(true);
  };

  // Reset when user starts typing after getting results
  useEffect(() => {
    if (showResults && inputText !== classificationResult?.text) {
      setShowResults(false);
    }
  }, [inputText, showResults, classificationResult?.text]);

  // Get verdict styling
  const getVerdictStyle = (prediction?: 'fake' | 'real') => {
    if (prediction === 'real') {
      return {
        textColor: 'text-green-700',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        icon: <CheckCircle2 className="h-8 w-8 text-green-600" />,
        label: 'Likely Real News',
      };
    }
    return {
      textColor: 'text-red-700',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      icon: <XCircle className="h-8 w-8 text-red-600" />,
      label: 'Likely Fake News',
    };
  };

  const verdictStyle = getVerdictStyle(classificationResult?.prediction);

  // Prepare probability data for bar chart
  const probabilityData = classificationResult
    ? [
        {
          label: 'Real News',
          probability: (classificationResult.probabilities.real * 100).toFixed(1),
        },
        {
          label: 'Fake News',
          probability: (classificationResult.probabilities.fake * 100).toFixed(1),
        },
      ]
    : [];

  const probabilitySeries = [
    {
      dataKey: 'probability',
      name: 'Probability (%)',
      color: '#3b82f6',
    },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-3 bg-blue-100 rounded-lg">
            <Sparkles className="h-6 w-6 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
            Real-Time News Classifier
          </h1>
        </div>
        <p className="text-gray-600 text-lg">
          Analyze news articles, posts, or other text to detect potential misinformation
        </p>
      </motion.div>

      {/* Main Classifier Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <FileText className="h-6 w-6 text-blue-600" />
              Text Classification
            </CardTitle>
            <CardDescription>
              Enter text below and select a model to analyze its authenticity
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Input Section */}
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Text to Analyze
                </label>
                <Textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste the news article or text you want to analyze...

Example: Breaking news: Scientists have made a groundbreaking discovery that could change everything we know about..."
                  className="min-h-[200px] text-base resize-y"
                  disabled={isLoading}
                />
                <div className="flex items-center justify-between mt-2">
                  <span className="text-xs text-gray-500">
                    {inputText.length} characters
                  </span>
                  {inputText.length > 0 && (
                    <span className="text-xs text-gray-500">
                      ~{Math.ceil(inputText.split(/\s+/).length)} words
                    </span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">
                    Classification Model
                  </label>
                  <Select
                    value={selectedModel}
                    onValueChange={(value) => setSelectedModel(value as ClassifierModelType)}
                    disabled={isLoading}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(MODEL_INFO).map(([key, info]) => (
                        <SelectItem key={key} value={key}>
                          {info.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500 mt-2">
                    {MODEL_INFO[selectedModel].description}
                  </p>
                </div>

                <div className="flex items-end">
                  <Button
                    onClick={handleAnalyze}
                    disabled={isLoading || !inputText.trim()}
                    size="lg"
                    className="w-full h-12 text-base font-semibold"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Brain className="mr-2 h-5 w-5" />
                        Analyze Text
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>

            {/* Error Display */}
            <AnimatePresence>
              {error && (
                <motion.div {...fadeInUp} transition={{ duration: 0.3 }}>
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Classification Error</AlertTitle>
                    <AlertDescription>
                      {error || 'An error occurred while analyzing the text. Please try again.'}
                    </AlertDescription>
                  </Alert>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results Display */}
            <AnimatePresence mode="wait">
              {showResults && classificationResult && !error && (
                <motion.div
                  key="results"
                  {...scaleIn}
                  transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
                >
                  <div className="pt-6 border-t border-gray-200 dark:border-gray-800">
                    <Tabs defaultValue="prediction" className="w-full">
                      <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="prediction" className="flex items-center gap-2">
                          <TrendingUp className="h-4 w-4" />
                          Prediction
                        </TabsTrigger>
                        <TabsTrigger value="probabilities" className="flex items-center gap-2">
                          <BarChart2 className="h-4 w-4" />
                          Probabilities
                        </TabsTrigger>
                        <TabsTrigger value="explanation" className="flex items-center gap-2">
                          <Info className="h-4 w-4" />
                          Explanation
                        </TabsTrigger>
                      </TabsList>

                      {/* Prediction Tab */}
                      <TabsContent value="prediction" className="space-y-6 mt-6">
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.4, delay: 0.1 }}
                          className={cn(
                            "p-8 rounded-xl border-2 text-center",
                            verdictStyle.bgColor,
                            verdictStyle.borderColor
                          )}
                        >
                          <div className="flex flex-col items-center gap-4">
                            {verdictStyle.icon}
                            <div>
                              <p className="text-sm font-medium text-gray-600 mb-1">
                                Classification Result
                              </p>
                              <h3 className={cn("text-3xl font-bold", verdictStyle.textColor)}>
                                {verdictStyle.label}
                              </h3>
                            </div>
                          </div>
                        </motion.div>

                        <div className="space-y-4">
                          <div>
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-sm font-medium text-gray-700">
                                Confidence Score
                              </span>
                              <Badge variant="outline" className="text-base font-semibold">
                                {(classificationResult.confidence * 100).toFixed(1)}%
                              </Badge>
                            </div>
                            <AnimatedProgress
                              value={classificationResult.confidence * 100}
                              className={cn(
                                "h-4",
                                classificationResult.prediction === 'real'
                                  ? '[&>div]:bg-green-600'
                                  : '[&>div]:bg-red-600'
                              )}
                            />
                            <p className="text-xs text-gray-500 mt-2">
                              {classificationResult.confidence >= 0.9
                                ? 'Very high confidence in this prediction'
                                : classificationResult.confidence >= 0.7
                                ? 'High confidence in this prediction'
                                : classificationResult.confidence >= 0.5
                                ? 'Moderate confidence in this prediction'
                                : 'Low confidence - results should be interpreted carefully'}
                            </p>
                          </div>

                          <div className="p-4 bg-gray-50 rounded-lg space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">Model Used:</span>
                              <span className="font-medium text-gray-900">
                                {classificationResult.model_used}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">Processing Time:</span>
                              <span className="font-medium text-gray-900">
                                {classificationResult.processing_time}ms
                              </span>
                            </div>
                            {classificationResult.metadata && (
                              <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-600">Text Length:</span>
                                <span className="font-medium text-gray-900">
                                  {classificationResult.metadata.text_length} characters
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </TabsContent>

                      {/* Probabilities Tab */}
                      <TabsContent value="probabilities" className="space-y-6 mt-6">
                        <motion.div
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay: 0.1 }}
                        >
                          <div className="mb-6">
                            <h4 className="text-lg font-semibold text-gray-900 mb-2">
                              Raw Classification Probabilities
                            </h4>
                            <p className="text-sm text-gray-600">
                              Detailed probability distribution across classification labels
                            </p>
                          </div>

                          <BarChart
                            data={probabilityData}
                            series={probabilitySeries}
                            xAxisKey="label"
                          />

                          <div className="grid grid-cols-2 gap-4 mt-6">
                            <motion.div
                              initial={{ opacity: 0, scale: 0.9 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ duration: 0.3, delay: 0.2 }}
                              className="p-4 bg-green-50 rounded-lg border border-green-200"
                            >
                              <div className="flex items-center gap-2 mb-2">
                                <CheckCircle2 className="h-5 w-5 text-green-600" />
                                <span className="font-medium text-gray-900">
                                  Real News
                                </span>
                              </div>
                              <div className="text-3xl font-bold text-green-700">
                                {(classificationResult.probabilities.real * 100).toFixed(1)}%
                              </div>
                            </motion.div>

                            <motion.div
                              initial={{ opacity: 0, scale: 0.9 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ duration: 0.3, delay: 0.3 }}
                              className="p-4 bg-red-50 rounded-lg border border-red-200"
                            >
                              <div className="flex items-center gap-2 mb-2">
                                <XCircle className="h-5 w-5 text-red-600" />
                                <span className="font-medium text-gray-900">
                                  Fake News
                                </span>
                              </div>
                              <div className="text-3xl font-bold text-red-700">
                                {(classificationResult.probabilities.fake * 100).toFixed(1)}%
                              </div>
                            </motion.div>
                          </div>
                        </motion.div>
                      </TabsContent>

                      {/* Explanation Tab */}
                      <TabsContent value="explanation" className="space-y-6 mt-6">
                        <motion.div
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay: 0.1 }}
                        >
                          <div className="mb-6">
                            <h4 className="text-lg font-semibold text-gray-900 mb-2">
                              Model Explanation
                            </h4>
                            <p className="text-sm text-gray-600">
                              Understanding which features influenced this prediction
                            </p>
                          </div>

                          {classificationResult.explanation ? (
                            <div className="space-y-6">
                              {/* Top Contributing Phrases */}
                              {classificationResult.explanation.top_phrases &&
                                classificationResult.explanation.top_phrases.length > 0 && (
                                  <div>
                                    <h5 className="text-sm font-semibold text-gray-900 mb-3">
                                      Key Influential Phrases
                                    </h5>
                                    <div className="space-y-2">
                                      {classificationResult.explanation.top_phrases.slice(0, 5).map((phrase, idx) => (
                                        <motion.div
                                          key={idx}
                                          initial={{ opacity: 0, x: -20 }}
                                          animate={{ opacity: 1, x: 0 }}
                                          transition={{ duration: 0.3, delay: 0.2 + idx * 0.05 }}
                                          className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg"
                                        >
                                          <Badge
                                            variant="outline"
                                            className={cn(
                                              "shrink-0",
                                              phrase.type === 'positive'
                                                ? 'border-green-500 text-green-700'
                                                : 'border-red-500 text-red-700'
                                            )}
                                          >
                                            {phrase.type === 'positive' ? 'Supports Real' : 'Indicates Fake'}
                                          </Badge>
                                          <span className="flex-1 text-sm text-gray-700">
                                            "{phrase.phrase}"
                                          </span>
                                          <span className="text-sm font-semibold text-gray-900">
                                            {(phrase.contribution * 100).toFixed(1)}%
                                          </span>
                                        </motion.div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                              {/* Feature Importance */}
                              {classificationResult.explanation.feature_importance &&
                                classificationResult.explanation.feature_importance.length > 0 && (
                                  <div>
                                    <h5 className="text-sm font-semibold text-gray-900 mb-3">
                                      Feature Importance
                                    </h5>
                                    <div className="space-y-2">
                                      {classificationResult.explanation.feature_importance.slice(0, 5).map((feature, idx) => (
                                        <motion.div
                                          key={idx}
                                          initial={{ opacity: 0, x: -20 }}
                                          animate={{ opacity: 1, x: 0 }}
                                          transition={{ duration: 0.3, delay: 0.3 + idx * 0.05 }}
                                          className="space-y-1"
                                        >
                                          <div className="flex items-center justify-between text-sm">
                                            <span className="font-medium text-gray-700">
                                              {feature.feature}
                                            </span>
                                            <span className="text-gray-900 font-semibold">
                                              {(feature.importance * 100).toFixed(1)}%
                                            </span>
                                          </div>
                                          <Progress
                                            value={feature.importance * 100}
                                            className={cn(
                                              "h-2",
                                              feature.type === 'positive'
                                                ? '[&>div]:bg-green-600'
                                                : '[&>div]:bg-red-600'
                                            )}
                                          />
                                        </motion.div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                              {/* Information Box */}
                              <Alert>
                                <Info className="h-4 w-4" />
                                <AlertTitle>Model Transparency</AlertTitle>
                                <AlertDescription>
                                  This explanation shows which parts of the text most strongly influenced
                                  the model's decision. Features marked as "Supports Real" push the
                                  prediction toward real news, while "Indicates Fake" features push toward
                                  fake news classification.
                                </AlertDescription>
                              </Alert>
                            </div>
                          ) : (
                            <Alert>
                              <Info className="h-4 w-4" />
                              <AlertTitle>Explanation Unavailable</AlertTitle>
                              <AlertDescription>
                                Model explanations are not available for this classification result.
                                This may occur with certain model types or if explanation generation was disabled.
                              </AlertDescription>
                            </Alert>
                          )}
                        </motion.div>
                      </TabsContent>
                    </Tabs>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </motion.div>

      {/* Information Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              How It Works
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            Our models analyze linguistic patterns, source credibility indicators, and content
            structure to identify potential misinformation with high accuracy.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              Model Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            Trained on 35,000+ samples from multiple datasets: Ensemble achieves 99.86% accuracy,
            Gradient Boosting 99.95%, Random Forest 99.89%, and Naive Bayes 94.83%.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-yellow-600" />
              Important Note
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            While highly accurate, no automated system is perfect. Always verify critical
            information through multiple trusted sources.
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}