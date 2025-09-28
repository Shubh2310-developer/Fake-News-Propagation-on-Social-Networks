// frontend/src/app/api/classifier/train/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Types for request/response
interface TrainRequest {
  datasetId: string;
  modelConfig: {
    modelType: 'bert' | 'roberta' | 'distilbert' | 'custom';
    learningRate?: number;
    batchSize?: number;
    epochs?: number;
    validationSplit?: number;
    earlyStopping?: boolean;
    maxLength?: number;
    freezeLayers?: number;
  };
  experimentName?: string;
  tags?: string[];
}

interface TrainResponse {
  trainingId: string;
  status: 'queued' | 'starting' | 'running';
  message: string;
  estimatedDurationMinutes: number;
  queuePosition?: number;
  experiment: {
    name: string;
    tags: string[];
    createdAt: string;
  };
  config: {
    datasetId: string;
    modelType: string;
    parameters: object;
  };
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
}

// Simulate training job queue and management
const activeTrainingJobs = new Map<string, {
  id: string;
  status: 'queued' | 'starting' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: Date;
  estimatedEndTime: Date;
  config: any;
}>();

// Mock dataset validation
const validateDataset = async (datasetId: string): Promise<{
  exists: boolean;
  size: number;
  format: string;
  samples: number;
}> => {
  // Simulate dataset lookup delay
  await new Promise(resolve => setTimeout(resolve, 100));

  // Mock dataset registry
  const mockDatasets = [
    { id: 'dataset_001', size: 45000000, format: 'csv', samples: 50000 },
    { id: 'dataset_002', size: 23000000, format: 'json', samples: 25000 },
    { id: 'dataset_003', size: 67000000, format: 'parquet', samples: 75000 },
    { id: 'fake_news_v1', size: 12000000, format: 'csv', samples: 15000 },
    { id: 'real_news_v1', size: 34000000, format: 'csv', samples: 40000 },
  ];

  const dataset = mockDatasets.find(d => d.id === datasetId);

  if (!dataset) {
    return { exists: false, size: 0, format: '', samples: 0 };
  }

  return {
    exists: true,
    size: dataset.size,
    format: dataset.format,
    samples: dataset.samples
  };
};

// Generate unique training ID
const generateTrainingId = (): string => {
  const timestamp = Date.now().toString(36);
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  return `train_${timestamp}_${randomSuffix}`;
};

// Estimate training duration based on config
const estimateTrainingDuration = (config: TrainRequest['modelConfig'], samples: number): number => {
  const baseTimePerSample = {
    'bert': 0.5,      // seconds per sample
    'roberta': 0.6,
    'distilbert': 0.3,
    'custom': 0.4
  };

  const epochs = config.epochs || 3;
  const batchSize = config.batchSize || 16;
  const timePerSample = baseTimePerSample[config.modelType] || 0.4;

  // Calculate total training time
  const totalBatches = Math.ceil(samples / batchSize) * epochs;
  const totalSeconds = totalBatches * timePerSample * batchSize / 60; // Convert to minutes

  // Add overhead for setup, validation, etc.
  return Math.round(totalSeconds * 1.2);
};

export async function POST(request: NextRequest) {
  try {
    // Parse request body
    let body: TrainRequest;

    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json(
        {
          error: 'Invalid JSON in request body',
          code: 'INVALID_JSON',
          details: 'Request body must be valid JSON'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate required fields
    if (!body.datasetId) {
      return NextResponse.json(
        {
          error: 'Missing required field: datasetId',
          code: 'MISSING_DATASET_ID',
          details: 'The datasetId field is required'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (!body.modelConfig) {
      return NextResponse.json(
        {
          error: 'Missing required field: modelConfig',
          code: 'MISSING_MODEL_CONFIG',
          details: 'The modelConfig field is required'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (!body.modelConfig.modelType) {
      return NextResponse.json(
        {
          error: 'Missing required field: modelConfig.modelType',
          code: 'MISSING_MODEL_TYPE',
          details: 'The modelConfig.modelType field is required'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate model type
    const validModelTypes = ['bert', 'roberta', 'distilbert', 'custom'];
    if (!validModelTypes.includes(body.modelConfig.modelType)) {
      return NextResponse.json(
        {
          error: 'Invalid model type',
          code: 'INVALID_MODEL_TYPE',
          details: `Model type must be one of: ${validModelTypes.join(', ')}`
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate model configuration parameters
    const config = body.modelConfig;

    if (config.learningRate && (config.learningRate <= 0 || config.learningRate > 1)) {
      return NextResponse.json(
        {
          error: 'Invalid learning rate',
          code: 'INVALID_LEARNING_RATE',
          details: 'Learning rate must be between 0 and 1'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (config.batchSize && (config.batchSize < 1 || config.batchSize > 128)) {
      return NextResponse.json(
        {
          error: 'Invalid batch size',
          code: 'INVALID_BATCH_SIZE',
          details: 'Batch size must be between 1 and 128'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (config.epochs && (config.epochs < 1 || config.epochs > 100)) {
      return NextResponse.json(
        {
          error: 'Invalid epochs',
          code: 'INVALID_EPOCHS',
          details: 'Epochs must be between 1 and 100'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate dataset exists
    const datasetInfo = await validateDataset(body.datasetId);

    if (!datasetInfo.exists) {
      return NextResponse.json(
        {
          error: 'Dataset not found',
          code: 'DATASET_NOT_FOUND',
          details: `Dataset with ID '${body.datasetId}' does not exist`
        } as ErrorResponse,
        { status: 404 }
      );
    }

    // Check if dataset has enough samples
    if (datasetInfo.samples < 100) {
      return NextResponse.json(
        {
          error: 'Insufficient training data',
          code: 'INSUFFICIENT_DATA',
          details: 'Dataset must contain at least 100 samples for training'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Generate training job
    const trainingId = generateTrainingId();
    const estimatedDuration = estimateTrainingDuration(config, datasetInfo.samples);
    const experimentName = body.experimentName || `${config.modelType}_${new Date().toISOString().split('T')[0]}`;

    // Set default configuration values
    const finalConfig = {
      modelType: config.modelType,
      learningRate: config.learningRate || 2e-5,
      batchSize: config.batchSize || 16,
      epochs: config.epochs || 3,
      validationSplit: config.validationSplit || 0.2,
      earlyStopping: config.earlyStopping ?? true,
      maxLength: config.maxLength || 512,
      freezeLayers: config.freezeLayers || 0,
    };

    // Store training job (in real app, this would be in a database/queue)
    const trainingJob = {
      id: trainingId,
      status: 'queued' as const,
      progress: 0,
      startTime: new Date(),
      estimatedEndTime: new Date(Date.now() + estimatedDuration * 60 * 1000),
      config: {
        datasetId: body.datasetId,
        modelConfig: finalConfig,
        experimentName,
        tags: body.tags || []
      }
    };

    activeTrainingJobs.set(trainingId, trainingJob);

    // Simulate job queue position
    const queuePosition = Math.floor(Math.random() * 3) + 1;

    // Prepare response
    const response: TrainResponse = {
      trainingId,
      status: 'queued',
      message: 'Training job has been queued successfully',
      estimatedDurationMinutes: estimatedDuration,
      queuePosition,
      experiment: {
        name: experimentName,
        tags: body.tags || [],
        createdAt: new Date().toISOString()
      },
      config: {
        datasetId: body.datasetId,
        modelType: config.modelType,
        parameters: finalConfig
      }
    };

    // Log training job creation
    console.log(`[CLASSIFIER] Training job created: ${trainingId} for dataset ${body.datasetId}`);

    // Simulate async job processing (in real app, this would be handled by a job queue)
    setTimeout(() => {
      const job = activeTrainingJobs.get(trainingId);
      if (job) {
        job.status = 'starting';
        activeTrainingJobs.set(trainingId, job);

        setTimeout(() => {
          const runningJob = activeTrainingJobs.get(trainingId);
          if (runningJob) {
            runningJob.status = 'running';
            activeTrainingJobs.set(trainingId, runningJob);
          }
        }, 5000);
      }
    }, 2000);

    return NextResponse.json(response, { status: 202 });

  } catch (error) {
    console.error('[CLASSIFIER] Training initiation error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during training initiation',
        code: 'TRAINING_INIT_FAILED',
        details: 'An unexpected error occurred while initiating the training job'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// Get training job status
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const trainingId = searchParams.get('id');

  if (!trainingId) {
    return NextResponse.json(
      {
        error: 'Missing training ID',
        code: 'MISSING_TRAINING_ID',
        details: 'Training ID is required as a query parameter'
      } as ErrorResponse,
      { status: 400 }
    );
  }

  const job = activeTrainingJobs.get(trainingId);

  if (!job) {
    return NextResponse.json(
      {
        error: 'Training job not found',
        code: 'TRAINING_JOB_NOT_FOUND',
        details: `No training job found with ID: ${trainingId}`
      } as ErrorResponse,
      { status: 404 }
    );
  }

  // Simulate progress for running jobs
  if (job.status === 'running') {
    const elapsed = Date.now() - job.startTime.getTime();
    const estimated = job.estimatedEndTime.getTime() - job.startTime.getTime();
    job.progress = Math.min(95, Math.round((elapsed / estimated) * 100));
  }

  return NextResponse.json({
    trainingId: job.id,
    status: job.status,
    progress: job.progress,
    startTime: job.startTime.toISOString(),
    estimatedEndTime: job.estimatedEndTime.toISOString(),
    config: job.config
  });
}

// Handle unsupported methods
export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint supports POST (create training job) and GET (check status) requests'
    } as ErrorResponse,
    { status: 405 }
  );
}

export async function DELETE() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint supports POST (create training job) and GET (check status) requests'
    } as ErrorResponse,
    { status: 405 }
  );
}