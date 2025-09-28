// frontend/src/app/api/simulation/run/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Types for simulation request/response
interface SimulationRunRequest {
  name?: string;
  description?: string;
  parameters: {
    // Network Configuration
    networkSize: number;
    networkType: 'scale-free' | 'small-world' | 'random' | 'grid';
    averageDegree: number;
    clusteringCoefficient?: number;

    // Agent Configuration
    spreaderRatio: number;
    moderatorRatio: number;
    userRatio: number;
    botRatio?: number;

    // Propagation Parameters
    basePropagationRate: number;
    decayRate: number;
    recoveryRate: number;
    immunityRate?: number;

    // Game Theory Parameters
    spreaderReward: number;
    moderatorReward: number;
    detectionPenalty: number;
    falsePositivePenalty: number;
    learningRate: number;
    adaptationFrequency: number;

    // Simulation Settings
    timeHorizon: number;
    randomSeed?: number;
    enableLearning: boolean;
    enableNetworkEvolution?: boolean;
    saveFrequency?: number;

    // Advanced Settings
    noiseLevel?: number;
    memoryLength?: number;
    explorationRate?: number;
    convergenceThreshold?: number;
  };
  tags?: string[];
  priority?: 'low' | 'normal' | 'high';
}

interface SimulationRunResponse {
  simulationId: string;
  status: 'queued' | 'initializing' | 'running';
  message: string;
  estimatedDurationMinutes: number;
  queuePosition?: number;
  progress: {
    currentStep: number;
    totalSteps: number;
    percentage: number;
  };
  experiment: {
    name: string;
    description?: string;
    tags: string[];
    createdAt: string;
    priority: string;
  };
  config: {
    parameters: object;
    validatedAt: string;
  };
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
  validationErrors?: Array<{
    field: string;
    message: string;
    value?: any;
  }>;
}

// Simulation job management
const activeSimulations = new Map<string, {
  id: string;
  status: 'queued' | 'initializing' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: { currentStep: number; totalSteps: number; percentage: number };
  startTime: Date;
  estimatedEndTime: Date;
  config: any;
  results?: any;
}>();

// Validate simulation parameters
const validateSimulationParameters = (params: SimulationRunRequest['parameters']): string[] => {
  const errors: string[] = [];

  // Network validation
  if (params.networkSize < 10 || params.networkSize > 100000) {
    errors.push('Network size must be between 10 and 100,000');
  }

  if (params.averageDegree < 1 || params.averageDegree > params.networkSize / 2) {
    errors.push('Average degree must be between 1 and half the network size');
  }

  // Agent ratio validation
  const totalRatio = params.spreaderRatio + params.moderatorRatio + params.userRatio + (params.botRatio || 0);
  if (Math.abs(totalRatio - 1.0) > 0.001) {
    errors.push(`Agent ratios must sum to 1.0 (current sum: ${totalRatio.toFixed(3)})`);
  }

  if (params.spreaderRatio < 0 || params.spreaderRatio > 0.5) {
    errors.push('Spreader ratio must be between 0 and 0.5');
  }

  if (params.moderatorRatio < 0 || params.moderatorRatio > 0.5) {
    errors.push('Moderator ratio must be between 0 and 0.5');
  }

  // Propagation parameter validation
  if (params.basePropagationRate < 0 || params.basePropagationRate > 1) {
    errors.push('Base propagation rate must be between 0 and 1');
  }

  if (params.decayRate < 0 || params.decayRate > 1) {
    errors.push('Decay rate must be between 0 and 1');
  }

  if (params.recoveryRate < 0 || params.recoveryRate > 1) {
    errors.push('Recovery rate must be between 0 and 1');
  }

  // Game theory parameter validation
  if (params.learningRate < 0 || params.learningRate > 1) {
    errors.push('Learning rate must be between 0 and 1');
  }

  if (params.adaptationFrequency < 1 || params.adaptationFrequency > 100) {
    errors.push('Adaptation frequency must be between 1 and 100');
  }

  // Simulation settings validation
  if (params.timeHorizon < 10 || params.timeHorizon > 10000) {
    errors.push('Time horizon must be between 10 and 10,000');
  }

  // Optional parameter validation
  if (params.clusteringCoefficient !== undefined && (params.clusteringCoefficient < 0 || params.clusteringCoefficient > 1)) {
    errors.push('Clustering coefficient must be between 0 and 1');
  }

  if (params.noiseLevel !== undefined && (params.noiseLevel < 0 || params.noiseLevel > 1)) {
    errors.push('Noise level must be between 0 and 1');
  }

  return errors;
};

// Generate unique simulation ID
const generateSimulationId = (): string => {
  const timestamp = Date.now().toString(36);
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  return `sim_${timestamp}_${randomSuffix}`;
};

// Estimate simulation duration
const estimateSimulationDuration = (params: SimulationRunRequest['parameters']): number => {
  // Base calculation: network size and time horizon are primary factors
  const baseTime = (params.networkSize * params.timeHorizon) / 10000; // minutes

  // Complexity modifiers
  let complexityMultiplier = 1.0;

  if (params.enableLearning) complexityMultiplier *= 1.5;
  if (params.enableNetworkEvolution) complexityMultiplier *= 1.3;
  if (params.networkType === 'scale-free') complexityMultiplier *= 1.2;

  // Agent complexity
  const agentComplexity = 1 + (params.spreaderRatio + params.moderatorRatio) * 0.5;

  const totalTime = baseTime * complexityMultiplier * agentComplexity;

  // Add overhead and ensure minimum time
  return Math.max(1, Math.round(totalTime * 1.2));
};

export async function POST(request: NextRequest) {
  try {
    // Parse request body
    let body: SimulationRunRequest;

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
    if (!body.parameters) {
      return NextResponse.json(
        {
          error: 'Missing required field: parameters',
          code: 'MISSING_PARAMETERS',
          details: 'The parameters field is required'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate simulation parameters
    const validationErrors = validateSimulationParameters(body.parameters);

    if (validationErrors.length > 0) {
      return NextResponse.json(
        {
          error: 'Invalid simulation parameters',
          code: 'INVALID_PARAMETERS',
          details: 'One or more simulation parameters are invalid',
          validationErrors: validationErrors.map(error => ({
            field: 'parameters',
            message: error
          }))
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Check resource limits
    const activeRunningSimulations = Array.from(activeSimulations.values())
      .filter(sim => sim.status === 'running').length;

    if (activeRunningSimulations >= 5) {
      return NextResponse.json(
        {
          error: 'Simulation queue full',
          code: 'QUEUE_FULL',
          details: 'Maximum number of concurrent simulations reached. Please try again later.'
        } as ErrorResponse,
        { status: 503 }
      );
    }

    // Generate simulation job
    const simulationId = generateSimulationId();
    const estimatedDuration = estimateSimulationDuration(body.parameters);
    const experimentName = body.name || `Simulation_${new Date().toISOString().split('T')[0]}`;
    const priority = body.priority || 'normal';

    // Set default optional parameters
    const finalParameters = {
      ...body.parameters,
      clusteringCoefficient: body.parameters.clusteringCoefficient ?? 0.3,
      botRatio: body.parameters.botRatio ?? 0.0,
      immunityRate: body.parameters.immunityRate ?? 0.02,
      enableNetworkEvolution: body.parameters.enableNetworkEvolution ?? false,
      saveFrequency: body.parameters.saveFrequency ?? 10,
      noiseLevel: body.parameters.noiseLevel ?? 0.1,
      memoryLength: body.parameters.memoryLength ?? 10,
      explorationRate: body.parameters.explorationRate ?? 0.1,
      convergenceThreshold: body.parameters.convergenceThreshold ?? 0.001,
    };

    // Create simulation job
    const simulationJob = {
      id: simulationId,
      status: 'queued' as const,
      progress: {
        currentStep: 0,
        totalSteps: body.parameters.timeHorizon,
        percentage: 0
      },
      startTime: new Date(),
      estimatedEndTime: new Date(Date.now() + estimatedDuration * 60 * 1000),
      config: {
        parameters: finalParameters,
        experimentName,
        description: body.description,
        tags: body.tags || [],
        priority
      }
    };

    activeSimulations.set(simulationId, simulationJob);

    // Calculate queue position
    const queuedSimulations = Array.from(activeSimulations.values())
      .filter(sim => sim.status === 'queued')
      .sort((a, b) => {
        // Priority ordering: high > normal > low
        const priorityOrder = { high: 3, normal: 2, low: 1 };
        return priorityOrder[b.config.priority as keyof typeof priorityOrder] -
               priorityOrder[a.config.priority as keyof typeof priorityOrder];
      });

    const queuePosition = queuedSimulations.findIndex(sim => sim.id === simulationId) + 1;

    // Prepare response
    const response: SimulationRunResponse = {
      simulationId,
      status: 'queued',
      message: 'Simulation has been queued successfully',
      estimatedDurationMinutes: estimatedDuration,
      queuePosition,
      progress: {
        currentStep: 0,
        totalSteps: body.parameters.timeHorizon,
        percentage: 0
      },
      experiment: {
        name: experimentName,
        description: body.description,
        tags: body.tags || [],
        createdAt: new Date().toISOString(),
        priority
      },
      config: {
        parameters: finalParameters,
        validatedAt: new Date().toISOString()
      }
    };

    // Log simulation creation
    console.log(`[SIMULATION] Created simulation ${simulationId} with ${body.parameters.networkSize} nodes`);

    // Simulate async processing
    setTimeout(() => {
      const job = activeSimulations.get(simulationId);
      if (job) {
        job.status = 'initializing';
        activeSimulations.set(simulationId, job);

        setTimeout(() => {
          const runningJob = activeSimulations.get(simulationId);
          if (runningJob) {
            runningJob.status = 'running';
            activeSimulations.set(simulationId, runningJob);

            // Simulate progress updates
            const progressInterval = setInterval(() => {
              const progressJob = activeSimulations.get(simulationId);
              if (progressJob && progressJob.status === 'running') {
                progressJob.progress.currentStep += Math.floor(Math.random() * 5) + 1;
                progressJob.progress.percentage = Math.min(100,
                  (progressJob.progress.currentStep / progressJob.progress.totalSteps) * 100);

                if (progressJob.progress.currentStep >= progressJob.progress.totalSteps) {
                  progressJob.status = 'completed';
                  clearInterval(progressInterval);
                }

                activeSimulations.set(simulationId, progressJob);
              } else {
                clearInterval(progressInterval);
              }
            }, 1000);
          }
        }, 3000);
      }
    }, 1000);

    return NextResponse.json(response, { status: 202 });

  } catch (error) {
    console.error('[SIMULATION] Run initiation error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during simulation initiation',
        code: 'SIMULATION_INIT_FAILED',
        details: 'An unexpected error occurred while initiating the simulation'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// Get simulation status
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const simulationId = searchParams.get('id');

  if (!simulationId) {
    return NextResponse.json(
      {
        error: 'Missing simulation ID',
        code: 'MISSING_SIMULATION_ID',
        details: 'Simulation ID is required as a query parameter'
      } as ErrorResponse,
      { status: 400 }
    );
  }

  const simulation = activeSimulations.get(simulationId);

  if (!simulation) {
    return NextResponse.json(
      {
        error: 'Simulation not found',
        code: 'SIMULATION_NOT_FOUND',
        details: `No simulation found with ID: ${simulationId}`
      } as ErrorResponse,
      { status: 404 }
    );
  }

  return NextResponse.json({
    simulationId: simulation.id,
    status: simulation.status,
    progress: simulation.progress,
    startTime: simulation.startTime.toISOString(),
    estimatedEndTime: simulation.estimatedEndTime.toISOString(),
    config: simulation.config,
    ...(simulation.results && { results: simulation.results })
  });
}

// Cancel simulation
export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const simulationId = searchParams.get('id');

  if (!simulationId) {
    return NextResponse.json(
      {
        error: 'Missing simulation ID',
        code: 'MISSING_SIMULATION_ID',
        details: 'Simulation ID is required as a query parameter'
      } as ErrorResponse,
      { status: 400 }
    );
  }

  const simulation = activeSimulations.get(simulationId);

  if (!simulation) {
    return NextResponse.json(
      {
        error: 'Simulation not found',
        code: 'SIMULATION_NOT_FOUND',
        details: `No simulation found with ID: ${simulationId}`
      } as ErrorResponse,
      { status: 404 }
    );
  }

  if (simulation.status === 'completed') {
    return NextResponse.json(
      {
        error: 'Cannot cancel completed simulation',
        code: 'SIMULATION_ALREADY_COMPLETED',
        details: 'This simulation has already completed and cannot be cancelled'
      } as ErrorResponse,
      { status: 400 }
    );
  }

  // Cancel the simulation
  simulation.status = 'cancelled';
  activeSimulations.set(simulationId, simulation);

  console.log(`[SIMULATION] Cancelled simulation ${simulationId}`);

  return NextResponse.json({
    message: 'Simulation cancelled successfully',
    simulationId,
    cancelledAt: new Date().toISOString()
  });
}

// Handle unsupported methods
export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint supports POST (run simulation), GET (check status), and DELETE (cancel) requests'
    } as ErrorResponse,
    { status: 405 }
  );
}