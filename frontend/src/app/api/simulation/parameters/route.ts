// frontend/src/app/api/simulation/parameters/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Types for simulation parameters
interface SimulationParameters {
  // Network Configuration
  networkSize: number;
  networkType: 'scale-free' | 'small-world' | 'random' | 'grid';
  averageDegree: number;
  clusteringCoefficient: number;

  // Agent Configuration
  spreaderRatio: number;
  moderatorRatio: number;
  userRatio: number;
  botRatio: number;

  // Propagation Parameters
  basePropagationRate: number;
  decayRate: number;
  recoveryRate: number;
  immunityRate: number;

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
  enableNetworkEvolution: boolean;
  saveFrequency: number;

  // Advanced Settings
  noiseLevel: number;
  memoryLength: number;
  explorationRate: number;
  convergenceThreshold: number;
}

interface ParameterPreset {
  id: string;
  name: string;
  description: string;
  parameters: SimulationParameters;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  isDefault?: boolean;
  author?: string;
}

interface SaveParametersRequest {
  name: string;
  description?: string;
  parameters: SimulationParameters;
  tags?: string[];
  isDefault?: boolean;
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
}

// Default simulation parameters
const DEFAULT_PARAMETERS: SimulationParameters = {
  // Network Configuration
  networkSize: 1000,
  networkType: 'scale-free',
  averageDegree: 6,
  clusteringCoefficient: 0.3,

  // Agent Configuration
  spreaderRatio: 0.05,
  moderatorRatio: 0.1,
  userRatio: 0.8,
  botRatio: 0.05,

  // Propagation Parameters
  basePropagationRate: 0.1,
  decayRate: 0.05,
  recoveryRate: 0.08,
  immunityRate: 0.02,

  // Game Theory Parameters
  spreaderReward: 2.0,
  moderatorReward: 1.5,
  detectionPenalty: -1.0,
  falsePositivePenalty: -0.5,
  learningRate: 0.1,
  adaptationFrequency: 5,

  // Simulation Settings
  timeHorizon: 100,
  enableLearning: true,
  enableNetworkEvolution: false,
  saveFrequency: 10,

  // Advanced Settings
  noiseLevel: 0.1,
  memoryLength: 10,
  explorationRate: 0.1,
  convergenceThreshold: 0.001,
};

// Mock preset storage (in real app, this would be a database)
const parameterPresets = new Map<string, ParameterPreset>([
  ['default', {
    id: 'default',
    name: 'Default Parameters',
    description: 'Standard simulation parameters for general misinformation propagation studies',
    parameters: DEFAULT_PARAMETERS,
    tags: ['default', 'general'],
    createdAt: '2024-09-01T00:00:00Z',
    updatedAt: '2024-09-25T00:00:00Z',
    isDefault: true,
    author: 'system'
  }],
  ['high-density', {
    id: 'high-density',
    name: 'High Density Network',
    description: 'Parameters optimized for dense network topologies with rapid information spread',
    parameters: {
      ...DEFAULT_PARAMETERS,
      networkSize: 2000,
      averageDegree: 12,
      clusteringCoefficient: 0.6,
      basePropagationRate: 0.15,
      spreaderRatio: 0.03,
      moderatorRatio: 0.15,
      userRatio: 0.82,
    },
    tags: ['dense-network', 'rapid-spread'],
    createdAt: '2024-09-10T00:00:00Z',
    updatedAt: '2024-09-20T00:00:00Z',
    author: 'researcher'
  }],
  ['low-moderation', {
    id: 'low-moderation',
    name: 'Low Moderation Scenario',
    description: 'Simulates environments with minimal content moderation',
    parameters: {
      ...DEFAULT_PARAMETERS,
      spreaderRatio: 0.08,
      moderatorRatio: 0.02,
      userRatio: 0.9,
      moderatorReward: 1.0,
      detectionPenalty: -0.5,
    },
    tags: ['low-moderation', 'unregulated'],
    createdAt: '2024-09-15T00:00:00Z',
    updatedAt: '2024-09-22T00:00:00Z',
    author: 'researcher'
  }],
  ['ml-enhanced', {
    id: 'ml-enhanced',
    name: 'ML-Enhanced Detection',
    description: 'Parameters for scenarios with machine learning-powered content detection',
    parameters: {
      ...DEFAULT_PARAMETERS,
      moderatorRatio: 0.05,
      moderatorReward: 2.5,
      falsePositivePenalty: -0.2,
      enableLearning: true,
      learningRate: 0.15,
      adaptationFrequency: 3,
    },
    tags: ['ml-detection', 'automated-moderation'],
    createdAt: '2024-09-18T00:00:00Z',
    updatedAt: '2024-09-24T00:00:00Z',
    author: 'researcher'
  }]
]);

// Validate parameters
const validateParameters = (params: Partial<SimulationParameters>): string[] => {
  const errors: string[] = [];

  if (params.networkSize !== undefined && (params.networkSize < 10 || params.networkSize > 100000)) {
    errors.push('Network size must be between 10 and 100,000');
  }

  if (params.averageDegree !== undefined && params.networkSize !== undefined &&
      (params.averageDegree < 1 || params.averageDegree > params.networkSize / 2)) {
    errors.push('Average degree must be between 1 and half the network size');
  }

  // Check agent ratios sum to 1.0
  const spreaderRatio = params.spreaderRatio ?? DEFAULT_PARAMETERS.spreaderRatio;
  const moderatorRatio = params.moderatorRatio ?? DEFAULT_PARAMETERS.moderatorRatio;
  const userRatio = params.userRatio ?? DEFAULT_PARAMETERS.userRatio;
  const botRatio = params.botRatio ?? DEFAULT_PARAMETERS.botRatio;

  const totalRatio = spreaderRatio + moderatorRatio + userRatio + botRatio;
  if (Math.abs(totalRatio - 1.0) > 0.001) {
    errors.push(`Agent ratios must sum to 1.0 (current sum: ${totalRatio.toFixed(3)})`);
  }

  // Validate individual ranges
  const validationRules = [
    { field: 'spreaderRatio', min: 0, max: 0.5 },
    { field: 'moderatorRatio', min: 0, max: 0.5 },
    { field: 'userRatio', min: 0, max: 1 },
    { field: 'botRatio', min: 0, max: 0.3 },
    { field: 'basePropagationRate', min: 0, max: 1 },
    { field: 'decayRate', min: 0, max: 1 },
    { field: 'recoveryRate', min: 0, max: 1 },
    { field: 'immunityRate', min: 0, max: 1 },
    { field: 'learningRate', min: 0, max: 1 },
    { field: 'noiseLevel', min: 0, max: 1 },
    { field: 'explorationRate', min: 0, max: 1 },
    { field: 'convergenceThreshold', min: 0.0001, max: 0.1 },
    { field: 'clusteringCoefficient', min: 0, max: 1 },
  ];

  validationRules.forEach(rule => {
    const value = params[rule.field as keyof SimulationParameters];
    if (typeof value === 'number' && (value < rule.min || value > rule.max)) {
      errors.push(`${rule.field} must be between ${rule.min} and ${rule.max}`);
    }
  });

  return errors;
};

// GET: Retrieve simulation parameters
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const presetId = searchParams.get('preset');
    const includePresets = searchParams.get('include_presets') !== 'false';

    if (presetId) {
      // Get specific preset
      const preset = parameterPresets.get(presetId);

      if (!preset) {
        return NextResponse.json(
          {
            error: 'Preset not found',
            code: 'PRESET_NOT_FOUND',
            details: `No preset found with ID: ${presetId}`
          } as ErrorResponse,
          { status: 404 }
        );
      }

      return NextResponse.json(preset);
    }

    if (includePresets) {
      // Get all presets
      const presets = Array.from(parameterPresets.values())
        .sort((a, b) => {
          if (a.isDefault) return -1;
          if (b.isDefault) return 1;
          return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
        });

      return NextResponse.json({
        defaultParameters: DEFAULT_PARAMETERS,
        presets,
        metadata: {
          totalPresets: presets.length,
          retrievedAt: new Date().toISOString()
        }
      });
    }

    // Get just default parameters
    return NextResponse.json({
      parameters: DEFAULT_PARAMETERS,
      metadata: {
        type: 'default',
        retrievedAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('[SIMULATION] Parameters retrieval error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during parameters retrieval',
        code: 'PARAMETERS_RETRIEVAL_FAILED',
        details: 'An unexpected error occurred while retrieving simulation parameters'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// POST: Save simulation parameters as preset
export async function POST(request: NextRequest) {
  try {
    let body: SaveParametersRequest;

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
    if (!body.name) {
      return NextResponse.json(
        {
          error: 'Missing required field: name',
          code: 'MISSING_NAME',
          details: 'The name field is required'
        } as ErrorResponse,
        { status: 400 }
      );
    }

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

    // Validate parameters
    const validationErrors = validateParameters(body.parameters);
    if (validationErrors.length > 0) {
      return NextResponse.json(
        {
          error: 'Invalid parameters',
          code: 'INVALID_PARAMETERS',
          details: 'One or more parameters are invalid',
          validationErrors
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Generate unique ID
    const presetId = `preset_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

    // Check if trying to set as default
    if (body.isDefault) {
      // Remove default flag from existing presets
      parameterPresets.forEach(preset => {
        if (preset.isDefault) {
          preset.isDefault = false;
          preset.updatedAt = new Date().toISOString();
        }
      });
    }

    // Create new preset
    const newPreset: ParameterPreset = {
      id: presetId,
      name: body.name,
      description: body.description || '',
      parameters: body.parameters,
      tags: body.tags || [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isDefault: body.isDefault || false,
      author: 'user' // In real app, this would come from authentication
    };

    // Save preset
    parameterPresets.set(presetId, newPreset);

    console.log(`[SIMULATION] Saved parameter preset: ${body.name} (${presetId})`);

    return NextResponse.json({
      message: 'Parameters saved successfully',
      preset: newPreset
    }, { status: 201 });

  } catch (error) {
    console.error('[SIMULATION] Parameters save error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during parameters save',
        code: 'PARAMETERS_SAVE_FAILED',
        details: 'An unexpected error occurred while saving simulation parameters'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// PUT: Update existing preset
export async function PUT(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const presetId = searchParams.get('id');

    if (!presetId) {
      return NextResponse.json(
        {
          error: 'Missing preset ID',
          code: 'MISSING_PRESET_ID',
          details: 'Preset ID is required as a query parameter'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    const existingPreset = parameterPresets.get(presetId);
    if (!existingPreset) {
      return NextResponse.json(
        {
          error: 'Preset not found',
          code: 'PRESET_NOT_FOUND',
          details: `No preset found with ID: ${presetId}`
        } as ErrorResponse,
        { status: 404 }
      );
    }

    let body: Partial<SaveParametersRequest>;

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

    // Validate parameters if provided
    if (body.parameters) {
      const validationErrors = validateParameters(body.parameters);
      if (validationErrors.length > 0) {
        return NextResponse.json(
          {
            error: 'Invalid parameters',
            code: 'INVALID_PARAMETERS',
            details: 'One or more parameters are invalid',
            validationErrors
          } as ErrorResponse,
          { status: 400 }
        );
      }
    }

    // Update preset
    const updatedPreset: ParameterPreset = {
      ...existingPreset,
      ...(body.name && { name: body.name }),
      ...(body.description !== undefined && { description: body.description }),
      ...(body.parameters && { parameters: body.parameters }),
      ...(body.tags && { tags: body.tags }),
      updatedAt: new Date().toISOString()
    };

    // Handle default flag
    if (body.isDefault !== undefined) {
      if (body.isDefault) {
        // Remove default from other presets
        parameterPresets.forEach(preset => {
          if (preset.id !== presetId && preset.isDefault) {
            preset.isDefault = false;
            preset.updatedAt = new Date().toISOString();
          }
        });
      }
      updatedPreset.isDefault = body.isDefault;
    }

    parameterPresets.set(presetId, updatedPreset);

    console.log(`[SIMULATION] Updated parameter preset: ${updatedPreset.name} (${presetId})`);

    return NextResponse.json({
      message: 'Preset updated successfully',
      preset: updatedPreset
    });

  } catch (error) {
    console.error('[SIMULATION] Parameters update error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during parameters update',
        code: 'PARAMETERS_UPDATE_FAILED',
        details: 'An unexpected error occurred while updating simulation parameters'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// DELETE: Remove preset
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const presetId = searchParams.get('id');

    if (!presetId) {
      return NextResponse.json(
        {
          error: 'Missing preset ID',
          code: 'MISSING_PRESET_ID',
          details: 'Preset ID is required as a query parameter'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    const preset = parameterPresets.get(presetId);
    if (!preset) {
      return NextResponse.json(
        {
          error: 'Preset not found',
          code: 'PRESET_NOT_FOUND',
          details: `No preset found with ID: ${presetId}`
        } as ErrorResponse,
        { status: 404 }
      );
    }

    // Prevent deletion of default preset
    if (preset.id === 'default') {
      return NextResponse.json(
        {
          error: 'Cannot delete default preset',
          code: 'CANNOT_DELETE_DEFAULT',
          details: 'The default preset cannot be deleted'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    parameterPresets.delete(presetId);

    console.log(`[SIMULATION] Deleted parameter preset: ${preset.name} (${presetId})`);

    return NextResponse.json({
      message: 'Preset deleted successfully',
      deletedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('[SIMULATION] Parameters deletion error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during parameters deletion',
        code: 'PARAMETERS_DELETE_FAILED',
        details: 'An unexpected error occurred while deleting simulation parameters'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}