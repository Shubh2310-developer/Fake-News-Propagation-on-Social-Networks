// frontend/src/app/api/classifier/predict/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Types for request/response
interface PredictRequest {
  text: string;
  modelVersion?: string;
  includeFeatures?: boolean;
}

interface PredictResponse {
  prediction: 'Real' | 'Fake';
  confidence: number;
  model_used: string;
  processing_time_ms: number;
  features?: {
    word_count: number;
    sentiment_score: number;
    readability_score: number;
    clickbait_indicators: string[];
  };
  timestamp: string;
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
}

// Mock ML model simulation
const simulateModelPrediction = async (text: string): Promise<{
  prediction: 'Real' | 'Fake';
  confidence: number;
  features: any;
}> => {
  // Simulate processing delay
  await new Promise(resolve => setTimeout(resolve, Math.random() * 500 + 200));

  // Simple heuristic-based fake prediction for demonstration
  const suspiciousKeywords = [
    'breaking', 'shocking', 'exclusive', 'must see', 'you won\'t believe',
    'doctors hate', 'secret', 'exposed', 'leaked', 'urgent'
  ];

  const clickbaitPatterns = [
    /\d+\s+(things|ways|reasons|secrets)/i,
    /this\s+\w+\s+will\s+\w+\s+you/i,
    /what\s+happened\s+next\s+will/i
  ];

  const textLower = text.toLowerCase();
  const wordCount = text.split(/\s+/).length;

  // Calculate suspicion score
  let suspicionScore = 0;

  // Check for suspicious keywords
  const foundKeywords = suspiciousKeywords.filter(keyword =>
    textLower.includes(keyword)
  );
  suspicionScore += foundKeywords.length * 0.15;

  // Check for clickbait patterns
  const foundPatterns = clickbaitPatterns.filter(pattern =>
    pattern.test(text)
  );
  suspicionScore += foundPatterns.length * 0.2;

  // Check text characteristics
  if (wordCount < 20) suspicionScore += 0.1; // Too short
  if (wordCount > 500) suspicionScore += 0.05; // Very long
  if (text.split('!').length > 3) suspicionScore += 0.1; // Too many exclamations
  if (text.toUpperCase() === text && text.length > 10) suspicionScore += 0.2; // All caps

  // Simulate sentiment analysis
  const sentimentScore = Math.random() * 2 - 1; // Random between -1 and 1
  if (Math.abs(sentimentScore) > 0.7) suspicionScore += 0.1; // Extreme sentiment

  // Calculate readability (simplified)
  const avgWordsPerSentence = wordCount / (text.split(/[.!?]+/).length || 1);
  const readabilityScore = Math.max(0, Math.min(100, 100 - avgWordsPerSentence * 2));

  // Add some randomness to make it realistic
  suspicionScore += (Math.random() - 0.5) * 0.2;

  // Determine prediction
  const isFake = suspicionScore > 0.4;
  const confidence = Math.min(0.99, Math.max(0.51,
    isFake ? 0.5 + suspicionScore : 0.9 - suspicionScore
  ));

  return {
    prediction: isFake ? 'Fake' : 'Real',
    confidence: Math.round(confidence * 100) / 100,
    features: {
      word_count: wordCount,
      sentiment_score: Math.round(sentimentScore * 100) / 100,
      readability_score: Math.round(readabilityScore),
      clickbait_indicators: [...foundKeywords, ...foundPatterns.map(() => 'pattern_match')]
    }
  };
};

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Parse request body
    let body: PredictRequest;

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
    if (!body.text) {
      return NextResponse.json(
        {
          error: 'Missing required field: text',
          code: 'MISSING_TEXT',
          details: 'The text field is required and cannot be empty'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (typeof body.text !== 'string') {
      return NextResponse.json(
        {
          error: 'Invalid field type: text must be a string',
          code: 'INVALID_TEXT_TYPE',
          details: 'The text field must be a string'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate text content
    const trimmedText = body.text.trim();
    if (trimmedText.length === 0) {
      return NextResponse.json(
        {
          error: 'Empty text field',
          code: 'EMPTY_TEXT',
          details: 'The text field cannot be empty or contain only whitespace'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (trimmedText.length > 10000) {
      return NextResponse.json(
        {
          error: 'Text too long',
          code: 'TEXT_TOO_LONG',
          details: 'Text must be 10,000 characters or less'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate optional parameters
    const modelVersion = body.modelVersion || 'bert-large-v2.1';
    const includeFeatures = body.includeFeatures ?? true;

    // Simulate model prediction
    const prediction = await simulateModelPrediction(trimmedText);

    const processingTime = Date.now() - startTime;

    // Prepare response
    const response: PredictResponse = {
      prediction: prediction.prediction,
      confidence: prediction.confidence,
      model_used: modelVersion,
      processing_time_ms: processingTime,
      timestamp: new Date().toISOString(),
      ...(includeFeatures && { features: prediction.features })
    };

    // Log successful prediction
    console.log(`[CLASSIFIER] Prediction completed in ${processingTime}ms: ${prediction.prediction} (${prediction.confidence})`);

    return NextResponse.json(response, { status: 200 });

  } catch (error) {
    console.error('[CLASSIFIER] Prediction error:', error);

    // Handle specific error types
    if (error instanceof SyntaxError) {
      return NextResponse.json(
        {
          error: 'Request parsing failed',
          code: 'PARSE_ERROR',
          details: 'Could not parse request body'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Generic server error
    return NextResponse.json(
      {
        error: 'Internal server error during prediction',
        code: 'PREDICTION_FAILED',
        details: 'An unexpected error occurred while processing the prediction'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// Handle unsupported methods
export async function GET() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests'
    } as ErrorResponse,
    { status: 405 }
  );
}

export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests'
    } as ErrorResponse,
    { status: 405 }
  );
}

export async function DELETE() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests'
    } as ErrorResponse,
    { status: 405 }
  );
}