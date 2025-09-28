# API Reference

This document provides a complete technical blueprint for developers who need to programmatically interact with the backend API.

## Authentication

All API endpoints require authentication using JWT Bearer tokens.

```
Authorization: Bearer <your-jwt-token>
```

## Base URL

```
http://localhost:8000/api
```

## Response Format

All responses follow a consistent JSON structure:

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Classifier Endpoints

### POST /api/classifier/predict

Predict whether a given text is fake news or not.

**Request Body:**
```json
{
  "text": "string (required) - The text to classify",
  "model_version": "string (optional) - Specific model version to use"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": "fake|real",
    "confidence": 0.85,
    "probabilities": {
      "fake": 0.85,
      "real": 0.15
    },
    "model_version": "v1.2.0",
    "processing_time_ms": 245
  }
}
```

**Error Codes:**
- `400` - Invalid input (missing text)
- `404` - Model not found
- `500` - Internal server error

### POST /api/classifier/train

Start a new model training job.

**Request Body:**
```json
{
  "dataset_path": "string (required) - Path to training dataset",
  "model_type": "bert|ensemble (required)",
  "experiment_name": "string (optional)",
  "hyperparameters": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.00002
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "train_123456789",
    "status": "started",
    "estimated_duration_minutes": 120,
    "experiment_name": "exp_20240101_120000"
  }
}
```

### GET /api/classifier/metrics

Get performance metrics for trained models.

**Query Parameters:**
- `model_version` (optional) - Specific model version
- `limit` (optional, default: 10) - Number of results to return

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "version": "v1.2.0",
        "type": "bert",
        "metrics": {
          "accuracy": 0.92,
          "precision": 0.91,
          "recall": 0.93,
          "f1_score": 0.92
        },
        "training_date": "2024-01-01T12:00:00Z",
        "dataset_size": 50000
      }
    ]
  }
}
```

## Simulation Endpoints

### POST /api/simulation/run

Start a new game theory simulation.

**Request Body:**
```json
{
  "parameters": {
    "network_size": 1000,
    "network_type": "scale_free|erdos_renyi|small_world|regular",
    "detection_rate": 0.7,
    "simulation_steps": 100,
    "payoff_matrix": {
      "spreader": {
        "aggressive": [2, 0],
        "moderate": [1, 1]
      },
      "fact_checker": {
        "active": [1, 3],
        "passive": [0, 2]
      }
    }
  },
  "experiment_name": "string (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_123456789",
    "status": "running",
    "estimated_duration_seconds": 30,
    "parameters": {...}
  }
}
```

### GET /api/simulation/status/{simulation_id}

Get the status of a running simulation.

**Response:**
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_123456789",
    "status": "completed|running|failed",
    "progress": 100,
    "start_time": "2024-01-01T12:00:00Z",
    "end_time": "2024-01-01T12:00:30Z",
    "results_available": true
  }
}
```

### GET /api/simulation/results/{simulation_id}

Get results from a completed simulation.

**Response:**
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_123456789",
    "network_stats": {
      "nodes": 1000,
      "edges": 2500,
      "average_degree": 5.0
    },
    "equilibria": [
      {
        "strategies": [0, 1],
        "payoffs": {
          "spreader": 0.5,
          "fact_checker": 2.8
        },
        "type": "pure",
        "stability": 0.85
      }
    ],
    "propagation_metrics": {
      "peak_infection": 234,
      "final_believers": 89,
      "cascade_size": 456
    },
    "time_series": [
      {
        "step": 0,
        "infected": 1,
        "believers": 1,
        "fact_checkers_active": 100
      }
    ]
  }
}
```

### GET /api/simulation/parameters

Get available simulation parameters and their valid ranges.

**Response:**
```json
{
  "success": true,
  "data": {
    "network_types": ["scale_free", "erdos_renyi", "small_world", "regular"],
    "network_size": {
      "min": 10,
      "max": 10000,
      "default": 1000
    },
    "detection_rate": {
      "min": 0.0,
      "max": 1.0,
      "default": 0.7
    },
    "simulation_steps": {
      "min": 1,
      "max": 1000,
      "default": 100
    }
  }
}
```

## Health and Status

### GET /api/health

Check API health status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "database": "connected",
    "redis": "connected"
  }
}
```

## Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | VALIDATION_ERROR | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource already exists |
| 429 | RATE_LIMIT | Too many requests |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Service temporarily unavailable |

## Rate Limiting

API requests are limited to 100 requests per minute per API key. When rate limit is exceeded, the API returns HTTP 429 with:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT",
    "message": "Rate limit exceeded",
    "retry_after_seconds": 60
  }
}
```