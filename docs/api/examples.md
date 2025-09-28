# API Usage Examples

This document provides practical, copy-paste-ready examples that show developers how to use the API to perform common tasks.

## Authentication Setup

### Python (requests)

```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000/api"
API_TOKEN = "your-jwt-token-here"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def make_request(method, endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    response = requests.request(method, url, headers=headers, json=data)
    return response.json()
```

### JavaScript (fetch)

```javascript
// Configuration
const BASE_URL = "http://localhost:8000/api";
const API_TOKEN = "your-jwt-token-here";

const headers = {
    "Authorization": `Bearer ${API_TOKEN}`,
    "Content-Type": "application/json"
};

async function makeRequest(method, endpoint, data = null) {
    const url = `${BASE_URL}${endpoint}`;
    const options = {
        method,
        headers,
        ...(data && { body: JSON.stringify(data) })
    };

    const response = await fetch(url, options);
    return response.json();
}
```

### cURL

```bash
# Set environment variables
export API_TOKEN="your-jwt-token-here"
export BASE_URL="http://localhost:8000/api"

# Helper function for authenticated requests
api_request() {
    curl -H "Authorization: Bearer $API_TOKEN" \
         -H "Content-Type: application/json" \
         "$@"
}
```

## Fake News Classification

### Single Text Classification

**Python:**
```python
# Classify a single text
text_to_classify = "Breaking: Scientists discover that vaccines contain microchips!"

response = make_request("POST", "/classifier/predict", {
    "text": text_to_classify
})

if response["success"]:
    result = response["data"]
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
else:
    print(f"Error: {response['error']['message']}")
```

**JavaScript:**
```javascript
// Classify a single text
const textToClassify = "Breaking: Scientists discover that vaccines contain microchips!";

try {
    const response = await makeRequest("POST", "/classifier/predict", {
        text: textToClassify
    });

    if (response.success) {
        const result = response.data;
        console.log(`Prediction: ${result.prediction}`);
        console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
        console.log("Probabilities:", result.probabilities);
    } else {
        console.error("Error:", response.error.message);
    }
} catch (error) {
    console.error("Request failed:", error);
}
```

**cURL:**
```bash
# Classify a single text
api_request -X POST "$BASE_URL/classifier/predict" \
    -d '{
        "text": "Breaking: Scientists discover that vaccines contain microchips!"
    }'
```

### Batch Classification

**Python:**
```python
# Classify multiple texts
texts = [
    "Scientists publish peer-reviewed study on climate change",
    "Local politician caught in scandal, refuses to comment",
    "Miracle cure discovered! Doctors hate this one simple trick!"
]

results = []
for text in texts:
    response = make_request("POST", "/classifier/predict", {"text": text})
    if response["success"]:
        results.append({
            "text": text[:50] + "...",
            "prediction": response["data"]["prediction"],
            "confidence": response["data"]["confidence"]
        })

# Display results
for result in results:
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print("-" * 60)
```

## Model Training Workflow

### 1. Start Training Job

**Python:**
```python
# Start a BERT model training job
training_config = {
    "dataset_path": "/data/training/fake_news_dataset.csv",
    "model_type": "bert",
    "experiment_name": "bert_experiment_v1",
    "hyperparameters": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.00002
    }
}

response = make_request("POST", "/classifier/train", training_config)

if response["success"]:
    job_id = response["data"]["job_id"]
    print(f"Training started! Job ID: {job_id}")
    print(f"Estimated duration: {response['data']['estimated_duration_minutes']} minutes")
else:
    print(f"Failed to start training: {response['error']['message']}")
```

### 2. Monitor Training Progress

**Python:**
```python
import time

def monitor_training(job_id):
    while True:
        # Note: This endpoint would be implemented to check job status
        response = make_request("GET", f"/jobs/{job_id}/status")

        if response["success"]:
            status = response["data"]["status"]
            print(f"Training status: {status}")

            if status == "completed":
                print("Training completed successfully!")
                break
            elif status == "failed":
                print("Training failed!")
                break

        time.sleep(30)  # Check every 30 seconds

# Monitor the training job
monitor_training(job_id)
```

### 3. Get Model Metrics

**Python:**
```python
# Get metrics for the latest models
response = make_request("GET", "/classifier/metrics?limit=5")

if response["success"]:
    models = response["data"]["models"]

    print("Recent Model Performance:")
    print("-" * 80)

    for model in models:
        print(f"Version: {model['version']}")
        print(f"Type: {model['type']}")
        print(f"Accuracy: {model['metrics']['accuracy']:.3f}")
        print(f"F1 Score: {model['metrics']['f1_score']:.3f}")
        print(f"Training Date: {model['training_date']}")
        print("-" * 40)
```

## Game Theory Simulations

### Single Simulation

**Python:**
```python
# Run a single simulation
simulation_params = {
    "parameters": {
        "network_size": 1000,
        "network_type": "scale_free",
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
    "experiment_name": "scale_free_experiment_1"
}

response = make_request("POST", "/simulation/run", simulation_params)

if response["success"]:
    sim_id = response["data"]["simulation_id"]
    print(f"Simulation started! ID: {sim_id}")
else:
    print(f"Failed to start simulation: {response['error']['message']}")
```

### Monitor Simulation and Get Results

**Python:**
```python
def wait_for_simulation(sim_id):
    while True:
        # Check simulation status
        response = make_request("GET", f"/simulation/status/{sim_id}")

        if response["success"]:
            status = response["data"]["status"]
            progress = response["data"]["progress"]

            print(f"Simulation {sim_id}: {status} ({progress}%)")

            if status == "completed":
                # Get results
                results_response = make_request("GET", f"/simulation/results/{sim_id}")

                if results_response["success"]:
                    results = results_response["data"]

                    print("\nSimulation Results:")
                    print(f"Network: {results['network_stats']['nodes']} nodes, {results['network_stats']['edges']} edges")
                    print(f"Nash Equilibria: {len(results['equilibria'])}")
                    print(f"Peak Infection: {results['propagation_metrics']['peak_infection']} nodes")
                    print(f"Final Believers: {results['propagation_metrics']['final_believers']} nodes")

                    return results
                break
            elif status == "failed":
                print("Simulation failed!")
                break

        time.sleep(5)  # Check every 5 seconds

# Wait for simulation and get results
results = wait_for_simulation(sim_id)
```

### Parameter Sweep

**Python:**
```python
# Run simulations with different detection rates
detection_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
simulation_results = []

for rate in detection_rates:
    sim_params = {
        "parameters": {
            "network_size": 500,
            "network_type": "scale_free",
            "detection_rate": rate,
            "simulation_steps": 50
        },
        "experiment_name": f"detection_rate_{rate}"
    }

    response = make_request("POST", "/simulation/run", sim_params)

    if response["success"]:
        sim_id = response["data"]["simulation_id"]
        results = wait_for_simulation(sim_id)

        simulation_results.append({
            "detection_rate": rate,
            "peak_infection": results["propagation_metrics"]["peak_infection"],
            "final_believers": results["propagation_metrics"]["final_believers"]
        })

# Analyze results
print("\nParameter Sweep Results:")
print("Detection Rate | Peak Infection | Final Believers")
print("-" * 50)
for result in simulation_results:
    print(f"{result['detection_rate']:12.1f} | {result['peak_infection']:13d} | {result['final_believers']:14d}")
```

## Error Handling Examples

### Comprehensive Error Handling

**Python:**
```python
def safe_api_request(method, endpoint, data=None, max_retries=3):
    """Make API request with proper error handling and retries"""

    for attempt in range(max_retries):
        try:
            response = make_request(method, endpoint, data)

            if response.get("success"):
                return response["data"]
            else:
                error = response.get("error", {})
                error_code = error.get("code", "UNKNOWN_ERROR")

                # Handle specific error codes
                if error_code == "RATE_LIMIT":
                    retry_after = error.get("retry_after_seconds", 60)
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                elif error_code == "VALIDATION_ERROR":
                    print(f"Validation error: {error.get('message')}")
                    return None
                else:
                    print(f"API error: {error.get('message')}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print("Max retries exceeded")
    return None

# Usage example
result = safe_api_request("POST", "/classifier/predict", {
    "text": "Sample text to classify"
})

if result:
    print(f"Classification result: {result}")
else:
    print("Failed to get classification")
```

## Integration Examples

### Web Application Integration

**JavaScript (React):**
```javascript
import React, { useState, useEffect } from 'react';

function FakeNewsChecker() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const classifyText = async () => {
        if (!text.trim()) return;

        setLoading(true);
        try {
            const response = await makeRequest('POST', '/classifier/predict', {
                text: text
            });

            if (response.success) {
                setResult(response.data);
            } else {
                console.error('Classification failed:', response.error);
            }
        } catch (error) {
            console.error('Request failed:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text to check for fake news..."
                rows={4}
                cols={50}
            />
            <br />
            <button onClick={classifyText} disabled={loading}>
                {loading ? 'Analyzing...' : 'Check for Fake News'}
            </button>

            {result && (
                <div>
                    <h3>Result:</h3>
                    <p>Prediction: <strong>{result.prediction}</strong></p>
                    <p>Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
                </div>
            )}
        </div>
    );
}

export default FakeNewsChecker;
```

### CLI Tool Example

**Python CLI:**
```python
#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection CLI')
    parser.add_argument('text', help='Text to classify')
    parser.add_argument('--token', required=True, help='API token')
    parser.add_argument('--url', default='http://localhost:8000/api', help='API base URL')

    args = parser.parse_args()

    # Setup API client
    global BASE_URL, API_TOKEN
    BASE_URL = args.url
    API_TOKEN = args.token

    # Make classification request
    response = make_request("POST", "/classifier/predict", {
        "text": args.text
    })

    if response["success"]:
        result = response["data"]
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")

        # Exit with appropriate code
        sys.exit(0 if result['prediction'] == 'real' else 1)
    else:
        print(f"Error: {response['error']['message']}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
```

Usage:
```bash
python fake_news_cli.py "This is some news text" --token your-api-token
```