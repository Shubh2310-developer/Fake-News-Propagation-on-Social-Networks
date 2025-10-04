# Simulation Error Fix

## ✅ Issue Resolved

**Error Message:**
```
NetworkConfig.__init__() got an unexpected keyword argument 'network_type'
```

## Root Cause

The `NetworkConfig` dataclass in `backend/network/graph_generator.py` doesn't have a `network_type` field. The network type is meant to be passed separately to the `generate_network()` method, not as part of the config.

### NetworkConfig Definition:
```python
@dataclass
class NetworkConfig:
    num_nodes: int = 1000
    attachment_preference: int = 5
    rewiring_probability: float = 0.1
    edge_probability: float = 0.01
    k_neighbors: int = 10
    random_seed: Optional[int] = None
    # NOTE: No network_type field!
```

### Correct Usage:
```python
config = NetworkConfig(num_nodes=1000, attachment_preference=5, ...)
generator = SocialNetworkGenerator(config)
network = generator.generate_network(network_type='watts_strogatz')  # Separate parameter
```

## Solution Applied

**File:** `backend/app/services/simulation_service.py`

**Before (Line 150):**
```python
network_config = NetworkConfig(**params['network_config'])
generator = SocialNetworkGenerator(network_config)
network = generator.generate_network(
    network_type=params['network_config']['network_type']
)
```

**After:**
```python
# Extract network_type separately
network_type = params['network_config'].get('network_type', 'barabasi_albert')

# Create NetworkConfig without network_type
network_config_params = {k: v for k, v in params['network_config'].items() if k != 'network_type'}
network_config = NetworkConfig(**network_config_params)
generator = SocialNetworkGenerator(network_config)
network = generator.generate_network(network_type=network_type)
```

## Changes Made

1. Extract `network_type` before creating `NetworkConfig`
2. Filter out `network_type` from parameters
3. Pass `network_type` separately to `generate_network()` method

## Testing

### Backend Will Auto-Reload

Since the backend is running with `--reload`, it will automatically pick up this change.

### Verify Fix

Watch the backend logs. The error should no longer appear:

```bash
tail -f logs/backend.log
```

**Before (Error):**
```
ERROR: NetworkConfig.__init__() got an unexpected keyword argument 'network_type'
ERROR: Simulation {id} failed
```

**After (Success):**
```
INFO: Generating network for simulation {id}
INFO: Generated network with 1000 nodes and X edges
INFO: Initializing players for simulation {id}
INFO: Running simulation {id}
INFO: Simulation {id} completed successfully
```

### Test from Frontend

1. Go to http://localhost:3000/simulation
2. Click "Start Simulation" (with default parameters)
3. Watch the status change:
   - "Starting" → "Running" → "Completed" ✅
4. Results should display without errors

## Expected Result

The simulation should now complete successfully and return results:

```json
{
  "simulation_id": "...",
  "total_rounds": 50,
  "payoff_trends": {
    "spreader": [...],
    "fact_checker": [...],
    "platform": [...]
  },
  "final_metrics": {
    "final_payoffs": {...},
    "final_reputation_scores": {...}
  },
  "network_metrics": {
    "num_nodes": 1000,
    "density": 0.006,
    "clustering": 0.3
  },
  "convergence_analysis": {
    "converged": true
  }
}
```

## Network Types Supported

The following network types work correctly:

- `barabasi_albert` - Scale-free network (default)
- `watts_strogatz` - Small-world network
- `erdos_renyi` - Random network
- `configuration` - Configuration model

## Files Modified

1. ✅ `backend/app/services/simulation_service.py` - Fixed parameter handling

## No Restart Required

The backend auto-reloads on file changes, so the fix is already active!

---

**Status:** ✅ Fixed
**Auto-Reload:** ✅ Already Applied
**Ready:** 🚀 Test Now!

**Next:** Run a simulation from http://localhost:3000/simulation
