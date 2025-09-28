# Running Simulations Tutorial

This guide shows you how to use both the web interface and backend scripts to run and interpret game theory simulations for fake news propagation analysis.

## Web Interface Tutorial

### Accessing the Simulation Dashboard

1. **Open your browser** to http://localhost:3000
2. **Navigate** to the "Simulations" tab in the main menu
3. **You should see** the simulation configuration interface

### Configuring Your First Simulation

#### Basic Parameters

1. **Network Size**: Start with 1000 nodes
   - Small networks (100-500): Quick results, good for testing
   - Medium networks (1000-5000): Balanced performance and realism
   - Large networks (10000+): Realistic but slower computation

2. **Network Type**: Choose from four options
   - **Scale-Free**: Models social media with influencers (recommended for beginners)
   - **Small-World**: Models real-world social networks
   - **ErdQs-Rényi**: Random connections, good for baseline comparisons
   - **Regular**: All nodes have same degree, useful for controlled experiments

3. **Detection Rate**: Set between 0.0 and 1.0
   - 0.0: No fact-checking capability
   - 0.5: Moderate fact-checking
   - 0.7: Strong fact-checking (recommended starting point)
   - 1.0: Perfect fact-checking (unrealistic)

4. **Simulation Steps**: Number of time periods
   - 50 steps: Quick exploration
   - 100 steps: Standard analysis
   - 200+ steps: Deep investigation

#### Advanced Configuration

**Payoff Matrix Configuration**:

```
                    Fact-Checker
                 Active    Passive
Spreader Aggressive [2,1]     [3,0]
         Moderate   [1,3]     [2,2]
```

You can modify these values to explore different scenarios:
- Higher spreader payoffs ’ More aggressive misinformation
- Higher fact-checker payoffs ’ More active verification

### Running Your First Simulation

#### Step 1: Configure Parameters
```
Network Size: 1000
Network Type: Scale-Free
Detection Rate: 0.7
Simulation Steps: 100
Experiment Name: "My First Simulation"
```

#### Step 2: Start Simulation
1. Click **"Run Simulation"**
2. Monitor the progress bar
3. Wait for completion (typically 30-60 seconds)

#### Step 3: View Results
Once complete, you'll see several visualizations:

**Network Visualization**:
- Nodes colored by state (susceptible/infected/recovered)
- Edges showing information flow
- Size indicates influence/centrality

**Time Series Plot**:
- X-axis: Simulation steps
- Y-axis: Number of nodes in each state
- Shows propagation dynamics over time

**Equilibrium Analysis**:
- Nash equilibria found
- Strategy combinations and payoffs
- Stability measures

### Interpreting Results

#### Key Metrics to Understand

1. **Peak Infection**: Maximum number of nodes believing misinformation
   - Lower is better (effective fact-checking)
   - Higher suggests viral misinformation spread

2. **Final Believers**: Steady-state misinformation believers
   - Persistent misinformation despite fact-checking
   - Key metric for long-term impact

3. **Cascade Size**: Total nodes affected during simulation
   - Measures reach of misinformation campaign
   - Includes both persistent and temporary effects

4. **Nash Equilibria**: Strategic outcomes
   - **Pure equilibria**: Single best strategy for each player
   - **Mixed equilibria**: Randomized strategies
   - **Stability**: Resistance to strategy changes

#### Reading the Network Visualization

**Node Colors**:
- =â **Green**: Unaware/susceptible nodes
- =4 **Red**: Infected with misinformation
- =5 **Blue**: Fact-checked/recovered nodes
- =á **Yellow**: Fact-checker nodes

**Node Sizes**:
- Larger nodes have higher centrality/influence
- Focus on how misinformation spreads through hubs

**Edge Patterns**:
- Thick edges: Strong connections
- Edge direction: Information flow direction

#### Reading Time Series Plots

**Typical Patterns**:

1. **Rapid Rise, Slow Decline**:
   - Quick misinformation spread
   - Gradual fact-checking effect
   - Common in scale-free networks

2. **Oscillating Pattern**:
   - Competing influence between spreaders and fact-checkers
   - Indicates mixed strategy equilibria

3. **Plateau Effect**:
   - Persistent misinformation believers
   - Fact-checking fails to reach all nodes

### Parameter Sweep Experiments

#### Exploring Detection Rate Effects

Run multiple simulations with different detection rates:

```
Simulation 1: Detection Rate = 0.3
Simulation 2: Detection Rate = 0.5
Simulation 3: Detection Rate = 0.7
Simulation 4: Detection Rate = 0.9
```

**Expected Results**:
- Higher detection rates ’ Lower peak infection
- Diminishing returns at very high detection rates
- Different equilibria types across detection levels

#### Network Type Comparison

Run identical parameters across network types:

```
Fixed Parameters:
- Network Size: 1000
- Detection Rate: 0.7
- Steps: 100

Variable: Network Type
1. Scale-Free
2. Small-World
3. ErdQs-Rényi
4. Regular
```

**Expected Differences**:
- **Scale-Free**: Highest peak infection (influencer effect)
- **Small-World**: Moderate spread with clustering
- **Random**: Predictable, gradual spread
- **Regular**: Slowest, most uniform spread

## Command-Line Interface Tutorial

### Single Simulation via CLI

```bash
# Navigate to backend directory
cd backend

# Activate conda environment
conda activate fake_news

# Run a single simulation
python scripts/run_simulation.py --single \
  --network-size 1000 \
  --network-type scale_free \
  --detection-rate 0.7 \
  --simulation-steps 100
```

**Example Output**:
```
2024-01-01 12:00:00,123 - __main__ - INFO - Running single simulation...
Simulation completed successfully!
Network: 1000 nodes, 2847 edges
Nash equilibria found: 1
Peak infection: 234 nodes
Final believers: 89 nodes
Results saved to: results/simulation_20240101_120000.json
```

### Batch Simulations with Configuration Files

#### Create Configuration File

Create `experiments/detection_rate_sweep.yaml`:

```yaml
experiment_name: "Detection Rate Sensitivity Analysis"
output_dir: "results/detection_sweep"
parallel_workers: 4

simulations:
  - name: "low_detection"
    parameters:
      network_size: 1000
      network_type: "scale_free"
      detection_rate: 0.3
      simulation_steps: 100

  - name: "medium_detection"
    parameters:
      network_size: 1000
      network_type: "scale_free"
      detection_rate: 0.5
      simulation_steps: 100

  - name: "high_detection"
    parameters:
      network_size: 1000
      network_type: "scale_free"
      detection_rate: 0.7
      simulation_steps: 100

  - name: "very_high_detection"
    parameters:
      network_size: 1000
      network_type: "scale_free"
      detection_rate: 0.9
      simulation_steps: 100
```

#### Run Batch Simulation

```bash
# Run the experiment batch
python scripts/run_simulation.py \
  --config experiments/detection_rate_sweep.yaml \
  --parallel 4
```

**Batch Output**:
```
2024-01-01 12:00:00,123 - __main__ - INFO - Starting batch simulation...
2024-01-01 12:00:00,145 - __main__ - INFO - Found 4 simulations to run
2024-01-01 12:00:00,167 - __main__ - INFO - Using 4 parallel workers

Running simulations in parallel...
[====================================] 100% (4/4)

Batch simulation completed!
Total simulations: 4
Successful: 4
Failed: 0
Total time: 45.2 seconds

Results aggregated in: results/detection_sweep/summary.csv
```

### Analyzing Batch Results

#### Summary CSV Output

The batch simulation creates `summary.csv`:

```csv
simulation_name,network_size,network_type,detection_rate,peak_infection,final_believers,nash_equilibria_count,dominant_strategy_spreader,dominant_strategy_fact_checker,simulation_time
low_detection,1000,scale_free,0.3,456,234,1,aggressive,passive,12.4
medium_detection,1000,scale_free,0.5,312,156,1,aggressive,active,11.8
high_detection,1000,scale_free,0.7,198,89,1,moderate,active,12.1
very_high_detection,1000,scale_free,0.9,87,34,1,moderate,active,11.9
```

#### Quick Analysis Script

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/detection_sweep/summary.csv')

# Plot detection rate vs peak infection
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(df['detection_rate'], df['peak_infection'], 'bo-')
plt.xlabel('Detection Rate')
plt.ylabel('Peak Infection')
plt.title('Detection Rate vs Peak Infection')

# Plot detection rate vs final believers
plt.subplot(1, 2, 2)
plt.plot(df['detection_rate'], df['final_believers'], 'ro-')
plt.xlabel('Detection Rate')
plt.ylabel('Final Believers')
plt.title('Detection Rate vs Final Believers')

plt.tight_layout()
plt.savefig('detection_rate_analysis.png')
plt.show()
```

## Advanced Simulation Scenarios

### Scenario 1: Influencer Network Analysis

**Research Question**: How do high-degree influencers affect misinformation spread?

```yaml
experiment_name: "Influencer Network Analysis"
simulations:
  - name: "no_influencers"
    parameters:
      network_type: "erdos_renyi"
      network_size: 1000
      detection_rate: 0.7

  - name: "moderate_influencers"
    parameters:
      network_type: "scale_free"
      network_size: 1000
      detection_rate: 0.7

  - name: "super_influencers"
    parameters:
      network_type: "scale_free"
      network_size: 1000
      detection_rate: 0.7
      # Custom parameters for very skewed degree distribution
```

### Scenario 2: Echo Chamber Effect

**Research Question**: How does network clustering affect misinformation persistence?

```yaml
experiment_name: "Echo Chamber Analysis"
simulations:
  - name: "low_clustering"
    parameters:
      network_type: "erdos_renyi"

  - name: "moderate_clustering"
    parameters:
      network_type: "small_world"

  - name: "high_clustering"
    parameters:
      network_type: "small_world"
      # High clustering coefficient
```

### Scenario 3: Intervention Timing

**Research Question**: When is the optimal time to deploy fact-checking?

```yaml
experiment_name: "Intervention Timing"
simulations:
  - name: "immediate_intervention"
    parameters:
      fact_check_delay: 0

  - name: "delayed_intervention"
    parameters:
      fact_check_delay: 10

  - name: "late_intervention"
    parameters:
      fact_check_delay: 25
```

## Performance Tips

### Optimization for Large Simulations

1. **Use Parallel Processing**:
   ```bash
   # Use all CPU cores
   python scripts/run_simulation.py --parallel 0
   ```

2. **Optimize Network Size**:
   - 1000 nodes: Good balance of realism and speed
   - 5000+ nodes: Use only for final analysis
   - 10000+ nodes: Requires significant computing resources

3. **Batch Similar Experiments**:
   - Group experiments with similar parameters
   - Use the same network type to leverage caching

### Memory Management

```bash
# Monitor memory usage
htop

# For large experiments, consider:
export PYTHONHASHSEED=0  # Consistent memory usage

# Clean up between runs
python -c "import gc; gc.collect()"
```

## Troubleshooting Simulations

### Common Issues

1. **Simulation Hangs**:
   - Check network connectivity
   - Reduce network size
   - Verify parameter ranges

2. **No Nash Equilibria Found**:
   - Increase computation tolerance
   - Check payoff matrix validity
   - Try different initial conditions

3. **Unexpected Results**:
   - Verify parameter interpretation
   - Check random seed consistency
   - Compare with baseline scenarios

### Debugging Failed Simulations

```bash
# Run with verbose logging
python scripts/run_simulation.py --single --verbose \
  --network-size 100 --simulation-steps 10

# Check simulation logs
tail -f logs/simulation.log

# Validate network generation
python -c "
from scripts.simulation_utils import generate_network
G = generate_network('scale_free', 100)
print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
"
```

You're now ready to conduct sophisticated game theory simulations! Proceed to the [Model Training Tutorial](./model_training.md) to learn how to train custom fake news detection models.