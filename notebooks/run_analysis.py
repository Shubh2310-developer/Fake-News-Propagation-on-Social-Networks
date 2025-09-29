#!/usr/bin/env python3
"""
Fake News Game Theory - Comprehensive Simulation Analysis
Run the complete analysis from the 06_simulation_experiments.ipynb notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
class SimulationConfig:
    """Configuration for simulation experiments"""

    # Network parameters
    NETWORK_SIZES = [100, 500, 1000]
    NETWORK_TYPES = ['barabasi_albert', 'watts_strogatz', 'erdos_renyi']

    # Game theory parameters
    NUM_SPREADERS = [5, 10, 15]
    NUM_FACT_CHECKERS = [2, 5, 8]
    NUM_PLATFORMS = 1

    # Simulation parameters
    TIME_HORIZON = 30
    NUM_SIMULATIONS = 3  # Reduced for demo
    LEARNING_RATE = 0.01

    # Payoff weights
    SPREADER_WEIGHTS = {
        'reach': 1.0,
        'detection': -0.5,
        'reputation': -0.3
    }

    FACT_CHECKER_WEIGHTS = {
        'accuracy': 1.0,
        'effort': -0.2,
        'impact': 0.5
    }

    PLATFORM_WEIGHTS = {
        'engagement': 0.8,
        'reputation': -0.6,
        'cost': -0.1
    }

class NetworkGenerator:
    """Generate various network topologies for simulation"""

    @staticmethod
    def generate_barabasi_albert(n: int, m: int = 3) -> nx.Graph:
        """Generate scale-free network (mimics social media)"""
        G = nx.barabasi_albert_graph(n, m, seed=42)
        NetworkGenerator._add_node_attributes(G)
        NetworkGenerator._add_edge_attributes(G)
        return G

    @staticmethod
    def generate_watts_strogatz(n: int, k: int = 6, p: float = 0.1) -> nx.Graph:
        """Generate small-world network"""
        G = nx.watts_strogatz_graph(n, k, p, seed=42)
        NetworkGenerator._add_node_attributes(G)
        NetworkGenerator._add_edge_attributes(G)
        return G

    @staticmethod
    def generate_erdos_renyi(n: int, p: float = 0.01) -> nx.Graph:
        """Generate random network"""
        G = nx.erdos_renyi_graph(n, p, seed=42)
        NetworkGenerator._add_node_attributes(G)
        NetworkGenerator._add_edge_attributes(G)
        return G

    @staticmethod
    def _add_node_attributes(G: nx.Graph):
        """Add realistic user attributes to nodes"""
        for node in G.nodes():
            degree = G.degree(node)
            G.nodes[node]['influence_score'] = np.log1p(degree) / np.log1p(G.number_of_nodes())
            G.nodes[node]['credibility_score'] = np.random.beta(2, 2)
            G.nodes[node]['activity_level'] = np.random.gamma(2, 0.5)
            G.nodes[node]['user_type'] = np.random.choice(
                ['spreader', 'fact_checker', 'regular_user', 'platform'],
                p=[0.05, 0.03, 0.91, 0.01]
            )
            G.nodes[node]['verified'] = np.random.choice([True, False], p=[0.05, 0.95])

    @staticmethod
    def _add_edge_attributes(G: nx.Graph):
        """Add trust and interaction weights to edges"""
        for u, v in G.edges():
            # Trust based on similarity
            user_u = G.nodes[u]
            user_v = G.nodes[v]
            similarity = 1 - abs(user_u['credibility_score'] - user_v['credibility_score'])
            G.edges[u, v]['trust'] = np.clip(similarity + np.random.normal(0, 0.1), 0, 1)
            G.edges[u, v]['interaction_strength'] = np.random.exponential(0.3)

class InformationPropagationSimulator:
    """Simulate information spread through social networks"""

    def __init__(self, network: nx.Graph, content_quality: float = 0.5):
        self.network = network
        self.content_quality = content_quality  # 0 = fake, 1 = real
        self.states = {node: 'susceptible' for node in network.nodes()}
        self.propagation_history = []

    def independent_cascade(self, initial_spreaders: List[int],
                          time_steps: int = 50) -> Dict:
        """Independent Cascade propagation model"""
        newly_infected = set(initial_spreaders)
        for node in initial_spreaders:
            self.states[node] = 'infected'

        propagation_log = []

        for t in range(time_steps):
            current_newly_infected = set()

            for infected_node in newly_infected:
                neighbors = list(self.network.neighbors(infected_node))

                for neighbor in neighbors:
                    if self.states[neighbor] == 'susceptible':
                        # Calculate infection probability
                        prob = self._calculate_infection_probability(
                            infected_node, neighbor
                        )

                        if np.random.random() < prob:
                            self.states[neighbor] = 'infected'
                            current_newly_infected.add(neighbor)

            # Update for next iteration
            newly_infected = current_newly_infected

            # Log current state
            infected_count = sum(1 for s in self.states.values() if s == 'infected')
            propagation_log.append({
                'time_step': t,
                'total_infected': infected_count,
                'newly_infected': len(current_newly_infected),
                'infection_rate': infected_count / len(self.states)
            })

            # Stop if no new infections
            if not current_newly_infected:
                break

        return {
            'final_states': self.states.copy(),
            'propagation_log': propagation_log,
            'total_reach': sum(1 for s in self.states.values() if s == 'infected'),
            'cascade_depth': len(propagation_log)
        }

    def _calculate_infection_probability(self, spreader: int, target: int) -> float:
        """Calculate transmission probability between two users"""
        # Base transmission rate
        base_rate = 0.1

        # Spreader influence
        spreader_influence = self.network.nodes[spreader]['influence_score']

        # Target susceptibility (inverse of credibility)
        target_susceptibility = 1 - self.network.nodes[target]['credibility_score']

        # Edge trust
        edge_trust = self.network.edges[spreader, target]['trust']

        # Content quality effect (fake news spreads faster)
        content_multiplier = 1.0 + (1 - self.content_quality) * 0.5

        # Combine factors
        probability = (base_rate * spreader_influence * target_susceptibility *
                      edge_trust * content_multiplier)

        return min(probability, 1.0)

class FakeNewsGameSimulation:
    """Complete simulation framework integrating all components"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = []

    def run_experiments(self) -> pd.DataFrame:
        """Run comprehensive simulation experiments"""
        print("Starting simulation experiments...")

        experiment_id = 0
        total_experiments = (len(self.config.NETWORK_SIZES) *
                           len(self.config.NETWORK_TYPES) *
                           len(self.config.NUM_SPREADERS) *
                           self.config.NUM_SIMULATIONS)

        for network_size in self.config.NETWORK_SIZES:
            for network_type in self.config.NETWORK_TYPES:
                for num_spreaders in self.config.NUM_SPREADERS:
                    for sim_run in range(self.config.NUM_SIMULATIONS):
                        experiment_id += 1
                        print(f"Running experiment {experiment_id}/{total_experiments}...",
                              end='\r')

                        result = self._run_single_experiment(
                            network_size, network_type, num_spreaders, sim_run
                        )
                        self.results.append(result)

        print(f"\n✓ All {total_experiments} experiments completed")
        return pd.DataFrame(self.results)

    def _run_single_experiment(self, network_size: int, network_type: str,
                              num_spreaders: int, sim_run: int) -> Dict:
        """Run a single simulation experiment"""
        # Generate network
        if network_type == 'barabasi_albert':
            network = NetworkGenerator.generate_barabasi_albert(network_size)
        elif network_type == 'watts_strogatz':
            network = NetworkGenerator.generate_watts_strogatz(network_size)
        else:
            network = NetworkGenerator.generate_erdos_renyi(network_size)

        # Select initial spreaders (high influence nodes)
        degrees = dict(network.degree())
        initial_spreaders = sorted(degrees, key=degrees.get, reverse=True)[:num_spreaders]

        # Run propagation simulation (fake news)
        fake_propagator = InformationPropagationSimulator(network, content_quality=0.2)
        fake_result = fake_propagator.independent_cascade(initial_spreaders,
                                                         self.config.TIME_HORIZON)

        # Run propagation simulation (real news) for comparison
        real_propagator = InformationPropagationSimulator(network, content_quality=0.8)
        real_result = real_propagator.independent_cascade(initial_spreaders,
                                                         self.config.TIME_HORIZON)

        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(network)

        return {
            'experiment_id': f"{network_type}_{network_size}_{num_spreaders}_{sim_run}",
            'network_size': network_size,
            'network_type': network_type,
            'num_spreaders': num_spreaders,
            'sim_run': sim_run,
            'fake_news_reach': fake_result['total_reach'],
            'fake_news_cascade_depth': fake_result['cascade_depth'],
            'real_news_reach': real_result['total_reach'],
            'real_news_cascade_depth': real_result['cascade_depth'],
            'fake_vs_real_ratio': fake_result['total_reach'] / max(real_result['total_reach'], 1),
            'avg_clustering': network_metrics['avg_clustering'],
            'avg_path_length': network_metrics['avg_path_length'],
            'density': network_metrics['density'],
            'timestamp': pd.Timestamp.now()
        }

    def _calculate_network_metrics(self, G: nx.Graph) -> Dict:
        """Calculate key network topology metrics"""
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = 0

        return {
            'avg_clustering': nx.average_clustering(G),
            'avg_path_length': avg_path_length,
            'density': nx.density(G)
        }

def run_intervention_analysis(network_size: int = 500) -> pd.DataFrame:
    """Analyze different intervention strategies"""
    print("Running intervention strategy analysis...")

    network = NetworkGenerator.generate_barabasi_albert(network_size)
    degrees = dict(network.degree())
    initial_spreaders = sorted(degrees, key=degrees.get, reverse=True)[:10]

    intervention_results = []

    # Baseline (no intervention)
    baseline_sim = InformationPropagationSimulator(network, content_quality=0.2)
    baseline_result = baseline_sim.independent_cascade(initial_spreaders, 30)

    # Intervention 1: Target high-degree nodes (influencer strategy)
    modified_network = network.copy()
    high_degree_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    for node in high_degree_nodes:
        modified_network.nodes[node]['credibility_score'] = 0.9  # Boost credibility

    influencer_sim = InformationPropagationSimulator(modified_network, content_quality=0.2)
    influencer_result = influencer_sim.independent_cascade(initial_spreaders, 30)

    # Intervention 2: Increase edge trust globally (education strategy)
    modified_network2 = network.copy()
    for u, v in modified_network2.edges():
        modified_network2.edges[u, v]['trust'] *= 0.8  # Reduce trust in all connections

    education_sim = InformationPropagationSimulator(modified_network2, content_quality=0.2)
    education_result = education_sim.independent_cascade(initial_spreaders, 30)

    # Intervention 3: Remove weak links
    modified_network3 = network.copy()
    edges_to_remove = [(u, v) for u, v in modified_network3.edges()
                       if modified_network3.edges[u, v]['trust'] < 0.3]
    modified_network3.remove_edges_from(edges_to_remove)

    removal_sim = InformationPropagationSimulator(modified_network3, content_quality=0.2)
    removal_result = removal_sim.independent_cascade(initial_spreaders, 30)

    interventions = [
        ('Baseline', baseline_result['total_reach'], 0),
        ('Influencer Strategy', influencer_result['total_reach'],
         (baseline_result['total_reach'] - influencer_result['total_reach']) / baseline_result['total_reach'] * 100),
        ('Education Strategy', education_result['total_reach'],
         (baseline_result['total_reach'] - education_result['total_reach']) / baseline_result['total_reach'] * 100),
        ('Link Removal', removal_result['total_reach'],
         (baseline_result['total_reach'] - removal_result['total_reach']) / baseline_result['total_reach'] * 100)
    ]

    return pd.DataFrame(interventions, columns=['Strategy', 'Reach', 'Reduction_Pct'])

def create_visualizations(results_df, intervention_df):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")

    # Create figures directory if it doesn't exist
    os.makedirs('../reports/figures', exist_ok=True)

    # Figure 1: Main simulation results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Fake vs Real News Reach by Network Type
    ax = axes[0, 0]
    comparison_data = results_df.groupby('network_type')[
        ['fake_news_reach', 'real_news_reach']
    ].mean()
    comparison_data.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
    ax.set_title('Average Reach: Fake vs Real News by Network Type',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Network Type')
    ax.set_ylabel('Average Reach (nodes)')
    ax.legend(['Fake News', 'Real News'])
    ax.grid(True, alpha=0.3)

    # 2. Fake News Spread Ratio by Network Size
    ax = axes[0, 1]
    config = SimulationConfig()
    for net_type in config.NETWORK_TYPES:
        data = results_df[results_df['network_type'] == net_type]
        grouped = data.groupby('network_size')['fake_vs_real_ratio'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=net_type, linewidth=2)
    ax.set_title('Fake News Advantage by Network Size',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Network Size')
    ax.set_ylabel('Fake/Real Reach Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cascade Depth Distribution
    ax = axes[1, 0]
    for net_type in config.NETWORK_TYPES:
        data = results_df[results_df['network_type'] == net_type]
        ax.hist(data['fake_news_cascade_depth'], alpha=0.5, label=net_type, bins=10)
    ax.set_title('Fake News Cascade Depth Distribution',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cascade Depth')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Network Topology Effect
    ax = axes[1, 1]
    scatter_data = results_df.groupby('network_type').agg({
        'avg_clustering': 'mean',
        'fake_news_reach': 'mean'
    })
    colors = {'barabasi_albert': '#e74c3c', 'watts_strogatz': '#3498db',
              'erdos_renyi': '#2ecc71'}
    for net_type in scatter_data.index:
        ax.scatter(scatter_data.loc[net_type, 'avg_clustering'],
                  scatter_data.loc[net_type, 'fake_news_reach'],
                  s=200, alpha=0.6, color=colors[net_type], label=net_type)
    ax.set_title('Clustering vs Misinformation Reach',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Clustering Coefficient')
    ax.set_ylabel('Average Fake News Reach')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../reports/figures/simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Intervention strategies
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = intervention_df['Strategy']
    reaches = intervention_df['Reach']
    reductions = intervention_df['Reduction_Pct']

    # Color bars by effectiveness
    colors = ['#95a5a6' if r == 0 else '#e74c3c' if r > 25 else '#f39c12' if r > 10 else '#2ecc71'
              for r in reductions]

    bars = ax.bar(strategies, reaches, color=colors, alpha=0.7)
    ax.set_title('Intervention Strategy Effectiveness', fontsize=14, fontweight='bold')
    ax.set_ylabel('Misinformation Reach (nodes)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add reduction percentages as labels
    for i, (bar, reduction) in enumerate(zip(bars, reductions)):
        if reduction > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'-{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/figures/intervention_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_statistical_analysis(results_df):
    """Perform statistical analysis of results"""
    print("Performing statistical analysis...")

    # Test 1: Does network type significantly affect fake news spread?
    network_types = results_df['network_type'].unique()
    groups = [results_df[results_df['network_type'] == nt]['fake_news_reach'].values
              for nt in network_types]
    f_stat, p_value = stats.f_oneway(*groups)

    # Test 2: Correlation between network metrics and reach
    correlations = results_df[['avg_clustering', 'avg_path_length',
                              'density', 'fake_news_reach']].corr()

    # Test 3: Effect size of fake vs real news
    fake_mean = results_df['fake_news_reach'].mean()
    real_mean = results_df['real_news_reach'].mean()
    pooled_std = np.sqrt((results_df['fake_news_reach'].std()**2 +
                         results_df['real_news_reach'].std()**2) / 2)
    cohens_d = (fake_mean - real_mean) / pooled_std if pooled_std > 0 else 0

    return {
        'anova_f_stat': f_stat,
        'anova_p_value': p_value,
        'correlations': correlations.loc['fake_news_reach'].to_dict(),
        'fake_mean': fake_mean,
        'real_mean': real_mean,
        'cohens_d': cohens_d
    }

def main():
    """Main execution function"""
    print("="*70)
    print("FAKE NEWS GAME THEORY - SIMULATION EXPERIMENTS")
    print("="*70)

    # Initialize configuration
    config = SimulationConfig()
    print(f"✓ Configuration loaded")
    print(f"  Network sizes: {config.NETWORK_SIZES}")
    print(f"  Network types: {config.NETWORK_TYPES}")
    print(f"  Simulations per configuration: {config.NUM_SIMULATIONS}")

    # Run experiments
    simulator = FakeNewsGameSimulation(config)
    results_df = simulator.run_experiments()

    # Save results
    results_df.to_csv('../data/processed/simulation_results_new.csv', index=False)
    print(f"✓ Results saved: {len(results_df)} experiments")

    # Run intervention analysis
    intervention_df = run_intervention_analysis()
    print(f"✓ Intervention analysis completed")

    # Perform statistical analysis
    stats_results = perform_statistical_analysis(results_df)
    print(f"✓ Statistical analysis completed")

    # Create visualizations
    create_visualizations(results_df, intervention_df)
    print(f"✓ Visualizations created")

    # Generate summary report
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    print(results_df.groupby(['network_type', 'network_size'])[
        ['fake_news_reach', 'real_news_reach', 'fake_vs_real_ratio']
    ].mean())

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Network Type Effect on Fake News Reach:")
    print(f"  F-statistic: {stats_results['anova_f_stat']:.4f}")
    print(f"  P-value: {stats_results['anova_p_value']:.4f}")
    print(f"  Result: {'Significant' if stats_results['anova_p_value'] < 0.05 else 'Not significant'} (α=0.05)")

    print(f"\nCorrelations with Fake News Reach:")
    for metric, corr in stats_results['correlations'].items():
        if metric != 'fake_news_reach':
            print(f"  {metric}: {corr:.4f}")

    print(f"\nEffect Size (Cohen's d) - Fake vs Real News Reach:")
    print(f"  Fake News Mean: {stats_results['fake_mean']:.2f}")
    print(f"  Real News Mean: {stats_results['real_mean']:.2f}")
    print(f"  Cohen's d: {stats_results['cohens_d']:.4f}")

    print("\n" + "="*60)
    print("INTERVENTION ANALYSIS")
    print("="*60)
    print(intervention_df.to_string(index=False))

    # Export final summary
    summary = {
        'experiment_metadata': {
            'total_experiments': len(results_df),
            'completion_time': str(pd.Timestamp.now())
        },
        'key_findings': {
            'fake_news_advantage': float(results_df['fake_vs_real_ratio'].mean()),
            'most_vulnerable_network': results_df.groupby('network_type')['fake_news_reach'].mean().idxmax(),
            'best_intervention': intervention_df.loc[intervention_df['Reduction_Pct'].idxmax(), 'Strategy']
        },
        'statistical_results': stats_results
    }

    with open('../reports/simulation_experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Summary exported to ../reports/simulation_experiments_summary.json")
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()