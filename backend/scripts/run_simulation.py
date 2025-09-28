#!/usr/bin/env python3
"""
Batch Simulation Runner Script

This script enables researchers to run large batches of game theory simulations
programmatically. It supports systematic experiments across parameter spaces,
parallel processing, and comprehensive result aggregation.

Usage:
    python run_simulation.py --config simulations.yaml --output results/
    python run_simulation.py --config experiments/sweep_detection.yaml --parallel 8
    python run_simulation.py --single --network-size 1000 --detection-rate 0.8

Features:
    - Configuration-driven execution from YAML files
    - Parallel processing with multiprocessing
    - Result aggregation and summarization
    - Robust error handling and isolation
    - Progress tracking and logging
"""

import argparse
import logging
import sys
import os
import json
import yaml
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import pandas as pd
import numpy as np
import nashpy as nash
from scipy.optimize import minimize
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class GameTheorySimulation:
    """Individual game theory simulation instance."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.results = {}

    def generate_network(self) -> nx.Graph:
        """Generate social network based on parameters."""
        network_type = self.params.get('network_type', 'erdos_renyi')
        n_nodes = self.params.get('network_size', 1000)

        if network_type == 'erdos_renyi':
            p = self.params.get('connection_probability', 0.01)
            G = nx.erdos_renyi_graph(n_nodes, p, seed=self.params.get('seed', 42))

        elif network_type == 'scale_free':
            m = self.params.get('preferential_attachment', 3)
            G = nx.barabasi_albert_graph(n_nodes, m, seed=self.params.get('seed', 42))

        elif network_type == 'small_world':
            k = self.params.get('nearest_neighbors', 6)
            p = self.params.get('rewiring_probability', 0.1)
            G = nx.watts_strogatz_graph(n_nodes, k, p, seed=self.params.get('seed', 42))

        elif network_type == 'regular':
            k = self.params.get('degree', 4)
            G = nx.random_regular_graph(k, n_nodes, seed=self.params.get('seed', 42))

        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(
                ['spreader', 'fact_checker', 'user'],
                p=self.params.get('node_type_probabilities', [0.1, 0.1, 0.8])
            )
            G.nodes[node]['influence'] = np.random.uniform(0.1, 1.0)
            G.nodes[node]['strategy'] = 'passive'  # Initial strategy

        return G

    def calculate_payoffs(self) -> np.ndarray:
        """Calculate payoff matrix based on simulation parameters."""
        # Base payoff values from parameters
        base_payoffs = self.params.get('base_payoffs', {
            'spreader_spread_undetected': 1.0,
            'spreader_spread_detected': -2.0,
            'spreader_no_spread': 0.0,
            'fact_checker_detect_fake': 1.5,
            'fact_checker_detect_real': -0.5,
            'fact_checker_no_detect': 0.0,
            'platform_remove_fake': 0.5,
            'platform_remove_real': -1.0,
            'platform_no_action': 0.0
        })

        # Detection accuracy affects payoffs
        detection_rate = self.params.get('detection_rate', 0.7)
        network_effect = self.params.get('network_effect_strength', 0.2)

        # Create 2x2 payoff matrix for spreader vs fact-checker game
        # Strategies: [spread/don't_spread] vs [check/don't_check]
        payoff_matrix = np.array([
            # Spreader spreads
            [
                [base_payoffs['spreader_spread_detected'] * detection_rate +
                 base_payoffs['spreader_spread_undetected'] * (1 - detection_rate),
                 base_payoffs['fact_checker_detect_fake'] * detection_rate +
                 base_payoffs['fact_checker_no_detect'] * (1 - detection_rate)],
                [base_payoffs['spreader_spread_undetected'],
                 base_payoffs['fact_checker_no_detect']]
            ],
            # Spreader doesn't spread
            [
                [base_payoffs['spreader_no_spread'],
                 base_payoffs['fact_checker_detect_real'] * 0.1],  # False positive rate
                [base_payoffs['spreader_no_spread'],
                 base_payoffs['fact_checker_no_detect']]
            ]
        ])

        return payoff_matrix

    def find_nash_equilibrium(self, payoff_matrix: np.ndarray) -> Dict[str, Any]:
        """Find Nash equilibria for the given payoff matrix."""
        try:
            # Create game from payoff matrices
            # For a 2x2 game, we need separate payoff matrices for each player
            player1_payoffs = payoff_matrix[:, :, 0]  # Spreader payoffs
            player2_payoffs = payoff_matrix[:, :, 1].T  # Fact-checker payoffs (transposed)

            game = nash.Game(player1_payoffs, player2_payoffs)

            # Find all Nash equilibria
            equilibria = list(game.support_enumeration())

            results = {
                'pure_equilibria': [],
                'mixed_equilibria': [],
                'num_equilibria': len(equilibria)
            }

            for eq in equilibria:
                strategy1, strategy2 = eq
                is_pure = (np.sum(strategy1 == 1) == 1) and (np.sum(strategy2 == 1) == 1)

                eq_data = {
                    'player1_strategy': strategy1.tolist(),
                    'player2_strategy': strategy2.tolist(),
                    'expected_payoffs': [
                        float(np.sum(strategy1 * player1_payoffs * strategy2[:, np.newaxis])),
                        float(np.sum(strategy2 * player2_payoffs * strategy1[:, np.newaxis]))
                    ]
                }

                if is_pure:
                    results['pure_equilibria'].append(eq_data)
                else:
                    results['mixed_equilibria'].append(eq_data)

            return results

        except Exception as e:
            logger.error(f"Failed to find Nash equilibrium: {e}")
            return {
                'pure_equilibria': [],
                'mixed_equilibria': [],
                'num_equilibria': 0,
                'error': str(e)
            }

    def simulate_network_dynamics(self, network: nx.Graph,
                                 payoff_matrix: np.ndarray) -> Dict[str, Any]:
        """Simulate information propagation dynamics on the network."""
        steps = self.params.get('simulation_steps', 100)
        initial_spreaders = self.params.get('initial_spreader_ratio', 0.05)
        propagation_rate = self.params.get('propagation_rate', 0.1)
        recovery_rate = self.params.get('recovery_rate', 0.05)

        # Initialize node states
        states = {node: 'susceptible' for node in network.nodes()}

        # Set initial spreaders
        n_initial = int(len(network) * initial_spreaders)
        initial_nodes = np.random.choice(list(network.nodes()), n_initial, replace=False)
        for node in initial_nodes:
            states[node] = 'spreading'

        # Track dynamics over time
        history = {
            'susceptible': [],
            'spreading': [],
            'recovered': [],
            'network_clustering': [],
            'average_degree': []
        }

        for step in range(steps):
            new_states = states.copy()

            # Update states based on network interactions
            for node in network.nodes():
                if states[node] == 'susceptible':
                    # Check for exposure to spreading neighbors
                    spreading_neighbors = [n for n in network.neighbors(node)
                                         if states[n] == 'spreading']

                    exposure_probability = 1 - (1 - propagation_rate) ** len(spreading_neighbors)
                    if np.random.random() < exposure_probability:
                        new_states[node] = 'spreading'

                elif states[node] == 'spreading':
                    # Chance to recover (stop spreading)
                    if np.random.random() < recovery_rate:
                        new_states[node] = 'recovered'

            states = new_states

            # Record current state
            state_counts = {state: sum(1 for s in states.values() if s == state)
                          for state in ['susceptible', 'spreading', 'recovered']}

            for state in ['susceptible', 'spreading', 'recovered']:
                history[state].append(state_counts[state])

            # Network metrics
            history['network_clustering'].append(nx.average_clustering(network))
            history['average_degree'].append(np.mean([d for n, d in network.degree()]))

        return {
            'final_state_distribution': {
                state: history[state][-1] for state in ['susceptible', 'spreading', 'recovered']
            },
            'dynamics_history': history,
            'total_infected': max(history['spreading']),
            'infection_peak_time': np.argmax(history['spreading']),
            'final_recovered': history['recovered'][-1]
        }

    def run_simulation(self) -> Dict[str, Any]:
        """Execute complete simulation with all components."""
        try:
            # Generate network
            network = self.generate_network()

            # Calculate payoffs
            payoff_matrix = self.calculate_payoffs()

            # Find Nash equilibria
            equilibrium_results = self.find_nash_equilibrium(payoff_matrix)

            # Simulate network dynamics
            dynamics_results = self.simulate_network_dynamics(network, payoff_matrix)

            # Calculate additional metrics
            network_metrics = {
                'nodes': len(network),
                'edges': len(network.edges()),
                'density': nx.density(network),
                'clustering_coefficient': nx.average_clustering(network),
                'diameter': nx.diameter(network) if nx.is_connected(network) else float('inf'),
                'average_path_length': (nx.average_shortest_path_length(network)
                                      if nx.is_connected(network) else float('inf'))
            }

            results = {
                'simulation_id': self.params.get('simulation_id', 'unknown'),
                'parameters': self.params,
                'network_metrics': network_metrics,
                'equilibrium_analysis': equilibrium_results,
                'dynamics_results': dynamics_results,
                'payoff_matrix': payoff_matrix.tolist(),
                'simulation_successful': True,
                'execution_time': datetime.now().isoformat()
            }

            return results

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'simulation_id': self.params.get('simulation_id', 'unknown'),
                'parameters': self.params,
                'simulation_successful': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': datetime.now().isoformat()
            }


def run_single_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for running a single simulation (for multiprocessing)."""
    sim = GameTheorySimulation(params)
    return sim.run_simulation()


class BatchSimulationRunner:
    """Orchestrates batch execution of multiple simulations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Execution statistics
        self.stats = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'failed_simulations': 0,
            'start_time': None,
            'end_time': None
        }

    def load_simulation_configs(self) -> List[Dict[str, Any]]:
        """Load simulation configurations from config file."""
        if 'simulations' in self.config:
            # Direct configuration
            return self.config['simulations']

        elif 'parameter_sweep' in self.config:
            # Generate parameter sweep
            return self._generate_parameter_sweep()

        else:
            raise ValueError("No simulation configurations found in config")

    def _generate_parameter_sweep(self) -> List[Dict[str, Any]]:
        """Generate simulations from parameter sweep configuration."""
        sweep_config = self.config['parameter_sweep']
        base_params = sweep_config.get('base_parameters', {})
        sweep_params = sweep_config.get('sweep_parameters', {})

        simulations = []
        simulation_id = 1

        # Generate Cartesian product of sweep parameters
        import itertools

        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]

        for combination in itertools.product(*param_values):
            params = base_params.copy()
            params.update(dict(zip(param_names, combination)))
            params['simulation_id'] = f"sweep_{simulation_id:04d}"

            simulations.append(params)
            simulation_id += 1

        logger.info(f"Generated {len(simulations)} simulations from parameter sweep")
        return simulations

    def run_simulations_parallel(self, simulation_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run simulations in parallel using multiprocessing."""
        n_workers = self.config.get('parallel_workers', mp.cpu_count())
        logger.info(f"Running {len(simulation_configs)} simulations with {n_workers} workers")

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all simulations
            future_to_config = {
                executor.submit(run_single_simulation, config): config
                for config in simulation_configs
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_config)):
                try:
                    result = future.result()
                    results.append(result)

                    if result['simulation_successful']:
                        self.stats['successful_simulations'] += 1
                    else:
                        self.stats['failed_simulations'] += 1

                    # Log progress
                    if (i + 1) % 10 == 0 or (i + 1) == len(simulation_configs):
                        logger.info(f"Completed {i + 1}/{len(simulation_configs)} simulations")

                except Exception as e:
                    logger.error(f"Simulation execution failed: {e}")
                    self.stats['failed_simulations'] += 1

                    # Create error result
                    config = future_to_config[future]
                    error_result = {
                        'simulation_id': config.get('simulation_id', 'unknown'),
                        'simulation_successful': False,
                        'error': str(e),
                        'parameters': config
                    }
                    results.append(error_result)

        return results

    def run_simulations_sequential(self, simulation_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run simulations sequentially (for debugging or memory constraints)."""
        logger.info(f"Running {len(simulation_configs)} simulations sequentially")

        results = []
        for i, config in enumerate(simulation_configs):
            try:
                result = run_single_simulation(config)
                results.append(result)

                if result['simulation_successful']:
                    self.stats['successful_simulations'] += 1
                else:
                    self.stats['failed_simulations'] += 1

                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == len(simulation_configs):
                    logger.info(f"Completed {i + 1}/{len(simulation_configs)} simulations")

            except Exception as e:
                logger.error(f"Simulation {i+1} failed: {e}")
                self.stats['failed_simulations'] += 1

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate and summarize simulation results."""
        successful_results = [r for r in results if r.get('simulation_successful', False)]

        if not successful_results:
            return {'error': 'No successful simulations to aggregate'}

        # Extract key metrics
        metrics = {
            'network_sizes': [],
            'detection_rates': [],
            'num_equilibria': [],
            'total_infected': [],
            'infection_peaks': [],
            'final_recovered': [],
            'network_densities': [],
            'clustering_coefficients': []
        }

        for result in successful_results:
            params = result['parameters']
            metrics['network_sizes'].append(params.get('network_size', 0))
            metrics['detection_rates'].append(params.get('detection_rate', 0))

            equilibrium = result.get('equilibrium_analysis', {})
            metrics['num_equilibria'].append(equilibrium.get('num_equilibria', 0))

            dynamics = result.get('dynamics_results', {})
            metrics['total_infected'].append(dynamics.get('total_infected', 0))
            metrics['infection_peaks'].append(dynamics.get('infection_peak_time', 0))
            metrics['final_recovered'].append(dynamics.get('final_recovered', 0))

            network = result.get('network_metrics', {})
            metrics['network_densities'].append(network.get('density', 0))
            metrics['clustering_coefficients'].append(network.get('clustering_coefficient', 0))

        # Calculate summary statistics
        summary = {}
        for metric, values in metrics.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        # Correlation analysis
        df = pd.DataFrame(metrics)
        correlation_matrix = df.corr().to_dict()

        return {
            'summary_statistics': summary,
            'correlations': correlation_matrix,
            'total_simulations': len(results),
            'successful_simulations': len(successful_results),
            'failed_simulations': len(results) - len(successful_results)
        }

    def save_results(self, results: List[Dict[str, Any]], aggregated: Dict[str, Any]):
        """Save simulation results and aggregated analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save individual results
        results_file = self.output_dir / f"simulation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save aggregated results
        summary_file = self.output_dir / f"simulation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)

        # Save CSV summary for easy analysis
        successful_results = [r for r in results if r.get('simulation_successful', False)]
        if successful_results:
            csv_data = []
            for result in successful_results:
                row = result['parameters'].copy()

                # Add key results
                equilibrium = result.get('equilibrium_analysis', {})
                row['num_equilibria'] = equilibrium.get('num_equilibria', 0)

                dynamics = result.get('dynamics_results', {})
                row['total_infected'] = dynamics.get('total_infected', 0)
                row['final_recovered'] = dynamics.get('final_recovered', 0)

                network = result.get('network_metrics', {})
                row['network_density'] = network.get('density', 0)
                row['clustering_coefficient'] = network.get('clustering_coefficient', 0)

                csv_data.append(row)

            csv_file = self.output_dir / f"simulation_results_{timestamp}.csv"
            pd.DataFrame(csv_data).to_csv(csv_file, index=False)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Individual results: {results_file}")
        logger.info(f"Summary: {summary_file}")

    def run_batch(self) -> Dict[str, Any]:
        """Execute the complete batch simulation process."""
        self.stats['start_time'] = datetime.now()

        try:
            # Load simulation configurations
            simulation_configs = self.load_simulation_configs()
            self.stats['total_simulations'] = len(simulation_configs)

            # Run simulations
            if self.config.get('parallel', True):
                results = self.run_simulations_parallel(simulation_configs)
            else:
                results = self.run_simulations_sequential(simulation_configs)

            # Aggregate results
            aggregated = self.aggregate_results(results)

            # Save results
            self.save_results(results, aggregated)

            self.stats['end_time'] = datetime.now()
            execution_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            logger.info(f"Batch simulation completed in {execution_time:.2f} seconds")
            logger.info(f"Successfully completed: {self.stats['successful_simulations']}")
            logger.info(f"Failed: {self.stats['failed_simulations']}")

            return {
                'status': 'success',
                'statistics': self.stats,
                'aggregated_results': aggregated
            }

        except Exception as e:
            logger.error(f"Batch simulation failed: {e}")
            self.stats['end_time'] = datetime.now()
            return {
                'status': 'failed',
                'error': str(e),
                'statistics': self.stats
            }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch game theory simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=0,
        help='Number of parallel workers (0 = number of CPU cores)'
    )

    # Single simulation mode
    parser.add_argument(
        '--single',
        action='store_true',
        help='Run a single simulation with command line parameters'
    )

    parser.add_argument(
        '--network-size',
        type=int,
        default=1000,
        help='Network size for single simulation'
    )

    parser.add_argument(
        '--network-type',
        type=str,
        default='scale_free',
        choices=['erdos_renyi', 'scale_free', 'small_world', 'regular'],
        help='Network type for single simulation'
    )

    parser.add_argument(
        '--detection-rate',
        type=float,
        default=0.7,
        help='Detection rate for single simulation'
    )

    parser.add_argument(
        '--simulation-steps',
        type=int,
        default=100,
        help='Number of simulation steps'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.single:
        # Run single simulation
        params = {
            'simulation_id': 'single_run',
            'network_size': args.network_size,
            'network_type': args.network_type,
            'detection_rate': args.detection_rate,
            'simulation_steps': args.simulation_steps,
            'seed': 42
        }

        logger.info("Running single simulation...")
        result = run_single_simulation(params)

        if result['simulation_successful']:
            print("\nSimulation completed successfully!")
            print(f"Network: {result['network_metrics']['nodes']} nodes, "
                  f"{result['network_metrics']['edges']} edges")
            print(f"Nash equilibria found: {result['equilibrium_analysis']['num_equilibria']}")
            print(f"Peak infection: {result['dynamics_results']['total_infected']} nodes")
        else:
            print(f"Simulation failed: {result.get('error', 'Unknown error')}")

    else:
        # Run batch simulations
        if not args.config:
            print("Error: --config is required for batch mode")
            sys.exit(1)

        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        config['output_dir'] = args.output_dir
        if args.parallel > 0:
            config['parallel_workers'] = args.parallel

        logger.info(f"Starting batch simulation with config: {args.config}")

        # Run batch simulation
        runner = BatchSimulationRunner(config)
        results = runner.run_batch()

        if results['status'] == 'success':
            print("\nBatch simulation completed successfully!")
            stats = results['statistics']
            print(f"Total simulations: {stats['total_simulations']}")
            print(f"Successful: {stats['successful_simulations']}")
            print(f"Failed: {stats['failed_simulations']}")

            execution_time = (stats['end_time'] - stats['start_time']).total_seconds()
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Batch simulation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()