# backend/app/services/simulation_service.py

from typing import Dict, Any, List, Optional
import logging
import asyncio
from uuid import uuid4
from datetime import datetime
import json
from pathlib import Path

from network import SocialNetworkGenerator, NetworkConfig
from game_theory import (
    Spreader,
    FactChecker,
    Platform,
    RepeatedGameSimulation,
    SimulationConfig,
    RoundResult
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# In-memory storage for simulation jobs
# In production, this would use Redis or a database
simulation_jobs: Dict[str, Any] = {}


class SimulationService:
    """Manages the setup and execution of game theory simulations."""

    def __init__(self):
        """Initialize the simulation service."""
        self.simulation_storage_path = Path(getattr(settings, 'SIMULATION_STORAGE_PATH', 'simulations'))
        self.simulation_storage_path.mkdir(exist_ok=True)

        # Load existing simulation metadata
        self._load_existing_simulations()

        logger.info("SimulationService initialized")

    def _load_existing_simulations(self) -> None:
        """Load metadata for existing simulations from storage."""
        try:
            metadata_file = self.simulation_storage_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                    simulation_jobs.update(existing_metadata)
                logger.info(f"Loaded {len(existing_metadata)} existing simulations")
        except Exception as e:
            logger.warning(f"Failed to load existing simulation metadata: {e}")

    def _save_simulation_metadata(self) -> None:
        """Save simulation metadata to storage."""
        try:
            metadata_file = self.simulation_storage_path / 'metadata.json'
            # Only save metadata, not full results
            metadata = {
                sim_id: {k: v for k, v in sim_data.items() if k != 'results'}
                for sim_id, sim_data in simulation_jobs.items()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save simulation metadata: {e}")

    async def start_simulation(self, params: Dict[str, Any]) -> str:
        """
        Sets up and starts a new simulation.

        Args:
            params: Simulation parameters including network and game configuration

        Returns:
            Unique simulation ID
        """
        try:
            simulation_id = str(uuid4())

            # Validate parameters
            validated_params = self._validate_simulation_params(params)

            # Initialize simulation job
            simulation_jobs[simulation_id] = {
                "id": simulation_id,
                "status": "pending",
                "params": validated_params,
                "created_at": datetime.utcnow().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "progress": 0.0
            }

            # Save metadata
            self._save_simulation_metadata()

            # Start simulation asynchronously
            asyncio.create_task(self._run_simulation_async(simulation_id, validated_params))

            logger.info(f"Simulation {simulation_id} started with parameters: {validated_params}")
            return simulation_id

        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            raise

    def _validate_simulation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set defaults for simulation parameters."""
        validated = {
            # Network configuration
            'network_config': {
                'num_nodes': params.get('network_config', {}).get('num_nodes', 1000),
                'network_type': params.get('network_config', {}).get('network_type', 'barabasi_albert'),
                'attachment_preference': params.get('network_config', {}).get('attachment_preference', 5),
                'rewiring_probability': params.get('network_config', {}).get('rewiring_probability', 0.1),
                'random_seed': params.get('network_config', {}).get('random_seed', None)
            },

            # Game configuration
            'game_config': {
                'num_rounds': params.get('game_config', {}).get('num_rounds', 100),
                'num_spreaders': params.get('game_config', {}).get('num_spreaders', 10),
                'num_fact_checkers': params.get('game_config', {}).get('num_fact_checkers', 5),
                'num_platforms': params.get('game_config', {}).get('num_platforms', 1),
                'learning_rate': params.get('game_config', {}).get('learning_rate', 0.1),
                'exploration_rate': params.get('game_config', {}).get('exploration_rate', 0.1),
                'random_seed': params.get('game_config', {}).get('random_seed', None)
            },

            # Simulation metadata
            'description': params.get('description', ''),
            'tags': params.get('tags', []),
            'save_detailed_history': params.get('save_detailed_history', True),
            'save_network': params.get('save_network', True)
        }

        # Validate ranges
        if validated['network_config']['num_nodes'] > 10000:
            raise ValueError("Network size cannot exceed 10,000 nodes")

        if validated['game_config']['num_rounds'] > 1000:
            raise ValueError("Number of rounds cannot exceed 1,000")

        return validated

    async def _run_simulation_async(self, simulation_id: str, params: Dict[str, Any]) -> None:
        """Run the simulation asynchronously."""
        try:
            # Update status
            simulation_jobs[simulation_id]["status"] = "running"
            simulation_jobs[simulation_id]["started_at"] = datetime.utcnow().isoformat()
            self._save_simulation_metadata()

            # Run the actual simulation
            await self._execute_simulation(simulation_id, params)

        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            simulation_jobs[simulation_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            })
            self._save_simulation_metadata()

    async def _execute_simulation(self, simulation_id: str, params: Dict[str, Any]) -> None:
        """Execute the core simulation logic."""
        try:
            # Step 1: Generate network
            logger.info(f"Generating network for simulation {simulation_id}")
            network_config = NetworkConfig(**params['network_config'])
            generator = SocialNetworkGenerator(network_config)
            network = generator.generate_network(
                network_type=params['network_config']['network_type']
            )

            # Update progress
            simulation_jobs[simulation_id]["progress"] = 0.2

            # Step 2: Initialize players
            logger.info(f"Initializing players for simulation {simulation_id}")
            players = self._create_players(params['game_config'])

            # Update progress
            simulation_jobs[simulation_id]["progress"] = 0.3

            # Step 3: Configure simulation
            sim_config = SimulationConfig(
                num_rounds=params['game_config']['num_rounds'],
                learning_rate=params['game_config']['learning_rate'],
                exploration_rate=params['game_config']['exploration_rate'],
                random_seed=params['game_config']['random_seed']
            )

            # Step 4: Run simulation
            logger.info(f"Running simulation {simulation_id}")
            simulation_engine = RepeatedGameSimulation(players, sim_config)

            # Run simulation with progress updates
            results = await self._run_with_progress_updates(
                simulation_id, simulation_engine, sim_config.num_rounds
            )

            # Step 5: Process and save results
            logger.info(f"Processing results for simulation {simulation_id}")
            processed_results = await self._process_simulation_results(
                simulation_id, results, network, params
            )

            # Update final status
            simulation_jobs[simulation_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "progress": 1.0,
                "summary": self._create_result_summary(processed_results)
            })

            # Save full results to file
            await self._save_simulation_results(simulation_id, processed_results)

            self._save_simulation_metadata()
            logger.info(f"Simulation {simulation_id} completed successfully")

        except Exception as e:
            logger.error(f"Error executing simulation {simulation_id}: {e}")
            raise

    def _create_players(self, game_config: Dict[str, Any]) -> List:
        """Create player instances based on configuration."""
        players = []

        # Create spreaders
        for i in range(game_config['num_spreaders']):
            player = Spreader(
                player_id=f"spreader_{i}",
                spreader_type='profit-driven'
            )
            players.append(player)

        # Create fact checkers
        for i in range(game_config['num_fact_checkers']):
            player = FactChecker(
                player_id=f"fact_checker_{i}",
                checker_type='professional'
            )
            players.append(player)

        # Create platforms
        for i in range(game_config['num_platforms']):
            player = Platform(player_id=f"platform_{i}")
            players.append(player)

        return players

    async def _run_with_progress_updates(self,
                                       simulation_id: str,
                                       simulation_engine: RepeatedGameSimulation,
                                       total_rounds: int) -> List[RoundResult]:
        """Run simulation with periodic progress updates."""
        # This is a simplified version - in reality, you'd need to modify
        # RepeatedGameSimulation to support progress callbacks
        results = simulation_engine.run_simulation()

        # Update progress to 90% after simulation completes
        simulation_jobs[simulation_id]["progress"] = 0.9

        return results

    async def _process_simulation_results(self,
                                        simulation_id: str,
                                        results: List[RoundResult],
                                        network: Any,
                                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw simulation results into structured format."""
        # Calculate summary statistics
        total_rounds = len(results)

        # Extract payoff trends
        payoff_trends = {}
        for result in results:
            for player_type, payoff in result.payoffs.items():
                if player_type not in payoff_trends:
                    payoff_trends[player_type] = []
                payoff_trends[player_type].append(payoff)

        # Calculate final metrics
        final_metrics = {}
        if results:
            final_result = results[-1]
            final_metrics = {
                'final_payoffs': final_result.payoffs,
                'final_reputation_scores': {
                    player_id: state.get('reputation', 0.5)
                    for player_id, state in final_result.player_states.items()
                }
            }

        # Network metrics
        from network import NetworkAnalyzer
        network_analyzer = NetworkAnalyzer(network)
        network_metrics = network_analyzer.analyze_global_properties()

        processed_results = {
            'simulation_id': simulation_id,
            'parameters': params,
            'total_rounds': total_rounds,
            'payoff_trends': payoff_trends,
            'final_metrics': final_metrics,
            'network_metrics': network_metrics,
            'raw_results': results if params.get('save_detailed_history', True) else [],
            'convergence_analysis': self._analyze_convergence(results),
            'timestamp': datetime.utcnow().isoformat()
        }

        return processed_results

    def _analyze_convergence(self, results: List[RoundResult]) -> Dict[str, Any]:
        """Analyze convergence properties of the simulation."""
        if len(results) < 10:
            return {"status": "insufficient_data"}

        # Simple convergence analysis based on payoff variance in recent rounds
        recent_results = results[-20:]  # Last 20 rounds

        payoff_variances = {}
        for player_type in ['spreader', 'fact_checker', 'platform']:
            recent_payoffs = [
                result.payoffs.get(player_type, 0)
                for result in recent_results
            ]
            if recent_payoffs:
                import numpy as np
                payoff_variances[player_type] = float(np.var(recent_payoffs))

        return {
            "status": "analyzed",
            "recent_payoff_variances": payoff_variances,
            "convergence_threshold": 0.1,  # Example threshold
            "converged": all(var < 0.1 for var in payoff_variances.values())
        }

    def _create_result_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of simulation results."""
        return {
            'total_rounds': results['total_rounds'],
            'network_size': results['network_metrics'].get('num_nodes', 0),
            'final_payoffs': results['final_metrics'].get('final_payoffs', {}),
            'converged': results['convergence_analysis'].get('converged', False),
            'timestamp': results['timestamp']
        }

    async def _save_simulation_results(self, simulation_id: str, results: Dict[str, Any]) -> None:
        """Save simulation results to file."""
        try:
            results_file = self.simulation_storage_path / f"{simulation_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved for simulation {simulation_id}")

        except Exception as e:
            logger.error(f"Failed to save results for simulation {simulation_id}: {e}")
            raise

    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get the current status of a simulation.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            Dictionary containing simulation status and metadata
        """
        if simulation_id not in simulation_jobs:
            raise ValueError(f"Simulation {simulation_id} not found")

        job = simulation_jobs[simulation_id]
        return {
            'simulation_id': simulation_id,
            'status': job['status'],
            'progress': job.get('progress', 0.0),
            'created_at': job['created_at'],
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'error': job.get('error'),
            'summary': job.get('summary', {})
        }

    async def get_simulation_results(self, simulation_id: str, include_details: bool = True) -> Dict[str, Any]:
        """
        Retrieve the results of a completed simulation.

        Args:
            simulation_id: Unique simulation identifier
            include_details: Whether to include detailed round-by-round results

        Returns:
            Dictionary containing simulation results
        """
        if simulation_id not in simulation_jobs:
            raise ValueError(f"Simulation {simulation_id} not found")

        job = simulation_jobs[simulation_id]
        if job['status'] != 'completed':
            raise ValueError(f"Simulation {simulation_id} is not completed (status: {job['status']})")

        try:
            # Load results from file
            results_file = self.simulation_storage_path / f"{simulation_id}_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found for simulation {simulation_id}")

            with open(results_file, 'r') as f:
                results = json.load(f)

            # Optionally exclude detailed results to reduce response size
            if not include_details:
                results.pop('raw_results', None)

            return results

        except Exception as e:
            logger.error(f"Failed to load results for simulation {simulation_id}: {e}")
            raise

    async def list_simulations(self,
                             status: Optional[str] = None,
                             limit: int = 50,
                             offset: int = 0) -> Dict[str, Any]:
        """
        List simulations with optional filtering.

        Args:
            status: Optional status filter ('pending', 'running', 'completed', 'failed')
            limit: Maximum number of simulations to return
            offset: Number of simulations to skip

        Returns:
            Dictionary containing simulation list and metadata
        """
        # Filter simulations
        filtered_sims = []
        for sim_id, sim_data in simulation_jobs.items():
            if status is None or sim_data['status'] == status:
                filtered_sims.append({
                    'simulation_id': sim_id,
                    'status': sim_data['status'],
                    'created_at': sim_data['created_at'],
                    'description': sim_data.get('params', {}).get('description', ''),
                    'summary': sim_data.get('summary', {})
                })

        # Sort by creation time (newest first)
        filtered_sims.sort(key=lambda x: x['created_at'], reverse=True)

        # Apply pagination
        paginated_sims = filtered_sims[offset:offset + limit]

        return {
            'simulations': paginated_sims,
            'total': len(filtered_sims),
            'limit': limit,
            'offset': offset,
            'has_more': len(filtered_sims) > offset + limit
        }

    async def cancel_simulation(self, simulation_id: str) -> bool:
        """
        Cancel a running simulation.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            True if cancellation was successful
        """
        if simulation_id not in simulation_jobs:
            raise ValueError(f"Simulation {simulation_id} not found")

        job = simulation_jobs[simulation_id]
        if job['status'] not in ['pending', 'running']:
            raise ValueError(f"Cannot cancel simulation with status: {job['status']}")

        # Update status
        simulation_jobs[simulation_id].update({
            'status': 'cancelled',
            'completed_at': datetime.utcnow().isoformat(),
            'error': 'Cancelled by user'
        })

        self._save_simulation_metadata()
        logger.info(f"Simulation {simulation_id} cancelled")
        return True

    async def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete simulation data.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            True if deletion was successful
        """
        if simulation_id not in simulation_jobs:
            raise ValueError(f"Simulation {simulation_id} not found")

        try:
            # Remove results file
            results_file = self.simulation_storage_path / f"{simulation_id}_results.json"
            if results_file.exists():
                results_file.unlink()

            # Remove from memory
            del simulation_jobs[simulation_id]

            # Save updated metadata
            self._save_simulation_metadata()

            logger.info(f"Simulation {simulation_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete simulation {simulation_id}: {e}")
            raise

    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get overall simulation statistics."""
        status_counts = {}
        for sim_data in simulation_jobs.values():
            status = sim_data['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_simulations': len(simulation_jobs),
            'status_breakdown': status_counts,
            'storage_path': str(self.simulation_storage_path)
        }