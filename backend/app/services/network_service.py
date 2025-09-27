# backend/app/services/network_service.py

import networkx as nx
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

from network import (
    SocialNetworkGenerator,
    NetworkConfig,
    NetworkAnalyzer,
    InformationPropagationSimulator,
    ContentProperties,
    PropagationModel,
    export_for_visualization,
    export_propagation_visualization,
    export_community_visualization
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class NetworkService:
    """Provides an interface for network analysis and visualization tasks."""

    def __init__(self):
        """Initialize the network service."""
        self.networks_storage_path = Path(getattr(settings, 'NETWORKS_STORAGE_PATH', 'networks'))
        self.networks_storage_path.mkdir(exist_ok=True)

        # In-memory cache for frequently accessed networks
        self.network_cache: Dict[str, nx.Graph] = {}
        self.cache_limit = 10  # Maximum networks to keep in memory

        # Load default networks
        self._initialize_default_networks()

        logger.info("NetworkService initialized")

    def _initialize_default_networks(self) -> None:
        """Initialize default networks for testing and demonstration."""
        try:
            # Create a small default network
            config = NetworkConfig(num_nodes=100, attachment_preference=3, random_seed=42)
            generator = SocialNetworkGenerator(config)
            default_network = generator.generate_network('barabasi_albert')

            self.network_cache['default'] = default_network
            logger.info("Default network created and cached")

        except Exception as e:
            logger.warning(f"Failed to create default network: {e}")

    async def generate_network(self, network_config: Dict[str, Any]) -> str:
        """
        Generate a new network with specified configuration.

        Args:
            network_config: Configuration parameters for network generation

        Returns:
            Unique network identifier
        """
        try:
            # Create network configuration
            config = NetworkConfig(
                num_nodes=network_config.get('num_nodes', 1000),
                attachment_preference=network_config.get('attachment_preference', 5),
                rewiring_probability=network_config.get('rewiring_probability', 0.1),
                edge_probability=network_config.get('edge_probability', 0.01),
                k_neighbors=network_config.get('k_neighbors', 10),
                random_seed=network_config.get('random_seed', None)
            )

            # Generate network
            generator = SocialNetworkGenerator(config)
            network_type = network_config.get('network_type', 'barabasi_albert')
            network = generator.generate_network(network_type)

            # Generate unique network ID
            network_id = f"{network_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(network_config)) % 10000}"

            # Save network to storage
            await self._save_network(network_id, network, network_config)

            # Add to cache
            self._add_to_cache(network_id, network)

            logger.info(f"Generated network {network_id} with {network.number_of_nodes()} nodes")
            return network_id

        except Exception as e:
            logger.error(f"Failed to generate network: {e}")
            raise

    async def get_network_metrics(self, network_id: str = 'default') -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a network.

        Args:
            network_id: Identifier of the network to analyze

        Returns:
            Dictionary containing network metrics
        """
        try:
            network = await self._get_network(network_id)
            if not network:
                raise ValueError(f"Network {network_id} not found")

            analyzer = NetworkAnalyzer(network)

            # Compute comprehensive metrics
            metrics = {
                "network_id": network_id,
                "timestamp": datetime.utcnow().isoformat(),
                "global_properties": analyzer.analyze_global_properties(),
                "centrality_measures": analyzer.compute_centrality_measures(),
                "community_detection": analyzer.detect_communities(),
                "user_attributes": analyzer.analyze_user_attributes(),
                "edge_attributes": analyzer.analyze_edge_attributes(),
                "resilience": analyzer.compute_network_resilience()
            }

            logger.info(f"Computed metrics for network {network_id}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to compute metrics for network {network_id}: {e}")
            raise

    async def get_visualization_data(self,
                                   network_id: str = 'default',
                                   layout_algorithm: str = 'spring',
                                   max_nodes: int = 1000) -> Dict[str, Any]:
        """
        Generate visualization data for a network.

        Args:
            network_id: Identifier of the network to visualize
            layout_algorithm: Layout algorithm to use
            max_nodes: Maximum nodes to include in visualization

        Returns:
            Dictionary containing visualization data
        """
        try:
            network = await self._get_network(network_id)
            if not network:
                raise ValueError(f"Network {network_id} not found")

            from network.visualization import VisualizationConfig
            config = VisualizationConfig(
                layout_algorithm=layout_algorithm,
                max_nodes=max_nodes
            )

            visualization_data = export_for_visualization(network, config)

            # Add metadata
            visualization_data["metadata"]["network_id"] = network_id
            visualization_data["metadata"]["generated_at"] = datetime.utcnow().isoformat()

            logger.info(f"Generated visualization data for network {network_id}")
            return visualization_data

        except Exception as e:
            logger.error(f"Failed to generate visualization for network {network_id}: {e}")
            raise

    async def simulate_propagation(self,
                                 network_id: str,
                                 propagation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate information propagation on a network.

        Args:
            network_id: Identifier of the network
            propagation_config: Configuration for propagation simulation

        Returns:
            Dictionary containing propagation results
        """
        try:
            network = await self._get_network(network_id)
            if not network:
                raise ValueError(f"Network {network_id} not found")

            # Create propagation simulator
            from network.propagation import PropagationConfig
            config = PropagationConfig(
                base_transmission_rate=propagation_config.get('base_transmission_rate', 0.1),
                time_steps=propagation_config.get('time_steps', 50),
                recovery_rate=propagation_config.get('recovery_rate', 0.05),
                random_seed=propagation_config.get('random_seed', None)
            )

            simulator = InformationPropagationSimulator(network, config)

            # Set up content properties
            content_props = ContentProperties(
                content_type=propagation_config.get('content_type', 'neutral'),
                content_quality=propagation_config.get('content_quality', 0.5),
                virality_factor=propagation_config.get('virality_factor', 1.0),
                emotional_appeal=propagation_config.get('emotional_appeal', 0.5)
            )

            # Determine initial spreaders
            initial_spreaders = set(propagation_config.get('initial_spreaders', []))
            if not initial_spreaders:
                # Use highest degree nodes as default initial spreaders
                degree_centrality = nx.degree_centrality(network)
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                num_spreaders = propagation_config.get('num_initial_spreaders', 5)
                initial_spreaders = set([node for node, _ in top_nodes[:num_spreaders]])

            # Run propagation simulation
            model_type = PropagationModel(propagation_config.get('model', 'independent_cascade'))
            results = simulator.simulate_propagation(initial_spreaders, content_props, model_type)

            # Analyze results
            analysis = simulator.analyze_propagation_results(results)

            final_results = {
                "network_id": network_id,
                "simulation_timestamp": datetime.utcnow().isoformat(),
                "configuration": propagation_config,
                "results": results,
                "analysis": analysis
            }

            logger.info(f"Completed propagation simulation on network {network_id}")
            return final_results

        except Exception as e:
            logger.error(f"Failed to simulate propagation on network {network_id}: {e}")
            raise

    async def get_propagation_visualization(self,
                                          network_id: str,
                                          propagation_results: Dict[str, Any],
                                          layout_algorithm: str = 'spring') -> Dict[str, Any]:
        """
        Generate visualization data for propagation results.

        Args:
            network_id: Identifier of the network
            propagation_results: Results from propagation simulation
            layout_algorithm: Layout algorithm to use

        Returns:
            Dictionary containing propagation visualization data
        """
        try:
            network = await self._get_network(network_id)
            if not network:
                raise ValueError(f"Network {network_id} not found")

            from network.visualization import VisualizationConfig
            config = VisualizationConfig(layout_algorithm=layout_algorithm)

            visualization_data = export_propagation_visualization(
                network, propagation_results, config
            )

            logger.info(f"Generated propagation visualization for network {network_id}")
            return visualization_data

        except Exception as e:
            logger.error(f"Failed to generate propagation visualization: {e}")
            raise

    async def get_community_visualization(self,
                                        network_id: str,
                                        community_method: str = 'louvain',
                                        layout_algorithm: str = 'spring') -> Dict[str, Any]:
        """
        Generate visualization data highlighting community structure.

        Args:
            network_id: Identifier of the network
            community_method: Community detection method to use
            layout_algorithm: Layout algorithm to use

        Returns:
            Dictionary containing community visualization data
        """
        try:
            network = await self._get_network(network_id)
            if not network:
                raise ValueError(f"Network {network_id} not found")

            # Detect communities
            analyzer = NetworkAnalyzer(network)
            community_results = analyzer.detect_communities()

            # Use the specified method or fall back to available method
            if community_method in community_results:
                partition = community_results[community_method]['partition']
            else:
                # Use first available method
                available_methods = list(community_results.keys())
                if available_methods:
                    partition = community_results[available_methods[0]]['partition']
                else:
                    raise ValueError("No community detection results available")

            # Generate visualization
            from network.visualization import VisualizationConfig
            config = VisualizationConfig(layout_algorithm=layout_algorithm)

            visualization_data = export_community_visualization(network, partition, config)

            # Add community metadata
            visualization_data["community_metadata"] = {
                "method_used": community_method,
                "num_communities": len(set(partition.values())),
                "modularity": community_results.get(community_method, {}).get('modularity', None)
            }

            logger.info(f"Generated community visualization for network {network_id}")
            return visualization_data

        except Exception as e:
            logger.error(f"Failed to generate community visualization: {e}")
            raise

    async def compare_networks(self, network_ids: List[str]) -> Dict[str, Any]:
        """
        Compare metrics across multiple networks.

        Args:
            network_ids: List of network identifiers to compare

        Returns:
            Dictionary containing comparison results
        """
        try:
            if len(network_ids) < 2:
                raise ValueError("At least two networks are required for comparison")

            comparison_results = {
                "networks": network_ids,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {},
                "summary": {}
            }

            # Collect metrics for each network
            all_metrics = {}
            for network_id in network_ids:
                try:
                    metrics = await self.get_network_metrics(network_id)
                    all_metrics[network_id] = metrics
                except Exception as e:
                    logger.warning(f"Failed to get metrics for network {network_id}: {e}")
                    continue

            if len(all_metrics) < 2:
                raise ValueError("Could not load metrics for at least two networks")

            # Extract comparable metrics
            comparable_metrics = [
                'num_nodes', 'num_edges', 'density', 'average_clustering',
                'diameter', 'average_shortest_path_length'
            ]

            for metric in comparable_metrics:
                metric_values = {}
                for network_id, metrics in all_metrics.items():
                    global_props = metrics.get('global_properties', {})
                    if metric in global_props:
                        metric_values[network_id] = global_props[metric]

                if metric_values:
                    comparison_results["metrics"][metric] = metric_values

            # Calculate summary statistics
            comparison_results["summary"] = self._calculate_comparison_summary(
                comparison_results["metrics"]
            )

            logger.info(f"Completed comparison of {len(network_ids)} networks")
            return comparison_results

        except Exception as e:
            logger.error(f"Failed to compare networks: {e}")
            raise

    async def list_networks(self) -> Dict[str, Any]:
        """
        List all available networks.

        Returns:
            Dictionary containing network list and metadata
        """
        try:
            networks = []

            # Add cached networks
            for network_id in self.network_cache.keys():
                network = self.network_cache[network_id]
                networks.append({
                    "network_id": network_id,
                    "num_nodes": network.number_of_nodes(),
                    "num_edges": network.number_of_edges(),
                    "cached": True
                })

            # Check for saved networks
            for network_file in self.networks_storage_path.glob("*_network.gexf"):
                network_id = network_file.stem.replace('_network', '')
                if network_id not in self.network_cache:
                    # Load metadata without loading full network
                    metadata_file = self.networks_storage_path / f"{network_id}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            networks.append({
                                "network_id": network_id,
                                "num_nodes": metadata.get('num_nodes', 'unknown'),
                                "num_edges": metadata.get('num_edges', 'unknown'),
                                "cached": False
                            })

            return {
                "networks": networks,
                "total_count": len(networks),
                "cached_count": len(self.network_cache)
            }

        except Exception as e:
            logger.error(f"Failed to list networks: {e}")
            raise

    async def delete_network(self, network_id: str) -> bool:
        """
        Delete a network from storage and cache.

        Args:
            network_id: Identifier of the network to delete

        Returns:
            True if deletion was successful
        """
        try:
            if network_id == 'default':
                raise ValueError("Cannot delete default network")

            # Remove from cache
            if network_id in self.network_cache:
                del self.network_cache[network_id]

            # Remove from storage
            network_file = self.networks_storage_path / f"{network_id}_network.gexf"
            metadata_file = self.networks_storage_path / f"{network_id}_metadata.json"

            if network_file.exists():
                network_file.unlink()

            if metadata_file.exists():
                metadata_file.unlink()

            logger.info(f"Network {network_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete network {network_id}: {e}")
            raise

    async def _get_network(self, network_id: str) -> Optional[nx.Graph]:
        """Retrieve a network from cache or storage."""
        # Check cache first
        if network_id in self.network_cache:
            return self.network_cache[network_id]

        # Try to load from storage
        network_file = self.networks_storage_path / f"{network_id}_network.gexf"
        if network_file.exists():
            try:
                network = nx.read_gexf(str(network_file))
                self._add_to_cache(network_id, network)
                return network
            except Exception as e:
                logger.error(f"Failed to load network {network_id} from storage: {e}")

        return None

    async def _save_network(self, network_id: str, network: nx.Graph, config: Dict[str, Any]) -> None:
        """Save a network to storage."""
        try:
            # Save network
            network_file = self.networks_storage_path / f"{network_id}_network.gexf"
            nx.write_gexf(network, str(network_file))

            # Save metadata
            metadata = {
                "network_id": network_id,
                "num_nodes": network.number_of_nodes(),
                "num_edges": network.number_of_edges(),
                "config": config,
                "created_at": datetime.utcnow().isoformat()
            }

            metadata_file = self.networks_storage_path / f"{network_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Network {network_id} saved to storage")

        except Exception as e:
            logger.error(f"Failed to save network {network_id}: {e}")
            raise

    def _add_to_cache(self, network_id: str, network: nx.Graph) -> None:
        """Add a network to cache with LRU eviction."""
        # Remove oldest item if cache is full
        if len(self.network_cache) >= self.cache_limit:
            oldest_id = next(iter(self.network_cache))
            del self.network_cache[oldest_id]

        self.network_cache[network_id] = network

    def _calculate_comparison_summary(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for network comparison."""
        summary = {}

        for metric_name, network_values in metrics.items():
            if not network_values:
                continue

            values = list(network_values.values())
            try:
                # Filter numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]

                if numeric_values:
                    import numpy as np
                    summary[metric_name] = {
                        "min": float(np.min(numeric_values)),
                        "max": float(np.max(numeric_values)),
                        "mean": float(np.mean(numeric_values)),
                        "std": float(np.std(numeric_values)),
                        "range": float(np.max(numeric_values) - np.min(numeric_values))
                    }

            except Exception as e:
                logger.warning(f"Failed to calculate summary for metric {metric_name}: {e}")

        return summary