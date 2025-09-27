# backend/network/metrics.py

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import community detection library
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    try:
        import networkx.algorithms.community as nx_community
        HAS_COMMUNITY = False
        logger.warning("python-louvain not available, using NetworkX community detection")
    except ImportError:
        HAS_COMMUNITY = False
        logger.warning("Community detection libraries not available")


class NetworkAnalyzer:
    """Computes structural metrics for a given network graph."""

    def __init__(self, network: nx.Graph):
        """
        Initialize the network analyzer.

        Args:
            network: NetworkX graph to analyze
        """
        if not isinstance(network, nx.Graph):
            raise TypeError("Input must be a NetworkX graph.")

        self.network = network
        self.num_nodes = network.number_of_nodes()
        self.num_edges = network.number_of_edges()

        logger.info(f"Initialized NetworkAnalyzer for graph with {self.num_nodes} nodes and {self.num_edges} edges")

    def analyze_global_properties(self) -> Dict[str, Any]:
        """
        Computes and returns global network statistics.

        Returns:
            Dictionary containing global network properties
        """
        logger.debug("Computing global network properties")

        properties = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': nx.density(self.network),
            'is_connected': nx.is_connected(self.network),
        }

        # Clustering coefficient
        try:
            properties['average_clustering'] = nx.average_clustering(self.network)
            properties['transitivity'] = nx.transitivity(self.network)
        except Exception as e:
            logger.warning(f"Error computing clustering metrics: {e}")
            properties['average_clustering'] = 0.0
            properties['transitivity'] = 0.0

        # Path metrics for connected components
        if nx.is_connected(self.network):
            try:
                properties['diameter'] = nx.diameter(self.network)
                properties['radius'] = nx.radius(self.network)
                properties['average_shortest_path_length'] = nx.average_shortest_path_length(self.network)
            except Exception as e:
                logger.warning(f"Error computing path metrics: {e}")
                properties['diameter'] = None
                properties['radius'] = None
                properties['average_shortest_path_length'] = None
        else:
            # Analyze largest connected component
            largest_cc = max(nx.connected_components(self.network), key=len)
            properties['largest_component_size'] = len(largest_cc)
            properties['largest_component_fraction'] = len(largest_cc) / self.num_nodes

            if len(largest_cc) > 1:
                subgraph = self.network.subgraph(largest_cc)
                try:
                    properties['diameter'] = nx.diameter(subgraph)
                    properties['radius'] = nx.radius(subgraph)
                    properties['average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
                except Exception as e:
                    logger.warning(f"Error computing path metrics for largest component: {e}")
                    properties['diameter'] = None
                    properties['radius'] = None
                    properties['average_shortest_path_length'] = None

        # Degree statistics
        degrees = [d for n, d in self.network.degree()]
        if degrees:
            properties['degree_statistics'] = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': np.min(degrees),
                'max': np.max(degrees),
                'median': np.median(degrees)
            }

            # Degree distribution analysis
            properties['degree_distribution'] = self._analyze_degree_distribution(degrees)

        # Assortativity
        try:
            properties['degree_assortativity'] = nx.degree_assortativity_coefficient(self.network)
        except Exception as e:
            logger.warning(f"Error computing assortativity: {e}")
            properties['degree_assortativity'] = None

        # Small-world properties
        properties['small_world_metrics'] = self._compute_small_world_metrics()

        return properties

    def compute_centrality_measures(self) -> Dict[str, Dict[int, float]]:
        """
        Computes all major centrality measures for each node.

        Returns:
            Dictionary mapping centrality measure names to node-centrality dictionaries
        """
        logger.debug("Computing centrality measures")

        centralities = {}

        try:
            # Degree centrality (fastest)
            centralities['degree'] = nx.degree_centrality(self.network)

            # Closeness centrality
            centralities['closeness'] = nx.closeness_centrality(self.network)

            # Betweenness centrality (computationally expensive)
            if self.num_nodes <= 5000:  # Only compute for reasonably sized networks
                centralities['betweenness'] = nx.betweenness_centrality(self.network)
            else:
                # Use approximation for large networks
                centralities['betweenness'] = nx.betweenness_centrality(
                    self.network, k=min(500, self.num_nodes // 10)
                )

            # Eigenvector centrality
            try:
                centralities['eigenvector'] = nx.eigenvector_centrality(
                    self.network, max_iter=1000, tol=1e-6
                )
            except nx.NetworkXError:
                logger.warning("Eigenvector centrality did not converge, using PageRank instead")
                centralities['pagerank'] = nx.pagerank(self.network)

            # Additional centrality measures
            centralities['load'] = nx.load_centrality(self.network)

            # Katz centrality
            try:
                centralities['katz'] = nx.katz_centrality(self.network, max_iter=1000)
            except (nx.NetworkXError, np.linalg.LinAlgError):
                logger.warning("Katz centrality computation failed")

        except Exception as e:
            logger.error(f"Error computing centrality measures: {e}")
            raise

        return centralities

    def detect_communities(self) -> Dict[str, Any]:
        """
        Detects communities using various methods.

        Returns:
            Dictionary containing community detection results
        """
        logger.debug("Detecting communities")

        community_results = {}

        if HAS_COMMUNITY and 'community_louvain' in globals():
            # Louvain method (best for large networks)
            try:
                partition = community_louvain.best_partition(self.network)
                modularity = community_louvain.modularity(partition, self.network)

                community_results['louvain'] = {
                    'partition': partition,
                    'modularity': modularity,
                    'num_communities': len(set(partition.values())),
                    'community_sizes': self._get_community_sizes(partition)
                }
            except Exception as e:
                logger.warning(f"Louvain community detection failed: {e}")

        # NetworkX built-in community detection methods
        try:
            # Greedy modularity communities
            communities_greedy = nx.community.greedy_modularity_communities(self.network)
            partition_greedy = {}
            for i, community in enumerate(communities_greedy):
                for node in community:
                    partition_greedy[node] = i

            modularity_greedy = nx.community.modularity(self.network, communities_greedy)

            community_results['greedy_modularity'] = {
                'partition': partition_greedy,
                'modularity': modularity_greedy,
                'num_communities': len(communities_greedy),
                'community_sizes': [len(c) for c in communities_greedy]
            }

        except Exception as e:
            logger.warning(f"Greedy modularity community detection failed: {e}")

        # Label propagation algorithm
        try:
            communities_label = nx.community.label_propagation_communities(self.network)
            partition_label = {}
            for i, community in enumerate(communities_label):
                for node in community:
                    partition_label[node] = i

            community_results['label_propagation'] = {
                'partition': partition_label,
                'num_communities': len(list(communities_label)),
                'community_sizes': [len(c) for c in communities_label]
            }

        except Exception as e:
            logger.warning(f"Label propagation community detection failed: {e}")

        return community_results

    def analyze_user_attributes(self) -> Dict[str, Any]:
        """
        Analyze the distribution and relationships of user attributes.

        Returns:
            Dictionary containing user attribute analysis
        """
        logger.debug("Analyzing user attributes")

        analysis = {}

        # Extract all node attributes
        node_attributes = {}
        for node, data in self.network.nodes(data=True):
            for attr, value in data.items():
                if attr not in node_attributes:
                    node_attributes[attr] = []
                node_attributes[attr].append(value)

        # Analyze each attribute
        for attr_name, values in node_attributes.items():
            if not values:
                continue

            attr_analysis = {}

            # Check if numeric or categorical
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if len(numeric_values) == len(values):
                    # Numeric attribute
                    attr_analysis['type'] = 'numeric'
                    attr_analysis['statistics'] = {
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'median': np.median(numeric_values),
                        'quartiles': np.percentile(numeric_values, [25, 50, 75]).tolist()
                    }

                    # Correlation with centrality measures
                    attr_analysis['centrality_correlations'] = self._compute_attribute_centrality_correlations(
                        attr_name, numeric_values
                    )
                else:
                    raise ValueError("Not all numeric")

            except (ValueError, TypeError):
                # Categorical attribute
                attr_analysis['type'] = 'categorical'
                unique_values, counts = np.unique(values, return_counts=True)
                attr_analysis['distribution'] = dict(zip(unique_values, counts.tolist()))
                attr_analysis['entropy'] = self._compute_entropy(counts)

            analysis[attr_name] = attr_analysis

        return analysis

    def analyze_edge_attributes(self) -> Dict[str, Any]:
        """
        Analyze the distribution of edge attributes.

        Returns:
            Dictionary containing edge attribute analysis
        """
        logger.debug("Analyzing edge attributes")

        analysis = {}

        # Extract all edge attributes
        edge_attributes = {}
        for u, v, data in self.network.edges(data=True):
            for attr, value in data.items():
                if attr not in edge_attributes:
                    edge_attributes[attr] = []
                edge_attributes[attr].append(value)

        # Analyze each attribute
        for attr_name, values in edge_attributes.items():
            if not values:
                continue

            attr_analysis = {}

            try:
                numeric_values = [float(v) for v in values if v is not None]
                if len(numeric_values) == len(values):
                    # Numeric attribute
                    attr_analysis['type'] = 'numeric'
                    attr_analysis['statistics'] = {
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'median': np.median(numeric_values)
                    }
                else:
                    raise ValueError("Not all numeric")

            except (ValueError, TypeError):
                # Categorical attribute
                unique_values, counts = np.unique(values, return_counts=True)
                attr_analysis['distribution'] = dict(zip(unique_values, counts.tolist()))

            analysis[attr_name] = attr_analysis

        return analysis

    def compute_network_resilience(self) -> Dict[str, Any]:
        """
        Compute network resilience metrics.

        Returns:
            Dictionary containing resilience analysis
        """
        logger.debug("Computing network resilience")

        resilience = {}

        # Node connectivity
        try:
            resilience['node_connectivity'] = nx.node_connectivity(self.network)
            resilience['edge_connectivity'] = nx.edge_connectivity(self.network)
        except Exception as e:
            logger.warning(f"Error computing connectivity: {e}")
            resilience['node_connectivity'] = None
            resilience['edge_connectivity'] = None

        # Robustness to random failures
        resilience['random_failure_resilience'] = self._simulate_random_failures()

        # Robustness to targeted attacks
        resilience['targeted_attack_resilience'] = self._simulate_targeted_attacks()

        return resilience

    def _analyze_degree_distribution(self, degrees: List[int]) -> Dict[str, Any]:
        """Analyze the degree distribution of the network."""
        degree_counts = defaultdict(int)
        for degree in degrees:
            degree_counts[degree] += 1

        # Check if follows power law
        degrees_array = np.array(degrees)
        log_degrees = np.log(degrees_array[degrees_array > 0])

        # Simple power law fit
        if len(log_degrees) > 1:
            log_freq = np.log(np.histogram(degrees_array, bins=max(degrees))[0] + 1)
            # Remove zeros
            valid_indices = log_freq > 0
            if np.sum(valid_indices) > 1:
                slope, intercept = np.polyfit(
                    np.log(np.arange(1, len(log_freq) + 1))[valid_indices],
                    log_freq[valid_indices],
                    1
                )
                power_law_exponent = -slope
            else:
                power_law_exponent = None
        else:
            power_law_exponent = None

        return {
            'degree_counts': dict(degree_counts),
            'power_law_exponent': power_law_exponent,
            'gini_coefficient': self._compute_gini_coefficient(degrees)
        }

    def _compute_small_world_metrics(self) -> Dict[str, Any]:
        """Compute small-world network metrics."""
        try:
            # Clustering coefficient
            C = nx.average_clustering(self.network)

            # Characteristic path length (only for connected components)
            if nx.is_connected(self.network):
                L = nx.average_shortest_path_length(self.network)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(self.network), key=len)
                if len(largest_cc) > 1:
                    subgraph = self.network.subgraph(largest_cc)
                    L = nx.average_shortest_path_length(subgraph)
                else:
                    L = 0

            # Compare to random network
            n = self.network.number_of_nodes()
            m = self.network.number_of_edges()
            p = 2 * m / (n * (n - 1))  # Edge probability for random graph

            # Random network metrics (theoretical)
            C_random = p
            L_random = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')

            # Small-world coefficients
            sigma = (C / C_random) / (L / L_random) if L_random > 0 and C_random > 0 else None
            omega = L_random / L - C / C_random if L > 0 and C_random > 0 else None

            return {
                'clustering_coefficient': C,
                'characteristic_path_length': L,
                'clustering_coefficient_random': C_random,
                'characteristic_path_length_random': L_random,
                'small_world_sigma': sigma,
                'small_world_omega': omega
            }

        except Exception as e:
            logger.warning(f"Error computing small-world metrics: {e}")
            return {}

    def _get_community_sizes(self, partition: Dict[int, int]) -> List[int]:
        """Get sizes of communities from partition."""
        community_counts = defaultdict(int)
        for node, community_id in partition.items():
            community_counts[community_id] += 1
        return list(community_counts.values())

    def _compute_attribute_centrality_correlations(self, attr_name: str, values: List[float]) -> Dict[str, float]:
        """Compute correlations between node attributes and centrality measures."""
        correlations = {}

        try:
            # Get degree centrality
            degree_centrality = nx.degree_centrality(self.network)
            degree_values = [degree_centrality[node] for node in self.network.nodes()]

            if len(values) == len(degree_values):
                correlation = np.corrcoef(values, degree_values)[0, 1]
                if not np.isnan(correlation):
                    correlations['degree_centrality'] = correlation

        except Exception as e:
            logger.warning(f"Error computing correlations for {attr_name}: {e}")

        return correlations

    def _compute_entropy(self, counts: np.ndarray) -> float:
        """Compute Shannon entropy of a distribution."""
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _compute_gini_coefficient(self, values: List[int]) -> float:
        """Compute Gini coefficient for degree distribution."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    def _simulate_random_failures(self, fraction: float = 0.5) -> Dict[str, Any]:
        """Simulate random node failures and measure network fragmentation."""
        num_to_remove = int(self.num_nodes * fraction)
        nodes_to_remove = np.random.choice(
            list(self.network.nodes()),
            size=min(num_to_remove, self.num_nodes),
            replace=False
        )

        # Create copy and remove nodes
        network_copy = self.network.copy()
        network_copy.remove_nodes_from(nodes_to_remove)

        # Analyze remaining network
        if network_copy.number_of_nodes() > 0:
            largest_component_size = len(max(nx.connected_components(network_copy), key=len))
            fragmentation = 1 - (largest_component_size / self.num_nodes)
        else:
            fragmentation = 1.0

        return {
            'nodes_removed': len(nodes_to_remove),
            'fraction_removed': len(nodes_to_remove) / self.num_nodes,
            'fragmentation': fragmentation,
            'remaining_nodes': network_copy.number_of_nodes()
        }

    def _simulate_targeted_attacks(self, fraction: float = 0.1) -> Dict[str, Any]:
        """Simulate targeted attacks on high-degree nodes."""
        # Get nodes sorted by degree
        degree_centrality = nx.degree_centrality(self.network)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

        num_to_remove = int(self.num_nodes * fraction)
        nodes_to_remove = [node for node, _ in sorted_nodes[:num_to_remove]]

        # Create copy and remove nodes
        network_copy = self.network.copy()
        network_copy.remove_nodes_from(nodes_to_remove)

        # Analyze remaining network
        if network_copy.number_of_nodes() > 0:
            largest_component_size = len(max(nx.connected_components(network_copy), key=len))
            fragmentation = 1 - (largest_component_size / self.num_nodes)
        else:
            fragmentation = 1.0

        return {
            'nodes_removed': len(nodes_to_remove),
            'fraction_removed': len(nodes_to_remove) / self.num_nodes,
            'fragmentation': fragmentation,
            'remaining_nodes': network_copy.number_of_nodes(),
            'attack_type': 'high_degree_nodes'
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report of the network.

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Generating comprehensive network analysis report")

        report = {
            'network_info': {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'graph_type': type(self.network).__name__
            }
        }

        try:
            report['global_properties'] = self.analyze_global_properties()
        except Exception as e:
            logger.error(f"Error in global properties analysis: {e}")
            report['global_properties'] = {'error': str(e)}

        try:
            report['centrality_measures'] = self.compute_centrality_measures()
        except Exception as e:
            logger.error(f"Error in centrality analysis: {e}")
            report['centrality_measures'] = {'error': str(e)}

        try:
            report['community_detection'] = self.detect_communities()
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            report['community_detection'] = {'error': str(e)}

        try:
            report['user_attributes'] = self.analyze_user_attributes()
        except Exception as e:
            logger.error(f"Error in user attribute analysis: {e}")
            report['user_attributes'] = {'error': str(e)}

        try:
            report['edge_attributes'] = self.analyze_edge_attributes()
        except Exception as e:
            logger.error(f"Error in edge attribute analysis: {e}")
            report['edge_attributes'] = {'error': str(e)}

        try:
            report['resilience'] = self.compute_network_resilience()
        except Exception as e:
            logger.error(f"Error in resilience analysis: {e}")
            report['resilience'] = {'error': str(e)}

        logger.info("Comprehensive network analysis report completed")
        return report