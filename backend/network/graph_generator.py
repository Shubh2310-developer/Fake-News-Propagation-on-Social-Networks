# backend/network/graph_generator.py

import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration parameters for network generation."""
    num_nodes: int = 1000
    attachment_preference: int = 5
    rewiring_probability: float = 0.1
    edge_probability: float = 0.01
    k_neighbors: int = 10
    random_seed: Optional[int] = None

    # User attribute distribution parameters
    influence_distribution: str = 'log_normal'  # 'log_normal', 'power_law', 'uniform'
    credibility_distribution: str = 'beta'      # 'beta', 'normal', 'uniform'
    user_type_distribution: Dict[str, float] = None

    # Edge weight parameters
    trust_min: float = 0.2
    trust_max: float = 1.0
    interaction_strength_base: float = 0.5

    def __post_init__(self):
        if self.user_type_distribution is None:
            self.user_type_distribution = {
                'regular': 0.7,
                'influencer': 0.2,
                'bot': 0.05,
                'verified': 0.05
            }


class SocialNetworkGenerator:
    """Generates synthetic social network graphs with realistic attributes."""

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize the social network generator.

        Args:
            config: Configuration parameters for network generation
        """
        self.config = config or NetworkConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info(f"Initialized SocialNetworkGenerator with {self.config.num_nodes} nodes")

    def generate_network(self, network_type: str = 'barabasi_albert') -> nx.Graph:
        """
        Factory method to generate a network of a specific type.

        Args:
            network_type: Type of network to generate
                         Options: 'barabasi_albert', 'watts_strogatz', 'erdos_renyi', 'configuration'

        Returns:
            Generated NetworkX graph with user attributes and edge weights
        """
        logger.info(f"Generating {network_type} network")

        if network_type == 'barabasi_albert':
            graph = self._generate_barabasi_albert()
        elif network_type == 'watts_strogatz':
            graph = self._generate_watts_strogatz()
        elif network_type == 'erdos_renyi':
            graph = self._generate_erdos_renyi()
        elif network_type == 'configuration':
            graph = self._generate_configuration_model()
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Add realistic attributes
        self._add_user_attributes(graph)
        self._add_edge_weights(graph)

        logger.info(f"Generated network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph

    def _generate_barabasi_albert(self) -> nx.Graph:
        """
        Generates a scale-free network using the Barabási-Albert model.

        This model creates networks with power-law degree distribution,
        similar to many real-world social networks.

        Returns:
            NetworkX graph with scale-free properties
        """
        return nx.barabasi_albert_graph(
            n=self.config.num_nodes,
            m=self.config.attachment_preference,
            seed=self.config.random_seed
        )

    def _generate_watts_strogatz(self) -> nx.Graph:
        """
        Generates a small-world network using the Watts-Strogatz model.

        This model creates networks with high clustering and short path lengths,
        characteristic of many social networks.

        Returns:
            NetworkX graph with small-world properties
        """
        return nx.watts_strogatz_graph(
            n=self.config.num_nodes,
            k=self.config.k_neighbors,
            p=self.config.rewiring_probability,
            seed=self.config.random_seed
        )

    def _generate_erdos_renyi(self) -> nx.Graph:
        """
        Generates a random graph using the ErdQs-Rényi model.

        Returns:
            NetworkX random graph
        """
        return nx.erdos_renyi_graph(
            n=self.config.num_nodes,
            p=self.config.edge_probability,
            seed=self.config.random_seed
        )

    def _generate_configuration_model(self) -> nx.Graph:
        """
        Generates a network using the configuration model with specified degree sequence.

        This allows for more control over the degree distribution.

        Returns:
            NetworkX graph from configuration model
        """
        # Generate degree sequence following power law
        degrees = self._generate_power_law_degree_sequence()

        # Ensure sum of degrees is even
        if sum(degrees) % 2 != 0:
            degrees[0] += 1

        try:
            graph = nx.configuration_model(degrees, seed=self.config.random_seed)
            # Remove self-loops and multiple edges
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
        except nx.NetworkXError:
            logger.warning("Configuration model failed, falling back to Barabási-Albert")
            return self._generate_barabasi_albert()

    def _generate_power_law_degree_sequence(self) -> List[int]:
        """
        Generate a degree sequence following a power law distribution.

        Returns:
            List of degrees for each node
        """
        # Power law with exponent between 2-3 (typical for social networks)
        gamma = 2.5
        degrees = []

        for _ in range(self.config.num_nodes):
            # Generate power law distributed degree
            degree = int(np.random.pareto(gamma - 1) + 1)
            # Cap maximum degree to prevent unrealistic hubs
            max_degree = min(50, self.config.num_nodes // 10)
            degree = min(degree, max_degree)
            degrees.append(degree)

        return degrees

    def _add_user_attributes(self, graph: nx.Graph) -> None:
        """
        Adds realistic user attributes to each node in the graph.

        Args:
            graph: NetworkX graph to add attributes to
        """
        logger.debug("Adding user attributes to nodes")

        for node in graph.nodes():
            # Influence score based on network position and distribution type
            if self.config.influence_distribution == 'log_normal':
                base_influence = np.random.lognormal(0, 1)
            elif self.config.influence_distribution == 'power_law':
                base_influence = np.random.pareto(1) + 1
            else:  # uniform
                base_influence = np.random.uniform(0.1, 5.0)

            # Scale influence by degree (higher degree = higher influence)
            degree_factor = np.log1p(graph.degree(node))
            influence_score = base_influence * degree_factor

            # Normalize to reasonable range
            graph.nodes[node]['influence_score'] = min(influence_score / 10, 1.0)

            # Credibility score
            if self.config.credibility_distribution == 'beta':
                # Beta distribution with parameters favoring moderate credibility
                credibility_score = np.random.beta(2, 2)
            elif self.config.credibility_distribution == 'normal':
                credibility_score = np.clip(np.random.normal(0.6, 0.2), 0, 1)
            else:  # uniform
                credibility_score = np.random.uniform(0, 1)

            graph.nodes[node]['credibility_score'] = credibility_score

            # User type based on distribution
            user_type = np.random.choice(
                list(self.config.user_type_distribution.keys()),
                p=list(self.config.user_type_distribution.values())
            )
            graph.nodes[node]['user_type'] = user_type

            # Adjust attributes based on user type
            self._adjust_attributes_by_type(graph.nodes[node], user_type)

            # Additional attributes
            graph.nodes[node]['creation_time'] = np.random.randint(0, 365)  # Days ago
            graph.nodes[node]['activity_level'] = np.random.exponential(1.0)
            graph.nodes[node]['follower_count'] = int(graph.degree(node) * np.random.uniform(5, 50))

    def _adjust_attributes_by_type(self, node_data: Dict[str, Any], user_type: str) -> None:
        """
        Adjust node attributes based on user type.

        Args:
            node_data: Dictionary of node attributes
            user_type: Type of user
        """
        if user_type == 'influencer':
            # Influencers have higher influence but variable credibility
            node_data['influence_score'] *= 2.0
            node_data['influence_score'] = min(node_data['influence_score'], 1.0)

        elif user_type == 'bot':
            # Bots have moderate influence but low credibility
            node_data['credibility_score'] *= 0.3
            node_data['activity_level'] *= 3.0  # Bots are very active

        elif user_type == 'verified':
            # Verified users have high credibility
            node_data['credibility_score'] = min(node_data['credibility_score'] + 0.3, 1.0)

        # Regular users keep default attributes

    def _add_edge_weights(self, graph: nx.Graph) -> None:
        """
        Adds trust and interaction strength weights to edges.

        Args:
            graph: NetworkX graph to add edge weights to
        """
        logger.debug("Adding edge weights")

        for u, v in graph.edges():
            # Trust score between users
            # Higher trust if both users have high credibility
            u_credibility = graph.nodes[u]['credibility_score']
            v_credibility = graph.nodes[v]['credibility_score']

            # Base trust with some randomness
            base_trust = np.random.uniform(self.config.trust_min, self.config.trust_max)

            # Adjust based on credibility similarity
            credibility_similarity = 1 - abs(u_credibility - v_credibility)
            trust_adjustment = credibility_similarity * 0.3

            trust = min(base_trust + trust_adjustment, 1.0)
            graph.edges[u, v]['trust'] = trust

            # Interaction strength based on user types and activities
            u_activity = graph.nodes[u]['activity_level']
            v_activity = graph.nodes[v]['activity_level']

            interaction_strength = (
                self.config.interaction_strength_base *
                np.sqrt(u_activity * v_activity) *
                np.random.uniform(0.5, 1.5)
            )

            graph.edges[u, v]['interaction_strength'] = min(interaction_strength, 1.0)

            # Edge creation time (for temporal analysis)
            graph.edges[u, v]['creation_time'] = np.random.randint(0, 365)

    def get_network_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Calculate basic statistics for the generated network.

        Args:
            graph: NetworkX graph to analyze

        Returns:
            Dictionary containing network statistics
        """
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_degree': np.mean([d for n, d in graph.degree()]),
            'max_degree': max([d for n, d in graph.degree()]),
        }

        # Calculate clustering coefficient
        try:
            stats['average_clustering'] = nx.average_clustering(graph)
        except:
            stats['average_clustering'] = 0.0

        # Calculate diameter for largest connected component
        if nx.is_connected(graph):
            stats['diameter'] = nx.diameter(graph)
        else:
            largest_cc = max(nx.connected_components(graph), key=len)
            if len(largest_cc) > 1:
                subgraph = graph.subgraph(largest_cc)
                stats['diameter'] = nx.diameter(subgraph)
            else:
                stats['diameter'] = 0

        # User type distribution
        user_types = [graph.nodes[n]['user_type'] for n in graph.nodes()]
        unique_types, counts = np.unique(user_types, return_counts=True)
        stats['user_type_distribution'] = dict(zip(unique_types, counts.tolist()))

        # Attribute statistics
        influence_scores = [graph.nodes[n]['influence_score'] for n in graph.nodes()]
        credibility_scores = [graph.nodes[n]['credibility_score'] for n in graph.nodes()]

        stats['influence_stats'] = {
            'mean': np.mean(influence_scores),
            'std': np.std(influence_scores),
            'min': np.min(influence_scores),
            'max': np.max(influence_scores)
        }

        stats['credibility_stats'] = {
            'mean': np.mean(credibility_scores),
            'std': np.std(credibility_scores),
            'min': np.min(credibility_scores),
            'max': np.max(credibility_scores)
        }

        return stats

    def save_network(self, graph: nx.Graph, filepath: str, format: str = 'gexf') -> None:
        """
        Save the network to a file.

        Args:
            graph: NetworkX graph to save
            filepath: Path to save the file
            format: File format ('gexf', 'graphml', 'edgelist', 'adjlist')
        """
        try:
            if format == 'gexf':
                nx.write_gexf(graph, filepath)
            elif format == 'graphml':
                nx.write_graphml(graph, filepath)
            elif format == 'edgelist':
                nx.write_edgelist(graph, filepath, data=True)
            elif format == 'adjlist':
                nx.write_adjlist(graph, filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Network saved to {filepath} in {format} format")

        except Exception as e:
            logger.error(f"Failed to save network: {e}")
            raise

    @classmethod
    def load_network(cls, filepath: str, format: str = 'gexf') -> nx.Graph:
        """
        Load a network from a file.

        Args:
            filepath: Path to the network file
            format: File format

        Returns:
            Loaded NetworkX graph
        """
        try:
            if format == 'gexf':
                graph = nx.read_gexf(filepath)
            elif format == 'graphml':
                graph = nx.read_graphml(filepath)
            elif format == 'edgelist':
                graph = nx.read_edgelist(filepath, data=True)
            elif format == 'adjlist':
                graph = nx.read_adjlist(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Network loaded from {filepath}")
            return graph

        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            raise