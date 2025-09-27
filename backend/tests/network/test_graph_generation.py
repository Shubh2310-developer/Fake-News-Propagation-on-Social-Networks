# backend/tests/network/test_graph_generation.py

import pytest
import networkx as nx
import numpy as np
from unittest.mock import patch, MagicMock

from network.graph_generator import NetworkGenerator, NetworkConfig
from network.metrics import NetworkMetrics


@pytest.mark.unit
class TestNetworkGenerator:
    """Test suite for network generation functionality."""

    @pytest.fixture
    def generator(self):
        """Create a NetworkGenerator instance."""
        return NetworkGenerator()

    @pytest.fixture
    def barabasi_config(self):
        """Configuration for Barabási-Albert network."""
        return NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=100,
            attachment_preference=3,
            random_seed=42
        )

    @pytest.fixture
    def watts_strogatz_config(self):
        """Configuration for Watts-Strogatz network."""
        return NetworkConfig(
            network_type="watts_strogatz",
            num_nodes=100,
            k=6,
            p=0.3,
            random_seed=42
        )

    @pytest.fixture
    def erdos_renyi_config(self):
        """Configuration for ErdQs-Rényi network."""
        return NetworkConfig(
            network_type="erdos_renyi",
            num_nodes=100,
            edge_probability=0.05,
            random_seed=42
        )

    def test_generate_barabasi_albert_network(self, generator, barabasi_config):
        """Test generation of Barabási-Albert network."""
        network = generator.generate_network(barabasi_config)

        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() == 100
        assert network.number_of_edges() > 0

        # Check scale-free property (power law degree distribution)
        degrees = [network.degree(n) for n in network.nodes()]
        assert max(degrees) > min(degrees)  # Should have degree variation

    def test_generate_watts_strogatz_network(self, generator, watts_strogatz_config):
        """Test generation of Watts-Strogatz network."""
        network = generator.generate_network(watts_strogatz_config)

        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() == 100
        assert network.number_of_edges() > 0

        # Check small-world properties
        clustering = nx.average_clustering(network)
        path_length = nx.average_shortest_path_length(network)

        assert clustering > 0  # Should have clustering
        assert path_length > 0  # Should be connected

    def test_generate_erdos_renyi_network(self, generator, erdos_renyi_config):
        """Test generation of ErdQs-Rényi network."""
        network = generator.generate_network(erdos_renyi_config)

        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() == 100

        # Check expected number of edges
        expected_edges = 0.05 * 100 * 99 / 2
        actual_edges = network.number_of_edges()
        assert abs(actual_edges - expected_edges) < expected_edges * 0.3  # Within 30%

    def test_network_reproducibility(self, generator, barabasi_config):
        """Test that networks are reproducible with same seed."""
        network1 = generator.generate_network(barabasi_config)
        network2 = generator.generate_network(barabasi_config)

        # Should have same structure
        assert network1.number_of_nodes() == network2.number_of_nodes()
        assert network1.number_of_edges() == network2.number_of_edges()

        # Edge sets should be identical
        edges1 = set(network1.edges())
        edges2 = set(network2.edges())
        assert edges1 == edges2

    def test_network_connectivity(self, generator, barabasi_config):
        """Test that generated networks are connected."""
        network = generator.generate_network(barabasi_config)

        assert nx.is_connected(network), "Generated network should be connected"

    def test_invalid_network_type(self, generator):
        """Test handling of invalid network type."""
        invalid_config = NetworkConfig(
            network_type="invalid_type",
            num_nodes=100,
            random_seed=42
        )

        with pytest.raises(ValueError, match="Unsupported network type"):
            generator.generate_network(invalid_config)

    def test_network_size_validation(self, generator):
        """Test validation of network size parameters."""
        invalid_config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=0,  # Invalid
            attachment_preference=3,
            random_seed=42
        )

        with pytest.raises(ValueError, match="num_nodes must be positive"):
            generator.generate_network(invalid_config)

    def test_attachment_preference_validation(self, generator):
        """Test validation of attachment preference for Barabási-Albert."""
        invalid_config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=100,
            attachment_preference=150,  # Too large
            random_seed=42
        )

        with pytest.raises(ValueError, match="attachment_preference must be less than num_nodes"):
            generator.generate_network(invalid_config)

    def test_network_properties_calculation(self, generator, barabasi_config):
        """Test calculation of network properties."""
        network = generator.generate_network(barabasi_config)
        properties = generator.calculate_network_properties(network)

        assert "num_nodes" in properties
        assert "num_edges" in properties
        assert "density" in properties
        assert "average_clustering" in properties
        assert "average_path_length" in properties

        assert properties["num_nodes"] == 100
        assert properties["density"] >= 0
        assert properties["average_clustering"] >= 0

    def test_degree_distribution_analysis(self, generator, barabasi_config):
        """Test degree distribution analysis."""
        network = generator.generate_network(barabasi_config)
        degree_dist = generator.analyze_degree_distribution(network)

        assert "mean_degree" in degree_dist
        assert "std_degree" in degree_dist
        assert "max_degree" in degree_dist
        assert "degree_histogram" in degree_dist

        assert degree_dist["mean_degree"] > 0
        assert degree_dist["max_degree"] >= degree_dist["mean_degree"]

    @pytest.mark.slow
    def test_large_network_generation(self, generator):
        """Test generation of large networks."""
        large_config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=5000,
            attachment_preference=5,
            random_seed=42
        )

        network = generator.generate_network(large_config)

        assert network.number_of_nodes() == 5000
        assert nx.is_connected(network)

    def test_custom_network_parameters(self, generator):
        """Test network generation with custom parameters."""
        custom_config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=50,
            attachment_preference=2,
            custom_params={"directed": False, "weighted": False},
            random_seed=42
        )

        network = generator.generate_network(custom_config)

        assert not network.is_directed()
        assert network.number_of_nodes() == 50

    def test_network_export_import(self, generator, barabasi_config):
        """Test network export and import functionality."""
        network = generator.generate_network(barabasi_config)

        # Test export
        export_data = generator.export_network(network, format="graphml")
        assert export_data is not None

        # Test import
        imported_network = generator.import_network(export_data, format="graphml")
        assert imported_network.number_of_nodes() == network.number_of_nodes()
        assert imported_network.number_of_edges() == network.number_of_edges()


@pytest.mark.unit
class TestNetworkConfig:
    """Test suite for network configuration."""

    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=100,
            attachment_preference=3,
            random_seed=42
        )

        config.validate()  # Should not raise exception

    def test_config_validation_missing_required(self):
        """Test validation with missing required parameters."""
        config = NetworkConfig(
            network_type="barabasi_albert",
            num_nodes=100,
            # Missing attachment_preference
            random_seed=42
        )

        with pytest.raises(ValueError, match="attachment_preference is required"):
            config.validate()

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = NetworkConfig(
            network_type="watts_strogatz",
            num_nodes=100,
            k=6,
            p=0.3,
            random_seed=42
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["network_type"] == "watts_strogatz"
        assert config_dict["num_nodes"] == 100

        # Test from_dict
        restored_config = NetworkConfig.from_dict(config_dict)
        assert restored_config.network_type == config.network_type
        assert restored_config.num_nodes == config.num_nodes

    def test_config_parameter_validation(self):
        """Test validation of specific parameters."""
        # Test edge probability for ErdQs-Rényi
        with pytest.raises(ValueError, match="edge_probability must be between 0 and 1"):
            NetworkConfig(
                network_type="erdos_renyi",
                num_nodes=100,
                edge_probability=1.5,  # Invalid
                random_seed=42
            )

        # Test k parameter for Watts-Strogatz
        with pytest.raises(ValueError, match="k must be even and less than num_nodes"):
            NetworkConfig(
                network_type="watts_strogatz",
                num_nodes=100,
                k=101,  # Too large
                p=0.3,
                random_seed=42
            )


@pytest.mark.unit
class TestNetworkMetrics:
    """Test suite for network metrics calculation."""

    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing."""
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        return G

    @pytest.fixture
    def metrics_calculator(self):
        """Create a NetworkMetrics instance."""
        return NetworkMetrics()

    def test_basic_metrics_calculation(self, metrics_calculator, sample_network):
        """Test calculation of basic network metrics."""
        metrics = metrics_calculator.calculate_basic_metrics(sample_network)

        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert "density" in metrics
        assert "average_degree" in metrics

        assert metrics["num_nodes"] == 50
        assert metrics["density"] >= 0
        assert metrics["density"] <= 1

    def test_centrality_metrics(self, metrics_calculator, sample_network):
        """Test calculation of centrality metrics."""
        centrality = metrics_calculator.calculate_centrality_metrics(sample_network)

        assert "degree_centrality" in centrality
        assert "betweenness_centrality" in centrality
        assert "closeness_centrality" in centrality
        assert "eigenvector_centrality" in centrality

        # Check that centrality values are dictionaries with node keys
        assert len(centrality["degree_centrality"]) == 50
        assert all(0 <= v <= 1 for v in centrality["degree_centrality"].values())

    def test_clustering_metrics(self, metrics_calculator, sample_network):
        """Test calculation of clustering metrics."""
        clustering = metrics_calculator.calculate_clustering_metrics(sample_network)

        assert "average_clustering" in clustering
        assert "global_clustering" in clustering
        assert "local_clustering" in clustering

        assert 0 <= clustering["average_clustering"] <= 1
        assert len(clustering["local_clustering"]) == 50

    def test_path_metrics(self, metrics_calculator, sample_network):
        """Test calculation of path-related metrics."""
        path_metrics = metrics_calculator.calculate_path_metrics(sample_network)

        assert "average_shortest_path_length" in path_metrics
        assert "diameter" in path_metrics
        assert "radius" in path_metrics

        assert path_metrics["diameter"] >= path_metrics["radius"]
        assert path_metrics["average_shortest_path_length"] > 0

    def test_community_detection(self, metrics_calculator, sample_network):
        """Test community detection algorithms."""
        communities = metrics_calculator.detect_communities(sample_network)

        assert "modularity" in communities
        assert "num_communities" in communities
        assert "community_assignments" in communities

        assert communities["num_communities"] > 0
        assert len(communities["community_assignments"]) == 50

    def test_small_world_properties(self, metrics_calculator):
        """Test detection of small-world properties."""
        # Create a known small-world network
        ws_network = nx.watts_strogatz_graph(100, 6, 0.3, seed=42)

        small_world = metrics_calculator.analyze_small_world_properties(ws_network)

        assert "clustering_coefficient" in small_world
        assert "characteristic_path_length" in small_world
        assert "small_world_coefficient" in small_world

        assert small_world["clustering_coefficient"] > 0

    def test_degree_distribution_fitting(self, metrics_calculator, sample_network):
        """Test fitting of degree distribution."""
        distribution = metrics_calculator.fit_degree_distribution(sample_network)

        assert "power_law_alpha" in distribution
        assert "exponential_lambda" in distribution
        assert "best_fit" in distribution

        assert distribution["power_law_alpha"] > 0

    def test_network_robustness(self, metrics_calculator, sample_network):
        """Test network robustness analysis."""
        robustness = metrics_calculator.analyze_robustness(sample_network)

        assert "random_failure_threshold" in robustness
        assert "targeted_attack_threshold" in robustness
        assert "connectivity_robustness" in robustness

        assert 0 <= robustness["random_failure_threshold"] <= 1

    @pytest.mark.slow
    def test_metrics_performance(self, metrics_calculator, performance_timer):
        """Test performance of metrics calculation."""
        timer = performance_timer(threshold_seconds=1.0)

        large_network = nx.barabasi_albert_graph(1000, 5)
        metrics_calculator.calculate_basic_metrics(large_network)

        elapsed = timer()
        assert elapsed < 1.0, f"Metrics calculation took {elapsed:.3f}s"

    def test_temporal_network_metrics(self, metrics_calculator):
        """Test metrics for temporal/dynamic networks."""
        # Create a sequence of network snapshots
        snapshots = [
            nx.barabasi_albert_graph(50, 2, seed=i) for i in range(5)
        ]

        temporal_metrics = metrics_calculator.analyze_temporal_evolution(snapshots)

        assert "evolution_metrics" in temporal_metrics
        assert "stability_indices" in temporal_metrics
        assert len(temporal_metrics["evolution_metrics"]) == 5

    def test_comparative_analysis(self, metrics_calculator, sample_network):
        """Test comparative analysis between networks."""
        # Create another network for comparison
        comparison_network = nx.erdos_renyi_graph(50, 0.1, seed=42)

        comparison = metrics_calculator.compare_networks(
            sample_network, comparison_network
        )

        assert "metric_differences" in comparison
        assert "similarity_score" in comparison
        assert "statistical_tests" in comparison

        assert 0 <= comparison["similarity_score"] <= 1