# backend/network/visualization.py

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for network visualization export."""
    layout_algorithm: str = 'spring'
    scale_factor: float = 100.0
    node_size_attribute: str = 'influence_score'
    node_color_attribute: str = 'user_type'
    edge_width_attribute: str = 'interaction_strength'
    edge_color_attribute: str = 'trust'
    max_nodes: int = 1000  # Maximum nodes to visualize
    include_edge_labels: bool = False
    normalize_positions: bool = True


def export_for_visualization(graph: nx.Graph,
                           config: Optional[VisualizationConfig] = None) -> Dict[str, List[Dict]]:
    """
    Converts a NetworkX graph into a JSON format suitable for frontend libraries
    like D3.js or vis.js.

    Args:
        graph: NetworkX graph to export
        config: Configuration for visualization export

    Returns:
        Dictionary with 'nodes' and 'edges' lists suitable for frontend visualization
    """
    if config is None:
        config = VisualizationConfig()

    logger.info(f"Exporting graph with {graph.number_of_nodes()} nodes for visualization")

    # Handle large graphs by sampling
    if graph.number_of_nodes() > config.max_nodes:
        logger.warning(f"Graph has {graph.number_of_nodes()} nodes, sampling {config.max_nodes} for visualization")
        graph = _sample_graph_for_visualization(graph, config.max_nodes)

    # Compute layout positions
    positions = _compute_layout(graph, config.layout_algorithm)

    # Scale and normalize positions if requested
    if config.normalize_positions:
        positions = _normalize_positions(positions, config.scale_factor)

    # Export nodes
    nodes_data = _export_nodes(graph, positions, config)

    # Export edges
    edges_data = _export_edges(graph, config)

    result = {
        "nodes": nodes_data,
        "edges": edges_data,
        "metadata": {
            "original_node_count": graph.number_of_nodes(),
            "original_edge_count": graph.number_of_edges(),
            "layout_algorithm": config.layout_algorithm,
            "scale_factor": config.scale_factor
        }
    }

    logger.info(f"Exported {len(nodes_data)} nodes and {len(edges_data)} edges for visualization")
    return result


def _sample_graph_for_visualization(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    """
    Sample a subgraph for visualization when the original is too large.

    Uses a combination of high-degree nodes and random sampling to maintain
    network structure while reducing size.

    Args:
        graph: Original graph
        max_nodes: Maximum number of nodes to include

    Returns:
        Sampled subgraph
    """
    if graph.number_of_nodes() <= max_nodes:
        return graph

    # Strategy: Take top nodes by degree and add random nodes
    degree_centrality = nx.degree_centrality(graph)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # Take top 70% of max_nodes based on degree
    high_degree_count = int(max_nodes * 0.7)
    high_degree_nodes = [node for node, _ in sorted_nodes[:high_degree_count]]

    # Take random 30% from remaining nodes
    remaining_nodes = [node for node, _ in sorted_nodes[high_degree_count:]]
    random_count = max_nodes - high_degree_count

    if len(remaining_nodes) > random_count:
        random_nodes = np.random.choice(remaining_nodes, size=random_count, replace=False).tolist()
    else:
        random_nodes = remaining_nodes

    selected_nodes = high_degree_nodes + random_nodes
    return graph.subgraph(selected_nodes).copy()


def _compute_layout(graph: nx.Graph, layout_algo: str = 'spring') -> Dict[int, Tuple[float, float]]:
    """
    Compute node positions using specified layout algorithm.

    Args:
        graph: NetworkX graph
        layout_algo: Layout algorithm name

    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    logger.debug(f"Computing {layout_algo} layout for {graph.number_of_nodes()} nodes")

    try:
        if layout_algo == 'spring':
            # Spring layout with more iterations for better results
            positions = nx.spring_layout(graph, iterations=50, k=1/np.sqrt(graph.number_of_nodes()))
        elif layout_algo == 'kamada_kawai':
            positions = nx.kamada_kawai_layout(graph)
        elif layout_algo == 'circular':
            positions = nx.circular_layout(graph)
        elif layout_algo == 'shell':
            positions = nx.shell_layout(graph)
        elif layout_algo == 'spectral':
            positions = nx.spectral_layout(graph)
        elif layout_algo == 'random':
            positions = nx.random_layout(graph)
        elif layout_algo == 'fruchterman_reingold':
            positions = nx.fruchterman_reingold_layout(graph, iterations=50)
        else:
            logger.warning(f"Unknown layout algorithm {layout_algo}, using spring layout")
            positions = nx.spring_layout(graph, iterations=50)

    except Exception as e:
        logger.warning(f"Layout computation failed: {e}, falling back to random layout")
        positions = nx.random_layout(graph)

    return positions


def _normalize_positions(positions: Dict[int, Tuple[float, float]],
                        scale_factor: float) -> Dict[int, Tuple[float, float]]:
    """
    Normalize and scale position coordinates.

    Args:
        positions: Dictionary of node positions
        scale_factor: Scaling factor for coordinates

    Returns:
        Normalized and scaled positions
    """
    if not positions:
        return positions

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]

    # Calculate bounds
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Normalize to [0, 1] and then scale
    normalized_positions = {}
    for node, (x, y) in positions.items():
        if x_max != x_min:
            norm_x = (x - x_min) / (x_max - x_min)
        else:
            norm_x = 0.5

        if y_max != y_min:
            norm_y = (y - y_min) / (y_max - y_min)
        else:
            norm_y = 0.5

        # Scale and center around origin
        scaled_x = (norm_x - 0.5) * scale_factor
        scaled_y = (norm_y - 0.5) * scale_factor

        normalized_positions[node] = (scaled_x, scaled_y)

    return normalized_positions


def _export_nodes(graph: nx.Graph,
                 positions: Dict[int, Tuple[float, float]],
                 config: VisualizationConfig) -> List[Dict]:
    """
    Export node data for visualization.

    Args:
        graph: NetworkX graph
        positions: Node positions
        config: Visualization configuration

    Returns:
        List of node dictionaries for frontend
    """
    nodes_data = []

    # Get attribute ranges for normalization
    size_values = []
    for node in graph.nodes():
        size_attr = graph.nodes[node].get(config.node_size_attribute, 1.0)
        try:
            size_values.append(float(size_attr))
        except (ValueError, TypeError):
            size_values.append(1.0)

    # Normalize size values
    if size_values:
        min_size, max_size = min(size_values), max(size_values)
        size_range = max_size - min_size if max_size != min_size else 1.0
    else:
        min_size, size_range = 0.0, 1.0

    for i, (node, data) in enumerate(graph.nodes(data=True)):
        # Position
        pos = positions.get(node, (0, 0))

        # Size based on attribute
        size_attr = data.get(config.node_size_attribute, 1.0)
        try:
            normalized_size = (float(size_attr) - min_size) / size_range
            # Scale to reasonable range for visualization
            node_size = 5 + normalized_size * 15  # Size between 5 and 20
        except (ValueError, TypeError):
            node_size = 10  # Default size

        # Color/group based on attribute
        color_attr = data.get(config.node_color_attribute, 'unknown')

        # Create node data
        node_data = {
            'id': node,
            'label': str(node),
            'x': float(pos[0]),
            'y': float(pos[1]),
            'size': float(node_size),
            'group': str(color_attr),
            'title': _create_node_tooltip(node, data),
        }

        # Add all node attributes for potential use
        for attr, value in data.items():
            # Convert to JSON-serializable types
            if isinstance(value, (np.integer, np.floating)):
                node_data[attr] = float(value)
            elif isinstance(value, (list, dict)):
                node_data[attr] = value
            else:
                node_data[attr] = str(value)

        nodes_data.append(node_data)

    return nodes_data


def _export_edges(graph: nx.Graph, config: VisualizationConfig) -> List[Dict]:
    """
    Export edge data for visualization.

    Args:
        graph: NetworkX graph
        config: Visualization configuration

    Returns:
        List of edge dictionaries for frontend
    """
    edges_data = []

    # Get attribute ranges for normalization
    width_values = []
    for u, v in graph.edges():
        width_attr = graph.edges[u, v].get(config.edge_width_attribute, 1.0)
        try:
            width_values.append(float(width_attr))
        except (ValueError, TypeError):
            width_values.append(1.0)

    # Normalize width values
    if width_values:
        min_width, max_width = min(width_values), max(width_values)
        width_range = max_width - min_width if max_width != min_width else 1.0
    else:
        min_width, width_range = 0.0, 1.0

    for u, v, data in graph.edges(data=True):
        # Width based on attribute
        width_attr = data.get(config.edge_width_attribute, 1.0)
        try:
            normalized_width = (float(width_attr) - min_width) / width_range
            # Scale to reasonable range for visualization
            edge_width = 1 + normalized_width * 4  # Width between 1 and 5
        except (ValueError, TypeError):
            edge_width = 2  # Default width

        # Color based on attribute
        color_attr = data.get(config.edge_color_attribute, 0.5)

        # Create edge data
        edge_data = {
            'from': u,
            'to': v,
            'width': float(edge_width),
            'color': _edge_color_from_value(color_attr),
            'title': _create_edge_tooltip(u, v, data) if config.include_edge_labels else None,
        }

        # Add all edge attributes for potential use
        for attr, value in data.items():
            # Convert to JSON-serializable types
            if isinstance(value, (np.integer, np.floating)):
                edge_data[attr] = float(value)
            elif isinstance(value, (list, dict)):
                edge_data[attr] = value
            else:
                edge_data[attr] = str(value)

        edges_data.append(edge_data)

    return edges_data


def _create_node_tooltip(node: int, data: Dict[str, Any]) -> str:
    """Create HTML tooltip for a node."""
    tooltip_parts = [f"<b>Node {node}</b>"]

    # Add key attributes
    key_attrs = ['user_type', 'influence_score', 'credibility_score', 'follower_count']
    for attr in key_attrs:
        if attr in data:
            value = data[attr]
            if isinstance(value, float):
                tooltip_parts.append(f"{attr.replace('_', ' ').title()}: {value:.3f}")
            else:
                tooltip_parts.append(f"{attr.replace('_', ' ').title()}: {value}")

    return "<br/>".join(tooltip_parts)


def _create_edge_tooltip(u: int, v: int, data: Dict[str, Any]) -> str:
    """Create HTML tooltip for an edge."""
    tooltip_parts = [f"<b>Edge {u}  {v}</b>"]

    # Add key attributes
    key_attrs = ['trust', 'interaction_strength', 'creation_time']
    for attr in key_attrs:
        if attr in data:
            value = data[attr]
            if isinstance(value, float):
                tooltip_parts.append(f"{attr.replace('_', ' ').title()}: {value:.3f}")
            else:
                tooltip_parts.append(f"{attr.replace('_', ' ').title()}: {value}")

    return "<br/>".join(tooltip_parts)


def _edge_color_from_value(value: Any) -> str:
    """Convert edge attribute value to color."""
    try:
        numeric_value = float(value)
        # Map to color scale (red to green)
        if numeric_value <= 0.5:
            # Red to yellow
            red = 255
            green = int(255 * (numeric_value * 2))
            blue = 0
        else:
            # Yellow to green
            red = int(255 * (2 - numeric_value * 2))
            green = 255
            blue = 0

        return f"rgb({red}, {green}, {blue})"

    except (ValueError, TypeError):
        return "rgb(128, 128, 128)"  # Gray default


def export_subgraph_for_visualization(graph: nx.Graph,
                                    center_nodes: List[int],
                                    radius: int = 1,
                                    config: Optional[VisualizationConfig] = None) -> Dict[str, List[Dict]]:
    """
    Export a subgraph centered around specific nodes for visualization.

    Args:
        graph: Full NetworkX graph
        center_nodes: List of nodes to center the subgraph around
        radius: Number of hops to include from center nodes
        config: Visualization configuration

    Returns:
        Visualization data for the subgraph
    """
    if config is None:
        config = VisualizationConfig()

    logger.info(f"Exporting subgraph around {len(center_nodes)} center nodes with radius {radius}")

    # Get subgraph
    subgraph_nodes = set(center_nodes)

    for _ in range(radius):
        new_nodes = set()
        for node in subgraph_nodes:
            if node in graph:
                new_nodes.update(graph.neighbors(node))
        subgraph_nodes.update(new_nodes)

    subgraph = graph.subgraph(subgraph_nodes).copy()

    # Mark center nodes in the subgraph
    for node in center_nodes:
        if node in subgraph:
            subgraph.nodes[node]['is_center'] = True

    return export_for_visualization(subgraph, config)


def export_propagation_visualization(graph: nx.Graph,
                                   propagation_results: Dict[str, Any],
                                   config: Optional[VisualizationConfig] = None) -> Dict[str, Any]:
    """
    Export network with propagation overlay for visualization.

    Args:
        graph: NetworkX graph
        propagation_results: Results from propagation simulation
        config: Visualization configuration

    Returns:
        Visualization data with propagation information
    """
    if config is None:
        config = VisualizationConfig()

    logger.info("Exporting propagation visualization")

    # Get infected nodes
    infected_nodes = set(propagation_results.get('final_infected_set', []))

    # Create copy of graph and add propagation information
    viz_graph = graph.copy()

    # Mark infected nodes
    for node in viz_graph.nodes():
        viz_graph.nodes[node]['infected'] = node in infected_nodes
        viz_graph.nodes[node]['propagation_step'] = _get_infection_step(
            node, propagation_results.get('propagation_history', [])
        )

    # Export with propagation data
    viz_data = export_for_visualization(viz_graph, config)

    # Add propagation metadata
    viz_data['propagation_metadata'] = {
        'total_infected': len(infected_nodes),
        'infection_rate': len(infected_nodes) / graph.number_of_nodes(),
        'model_used': propagation_results.get('model', 'unknown'),
        'steps_to_convergence': propagation_results.get('steps_to_convergence', 0)
    }

    return viz_data


def _get_infection_step(node: int, propagation_history: List[Dict]) -> int:
    """Get the step at which a node was infected."""
    for step_data in propagation_history:
        if node in step_data.get('newly_infected', []):
            return step_data.get('step', -1)
    return -1  # Not infected


def export_community_visualization(graph: nx.Graph,
                                 community_partition: Dict[int, int],
                                 config: Optional[VisualizationConfig] = None) -> Dict[str, List[Dict]]:
    """
    Export network with community structure for visualization.

    Args:
        graph: NetworkX graph
        community_partition: Dictionary mapping nodes to community IDs
        config: Visualization configuration

    Returns:
        Visualization data with community information
    """
    if config is None:
        config = VisualizationConfig()

    logger.info(f"Exporting community visualization with {len(set(community_partition.values()))} communities")

    # Create copy of graph and add community information
    viz_graph = graph.copy()

    for node in viz_graph.nodes():
        viz_graph.nodes[node]['community'] = community_partition.get(node, -1)

    # Override color attribute to use community
    config.node_color_attribute = 'community'

    return export_for_visualization(viz_graph, config)


def save_visualization_data(viz_data: Dict[str, Any], filepath: str) -> None:
    """
    Save visualization data to a JSON file.

    Args:
        viz_data: Visualization data dictionary
        filepath: Path to save the JSON file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)
        logger.info(f"Visualization data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save visualization data: {e}")
        raise


def load_visualization_data(filepath: str) -> Dict[str, Any]:
    """
    Load visualization data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded visualization data
    """
    try:
        with open(filepath, 'r') as f:
            viz_data = json.load(f)
        logger.info(f"Visualization data loaded from {filepath}")
        return viz_data
    except Exception as e:
        logger.error(f"Failed to load visualization data: {e}")
        raise