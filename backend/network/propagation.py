# backend/network/propagation.py

import networkx as nx
import numpy as np
from typing import Set, Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PropagationModel(Enum):
    """Available propagation models."""
    INDEPENDENT_CASCADE = 'independent_cascade'
    LINEAR_THRESHOLD = 'linear_threshold'
    SUSCEPTIBLE_INFECTED_RECOVERED = 'sir'
    COMPLEX_CONTAGION = 'complex_contagion'


@dataclass
class ContentProperties:
    """Properties of the content being propagated."""
    content_type: str = 'neutral'  # 'fake_emotional', 'fake_neutral', 'true'
    content_quality: float = 0.5   # 0 = low quality, 1 = high quality
    virality_factor: float = 1.0   # Multiplier for spread probability
    emotional_appeal: float = 0.5  # How emotionally appealing the content is
    complexity: float = 0.5        # How complex the content is to understand
    source_credibility: float = 0.5 # Credibility of the original source


@dataclass
class PropagationConfig:
    """Configuration for propagation simulation."""
    base_transmission_rate: float = 0.1
    time_steps: int = 50
    recovery_rate: float = 0.05  # For SIR model
    threshold_factor: float = 0.3  # For linear threshold model
    complex_contagion_threshold: int = 2  # Number of infected neighbors needed
    random_seed: Optional[int] = None


class InformationPropagationSimulator:
    """Simulates the spread of information across a network graph."""

    def __init__(self, network: nx.Graph, config: Optional[PropagationConfig] = None):
        """
        Initialize the propagation simulator.

        Args:
            network: NetworkX graph representing the social network
            config: Configuration parameters for propagation
        """
        if not isinstance(network, nx.Graph):
            raise TypeError("Input must be a NetworkX graph")

        self.network = network
        self.config = config or PropagationConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info(f"Initialized propagation simulator with {network.number_of_nodes()} nodes")

    def simulate_propagation(self,
                           initial_spreaders: Set[int],
                           content_properties: ContentProperties,
                           model: PropagationModel = PropagationModel.INDEPENDENT_CASCADE) -> Dict[str, Any]:
        """
        Main method to simulate information propagation.

        Args:
            initial_spreaders: Set of node IDs that initially have the information
            content_properties: Properties of the content being spread
            model: Propagation model to use

        Returns:
            Dictionary containing propagation results
        """
        logger.info(f"Simulating {model.value} propagation from {len(initial_spreaders)} initial spreaders")

        if model == PropagationModel.INDEPENDENT_CASCADE:
            return self._independent_cascade_model(initial_spreaders, content_properties)
        elif model == PropagationModel.LINEAR_THRESHOLD:
            return self._linear_threshold_model(initial_spreaders, content_properties)
        elif model == PropagationModel.SUSCEPTIBLE_INFECTED_RECOVERED:
            return self._sir_model(initial_spreaders, content_properties)
        elif model == PropagationModel.COMPLEX_CONTAGION:
            return self._complex_contagion_model(initial_spreaders, content_properties)
        else:
            raise ValueError(f"Unknown propagation model: {model}")

    def _calculate_infection_probability(self,
                                       spreader: int,
                                       target: int,
                                       content_properties: ContentProperties) -> float:
        """
        Calculates transmission probability based on multiple factors.

        Args:
            spreader: Node ID of the spreader
            target: Node ID of the target
            content_properties: Properties of the content

        Returns:
            Probability of transmission (0-1)
        """
        # Base transmission rate
        base_rate = self.config.base_transmission_rate

        # Spreader influence
        spreader_influence = self.network.nodes[spreader].get('influence_score', 0.5)

        # Target susceptibility (inverse of credibility)
        target_credibility = self.network.nodes[target].get('credibility_score', 0.5)
        target_susceptibility = 1 - target_credibility

        # Edge trust and interaction strength
        if self.network.has_edge(spreader, target):
            edge_trust = self.network.edges[spreader, target].get('trust', 0.5)
            interaction_strength = self.network.edges[spreader, target].get('interaction_strength', 0.5)
        else:
            edge_trust = 0.1  # Low trust for non-connected users
            interaction_strength = 0.1

        # Content-based factors
        content_multiplier = self._calculate_content_multiplier(content_properties, target)

        # Combine all factors
        probability = (base_rate *
                      spreader_influence *
                      target_susceptibility *
                      edge_trust *
                      interaction_strength *
                      content_multiplier)

        return min(probability, 1.0)

    def _calculate_content_multiplier(self,
                                    content_properties: ContentProperties,
                                    target: int) -> float:
        """
        Calculate content-specific transmission multiplier.

        Args:
            content_properties: Properties of the content
            target: Target node ID

        Returns:
            Multiplier for transmission probability
        """
        multiplier = 1.0

        # Fake news spreads faster (as per research literature)
        if content_properties.content_type == 'fake_emotional':
            multiplier *= 1.5  # Emotional fake news spreads fastest
        elif content_properties.content_type == 'fake_neutral':
            multiplier *= 1.2  # Neutral fake news spreads moderately fast
        elif content_properties.content_type == 'true':
            multiplier *= 0.8  # True news spreads slower

        # Quality affects spread (lower quality spreads faster paradoxically)
        quality_factor = 1.0 + (1 - content_properties.content_quality) * 0.5
        multiplier *= quality_factor

        # Virality factor
        multiplier *= content_properties.virality_factor

        # Emotional appeal increases spread
        multiplier *= (1.0 + content_properties.emotional_appeal * 0.3)

        # Complex content spreads slower
        multiplier *= (1.0 - content_properties.complexity * 0.2)

        # Source credibility affects spread
        multiplier *= (0.5 + content_properties.source_credibility * 0.5)

        # Target-specific factors
        target_type = self.network.nodes[target].get('user_type', 'regular')
        if target_type == 'bot':
            multiplier *= 1.3  # Bots spread information more readily
        elif target_type == 'verified':
            multiplier *= 0.7  # Verified users are more cautious

        return multiplier

    def _independent_cascade_model(self,
                                 initial_spreaders: Set[int],
                                 content_properties: ContentProperties) -> Dict[str, Any]:
        """
        Implements the Independent Cascade (IC) model of propagation.

        In this model, each newly infected node gets one chance to infect
        each of its neighbors with some probability.

        Args:
            initial_spreaders: Initial set of infected nodes
            content_properties: Properties of the content

        Returns:
            Dictionary containing propagation results
        """
        infected_nodes = set(initial_spreaders)
        newly_infected = set(initial_spreaders)
        propagation_history = []
        activation_probabilities = {}

        for t in range(self.config.time_steps):
            if not newly_infected:
                break

            activated_this_step = set()
            step_activations = {}

            for spreader in newly_infected:
                for neighbor in self.network.neighbors(spreader):
                    if neighbor not in infected_nodes:
                        prob = self._calculate_infection_probability(
                            spreader, neighbor, content_properties
                        )

                        # Store probability for analysis
                        if neighbor not in activation_probabilities:
                            activation_probabilities[neighbor] = []
                        activation_probabilities[neighbor].append(prob)

                        if np.random.random() < prob:
                            activated_this_step.add(neighbor)
                            step_activations[neighbor] = {
                                'spreader': spreader,
                                'probability': prob,
                                'step': t
                            }

            # Record this step
            propagation_history.append({
                'step': t,
                'newly_infected': list(activated_this_step),
                'total_infected': len(infected_nodes) + len(activated_this_step),
                'activations': step_activations
            })

            newly_infected = activated_this_step
            infected_nodes.update(newly_infected)

        return {
            'model': 'independent_cascade',
            'total_reach': len(infected_nodes),
            'final_infected_set': list(infected_nodes),
            'propagation_history': propagation_history,
            'activation_probabilities': activation_probabilities,
            'content_properties': content_properties.__dict__,
            'steps_to_convergence': len(propagation_history)
        }

    def _linear_threshold_model(self,
                              initial_spreaders: Set[int],
                              content_properties: ContentProperties) -> Dict[str, Any]:
        """
        Implements the Linear Threshold (LT) model of propagation.

        In this model, each node has a threshold and gets infected when
        the sum of influence from infected neighbors exceeds this threshold.

        Args:
            initial_spreaders: Initial set of infected nodes
            content_properties: Properties of the content

        Returns:
            Dictionary containing propagation results
        """
        infected_nodes = set(initial_spreaders)
        propagation_history = []

        # Assign random thresholds to each node
        thresholds = {}
        for node in self.network.nodes():
            # Threshold based on user credibility (higher credibility = higher threshold)
            base_threshold = self.config.threshold_factor
            credibility_adjustment = self.network.nodes[node].get('credibility_score', 0.5) * 0.3
            thresholds[node] = base_threshold + credibility_adjustment

        for t in range(self.config.time_steps):
            newly_infected = set()
            step_influences = {}

            for node in self.network.nodes():
                if node not in infected_nodes:
                    # Calculate total influence from infected neighbors
                    total_influence = 0.0
                    influencing_neighbors = []

                    for neighbor in self.network.neighbors(node):
                        if neighbor in infected_nodes:
                            # Influence based on edge weight and content properties
                            edge_influence = self._calculate_threshold_influence(
                                neighbor, node, content_properties
                            )
                            total_influence += edge_influence
                            influencing_neighbors.append((neighbor, edge_influence))

                    step_influences[node] = {
                        'total_influence': total_influence,
                        'threshold': thresholds[node],
                        'influencing_neighbors': influencing_neighbors
                    }

                    # Check if threshold is exceeded
                    if total_influence > thresholds[node]:
                        newly_infected.add(node)

            if not newly_infected:
                break

            propagation_history.append({
                'step': t,
                'newly_infected': list(newly_infected),
                'total_infected': len(infected_nodes) + len(newly_infected),
                'influences': step_influences
            })

            infected_nodes.update(newly_infected)

        return {
            'model': 'linear_threshold',
            'total_reach': len(infected_nodes),
            'final_infected_set': list(infected_nodes),
            'propagation_history': propagation_history,
            'thresholds': thresholds,
            'content_properties': content_properties.__dict__,
            'steps_to_convergence': len(propagation_history)
        }

    def _calculate_threshold_influence(self,
                                     spreader: int,
                                     target: int,
                                     content_properties: ContentProperties) -> float:
        """
        Calculate influence for linear threshold model.

        Args:
            spreader: Influencing node
            target: Target node
            content_properties: Content properties

        Returns:
            Influence value
        """
        spreader_influence = self.network.nodes[spreader].get('influence_score', 0.5)
        edge_trust = self.network.edges.get((spreader, target), {}).get('trust', 0.5)
        content_multiplier = self._calculate_content_multiplier(content_properties, target)

        return spreader_influence * edge_trust * content_multiplier * 0.1

    def _sir_model(self,
                  initial_spreaders: Set[int],
                  content_properties: ContentProperties) -> Dict[str, Any]:
        """
        Implements the Susceptible-Infected-Recovered (SIR) model.

        In this model, nodes can recover from infection and become immune.

        Args:
            initial_spreaders: Initial set of infected nodes
            content_properties: Properties of the content

        Returns:
            Dictionary containing propagation results
        """
        susceptible = set(self.network.nodes()) - initial_spreaders
        infected = set(initial_spreaders)
        recovered = set()
        propagation_history = []

        for t in range(self.config.time_steps):
            newly_infected = set()
            newly_recovered = set()

            # Infection process
            for infected_node in list(infected):
                for neighbor in self.network.neighbors(infected_node):
                    if neighbor in susceptible:
                        prob = self._calculate_infection_probability(
                            infected_node, neighbor, content_properties
                        )
                        if np.random.random() < prob:
                            newly_infected.add(neighbor)

            # Recovery process
            for infected_node in list(infected):
                if np.random.random() < self.config.recovery_rate:
                    newly_recovered.add(infected_node)

            # Update states
            susceptible -= newly_infected
            infected = (infected | newly_infected) - newly_recovered
            recovered |= newly_recovered

            propagation_history.append({
                'step': t,
                'susceptible': len(susceptible),
                'infected': len(infected),
                'recovered': len(recovered),
                'newly_infected': list(newly_infected),
                'newly_recovered': list(newly_recovered)
            })

            if not infected:
                break

        total_affected = len(recovered) + len(infected)

        return {
            'model': 'sir',
            'total_reach': total_affected,
            'final_susceptible': list(susceptible),
            'final_infected': list(infected),
            'final_recovered': list(recovered),
            'propagation_history': propagation_history,
            'content_properties': content_properties.__dict__,
            'steps_to_convergence': len(propagation_history)
        }

    def _complex_contagion_model(self,
                               initial_spreaders: Set[int],
                               content_properties: ContentProperties) -> Dict[str, Any]:
        """
        Implements a complex contagion model where multiple infected neighbors
        are needed for transmission.

        Args:
            initial_spreaders: Initial set of infected nodes
            content_properties: Properties of the content

        Returns:
            Dictionary containing propagation results
        """
        infected_nodes = set(initial_spreaders)
        propagation_history = []

        for t in range(self.config.time_steps):
            newly_infected = set()
            step_exposures = {}

            for node in self.network.nodes():
                if node not in infected_nodes:
                    # Count infected neighbors
                    infected_neighbors = [
                        n for n in self.network.neighbors(node)
                        if n in infected_nodes
                    ]

                    step_exposures[node] = {
                        'infected_neighbors': len(infected_neighbors),
                        'threshold': self.config.complex_contagion_threshold,
                        'neighbors': infected_neighbors
                    }

                    # Check if threshold is met
                    if len(infected_neighbors) >= self.config.complex_contagion_threshold:
                        # Calculate combined probability from all infected neighbors
                        combined_prob = 0.0
                        for neighbor in infected_neighbors:
                            prob = self._calculate_infection_probability(
                                neighbor, node, content_properties
                            )
                            combined_prob += prob

                        # Normalize and apply
                        combined_prob = min(combined_prob / len(infected_neighbors), 1.0)

                        if np.random.random() < combined_prob:
                            newly_infected.add(node)

            if not newly_infected:
                break

            propagation_history.append({
                'step': t,
                'newly_infected': list(newly_infected),
                'total_infected': len(infected_nodes) + len(newly_infected),
                'exposures': step_exposures
            })

            infected_nodes.update(newly_infected)

        return {
            'model': 'complex_contagion',
            'total_reach': len(infected_nodes),
            'final_infected_set': list(infected_nodes),
            'propagation_history': propagation_history,
            'content_properties': content_properties.__dict__,
            'steps_to_convergence': len(propagation_history)
        }

    def analyze_propagation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze propagation results to extract key metrics.

        Args:
            results: Results from a propagation simulation

        Returns:
            Dictionary containing analysis metrics
        """
        analysis = {
            'final_reach_percentage': results['total_reach'] / self.network.number_of_nodes(),
            'model_used': results['model'],
            'convergence_time': results.get('steps_to_convergence', 0),
            'propagation_rate': 0.0,
            'peak_infection_rate': 0.0
        }

        # Calculate propagation rate
        if results['steps_to_convergence'] > 0:
            analysis['propagation_rate'] = results['total_reach'] / results['steps_to_convergence']

        # Find peak infection rate
        history = results.get('propagation_history', [])
        if history:
            infection_rates = []
            for step in history:
                if 'newly_infected' in step:
                    infection_rates.append(len(step['newly_infected']))
                elif 'infected' in step:
                    infection_rates.append(step['infected'])

            if infection_rates:
                analysis['peak_infection_rate'] = max(infection_rates)

        # Network coverage analysis
        if results['total_reach'] > 0:
            infected_nodes = results.get('final_infected_set', [])
            if infected_nodes:
                # Analyze reach by user type
                user_type_reach = {}
                for node in infected_nodes:
                    user_type = self.network.nodes[node].get('user_type', 'unknown')
                    user_type_reach[user_type] = user_type_reach.get(user_type, 0) + 1

                analysis['reach_by_user_type'] = user_type_reach

                # Analyze reach by influence level
                high_influence_infected = sum(
                    1 for node in infected_nodes
                    if self.network.nodes[node].get('influence_score', 0) > 0.7
                )
                analysis['high_influence_reach'] = high_influence_infected

        return analysis

    def compare_propagation_models(self,
                                 initial_spreaders: Set[int],
                                 content_properties: ContentProperties) -> Dict[str, Dict[str, Any]]:
        """
        Compare results across different propagation models.

        Args:
            initial_spreaders: Initial spreaders for all models
            content_properties: Content properties for all models

        Returns:
            Dictionary mapping model names to their results and analysis
        """
        logger.info("Comparing propagation across different models")

        comparison_results = {}

        for model in PropagationModel:
            try:
                results = self.simulate_propagation(initial_spreaders, content_properties, model)
                analysis = self.analyze_propagation_results(results)

                comparison_results[model.value] = {
                    'results': results,
                    'analysis': analysis
                }

            except Exception as e:
                logger.error(f"Error simulating {model.value}: {e}")
                comparison_results[model.value] = {'error': str(e)}

        return comparison_results