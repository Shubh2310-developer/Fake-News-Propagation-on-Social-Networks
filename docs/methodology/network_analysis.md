# Network Analysis Methodology

This document explains the network science principles used to model and analyze the social networks in our game theory simulations and information propagation studies.

## Overview

Our network analysis approach combines graph theory, complex systems modeling, and information diffusion algorithms to understand how misinformation spreads through social networks and how network structure affects the strategic interactions between information spreaders and fact-checkers.

## Graph Theory Foundations

### Network Representation

We model social networks as directed graphs G = (V, E) where:
- **Nodes (V)**: Individual users, organizations, or information sources
- **Edges (E)**: Relationships indicating information flow (follows, friends, citations)
- **Weights**: Strength of connections (interaction frequency, trust levels)

#### Node Types
1. **Regular Users**: Individual social media users
2. **Influencers**: High-follower accounts with broad reach
3. **Organizations**: News outlets, government agencies, institutions
4. **Bots**: Automated accounts for content amplification
5. **Fact-Checkers**: Dedicated verification accounts

#### Edge Types
1. **Social Connections**: Friend/follow relationships
2. **Information Sharing**: Retweets, shares, citations
3. **Engagement**: Likes, comments, reactions
4. **Trust Networks**: Explicit trust or credibility ratings

### Network Topologies

#### 1. ErdQs-R�nyi Random Networks
- **Construction**: Each edge exists with probability p
- **Properties**: Uniform degree distribution, low clustering
- **Use Case**: Baseline comparison for random information spread
- **Parameters**: N nodes, edge probability p = k/N (k = average degree)

#### 2. Scale-Free Networks (Barab�si-Albert)
- **Construction**: Preferential attachment during growth
- **Properties**: Power-law degree distribution, high-degree hubs
- **Use Case**: Modeling social media with influencers
- **Parameters**: N nodes, m edges per new node

#### 3. Small-World Networks (Watts-Strogatz)
- **Construction**: Ring lattice with random rewiring
- **Properties**: High clustering, short path lengths
- **Use Case**: Modeling real-world social networks
- **Parameters**: N nodes, k nearest neighbors, rewiring probability �

#### 4. Regular Networks
- **Construction**: Fixed degree for all nodes
- **Properties**: Uniform structure, predictable paths
- **Use Case**: Controlled experiments and theoretical analysis
- **Parameters**: N nodes, degree k

## Centrality Metrics

### Degree Centrality
```
C_D(v) = deg(v) / (n-1)
```
- **Interpretation**: Local influence based on direct connections
- **Application**: Identifying potential information spreaders
- **Variants**: In-degree (followers), out-degree (following)

### Betweenness Centrality
```
C_B(v) = Σ(σ_st(v) / σ_st) for all s≠v≠t
```
Where σ_st is shortest paths from s to t, σ_st(v) is paths through v
- **Interpretation**: Control over information flow between others
- **Application**: Identifying key bridges and information brokers
- **Computational Complexity**: O(n³) exact, O(n²) approximation

### Closeness Centrality
```
C_C(v) = (n-1) / Σd(v,u) for all u≠v
```
- **Interpretation**: Speed of information spreading from node
- **Application**: Identifying rapid information disseminators
- **Variants**: Harmonic closeness for disconnected graphs

### Eigenvector Centrality
```
λx = Ax (where A is adjacency matrix)
```
- **Interpretation**: Influence based on connections to influential nodes
- **Application**: PageRank-style importance in information networks
- **Computational**: Power iteration method for large networks

### Katz Centrality
```
C_K(v) = α Σ A^k_vu for k=1 to ∞
```
- **Interpretation**: Weighted sum of all path lengths
- **Application**: Balancing local and global influence
- **Parameter**: α controls relative importance of distant connections

## Information Diffusion Models

### Independent Cascade Model
Each activated node attempts to activate neighbors with probability p:

```python
def independent_cascade(graph, seed_nodes, activation_prob):
    activated = set(seed_nodes)
    newly_activated = set(seed_nodes)

    while newly_activated:
        next_round = set()
        for node in newly_activated:
            for neighbor in graph.neighbors(node):
                if neighbor not in activated:
                    if random.random() < activation_prob:
                        next_round.add(neighbor)

        newly_activated = next_round
        activated.update(newly_activated)

    return activated
```

### Linear Threshold Model
Nodes activate when weighted sum of activated neighbors exceeds threshold:

```python
def linear_threshold(graph, seed_nodes, thresholds, weights):
    activated = set(seed_nodes)

    while True:
        new_activations = set()
        for node in graph.nodes():
            if node not in activated:
                influence_sum = sum(weights[neighbor][node]
                                  for neighbor in graph.neighbors(node)
                                  if neighbor in activated)
                if influence_sum >= thresholds[node]:
                    new_activations.add(node)

        if not new_activations:
            break
        activated.update(new_activations)

    return activated
```

### SIR (Susceptible-Infected-Recovered) Model
Modified for information spread with fact-checking:

- **S (Susceptible)**: Unaware of information
- **I (Infected)**: Believing and spreading misinformation
- **R (Recovered)**: Fact-checked, immune to misinformation

Transition rates:
- S → I: β * (infected neighbors / total neighbors)
- I → R: γ (natural recovery) + δ * (fact-checker neighbors)

### Custom Misinformation Model
Incorporates credibility, verification, and strategic behavior:

```python
class MisinformationDiffusion:
    def __init__(self, graph, credibility_scores, verification_rates):
        self.graph = graph
        self.credibility = credibility_scores
        self.verification = verification_rates

    def activation_probability(self, source, target, content_type):
        base_prob = 0.1 if content_type == 'fake' else 0.3
        credibility_factor = self.credibility[source]
        verification_factor = 1 - self.verification[target]

        return base_prob * credibility_factor * verification_factor
```

## Network Structure Analysis

### Clustering and Communities

#### Local Clustering Coefficient
```
C_i = 2 * |edges between neighbors of i| / (k_i * (k_i - 1))
```

#### Global Clustering Coefficient
```
C = 3 * (number of triangles) / (number of connected triples)
```

#### Community Detection
- **Modularity Optimization**: Louvain algorithm for community structure
- **Spectral Clustering**: Eigenvalue decomposition for partitioning
- **Infomap**: Information-theoretic community detection

### Path Analysis

#### Average Path Length
```
L = (1 / (n*(n-1))) * Σ d(i,j) for all i≠j
```

#### Diameter
```
diam(G) = max d(i,j) for all i,j ∈ V
```

#### Small-World Coefficient
```
σ = (C/C_random) / (L/L_random)
```
Where σ > 1 indicates small-world properties

### Robustness and Resilience

#### Attack Strategies
1. **Random Removal**: Remove random fraction of nodes/edges
2. **Targeted Attack**: Remove highest-degree nodes first
3. **Betweenness Attack**: Remove highest-betweenness nodes
4. **Cascade Failure**: Model propagating failures

#### Resilience Metrics
- **Giant Component Size**: Fraction in largest connected component
- **Network Efficiency**: Average inverse shortest path length
- **Connectivity**: Minimum cuts and k-connectivity

## Dynamic Network Analysis

### Temporal Networks
Model time-evolving networks G(t) = (V(t), E(t)):

- **Node Dynamics**: Users joining/leaving platform
- **Edge Dynamics**: Friendship formation/dissolution
- **Weight Evolution**: Changing interaction frequencies

### Adaptive Networks
Networks that change based on information spreading:

```python
def update_network_structure(graph, misinformation_exposure):
    for node in graph.nodes():
        if misinformation_exposure[node] > threshold:
            # Unfollow sources of misinformation
            sources = get_misinformation_sources(node)
            for source in sources:
                if graph.has_edge(source, node):
                    graph.remove_edge(source, node)

            # Increase connections to fact-checkers
            fact_checkers = get_nearby_fact_checkers(node)
            for fc in fact_checkers[:max_new_connections]:
                graph.add_edge(fc, node)
```

### Coevolution Models
Simultaneous evolution of network structure and node states:

- **Homophily**: Similar users connect preferentially
- **Influence**: Connected users become more similar
- **Selection vs. Influence**: Disentangling mechanisms

## Strategic Network Analysis

### Influence Maximization
Find k nodes to maximize spread of information:

```python
def greedy_influence_maximization(graph, k, diffusion_model):
    selected = []

    for i in range(k):
        best_node = None
        best_gain = 0

        for candidate in graph.nodes():
            if candidate not in selected:
                test_set = selected + [candidate]
                spread = diffusion_model.simulate(test_set)
                gain = spread - current_spread

                if gain > best_gain:
                    best_gain = gain
                    best_node = candidate

        selected.append(best_node)
        current_spread += best_gain

    return selected
```

### Network Intervention Strategies

#### 1. Node-Based Interventions
- **Influencer Engagement**: Partner with high-centrality nodes
- **Bot Removal**: Identify and remove automated accounts
- **Fact-Checker Placement**: Strategically position verification accounts

#### 2. Edge-Based Interventions
- **Echo Chamber Breaking**: Introduce cross-cutting ties
- **Bridge Building**: Connect isolated communities
- **Weak Tie Strengthening**: Reinforce bridging connections

#### 3. Information-Based Interventions
- **Prebunking**: Proactive inoculation against misinformation
- **Counter-Narratives**: Strategic placement of corrective information
- **Algorithmic Adjustments**: Modify recommendation systems

## Empirical Network Analysis

### Data Collection

#### Social Media APIs
- **Twitter API**: Follower networks, retweet graphs
- **Facebook Graph API**: Friend networks, page interactions
- **YouTube API**: Subscription networks, comment graphs
- **Reddit API**: Subreddit networks, user interactions

#### Web Crawling
- **Hyperlink Networks**: Website citation patterns
- **News Source Networks**: Cross-referencing between outlets
- **Academic Networks**: Citation graphs from scholarly articles

### Network Statistics

#### Basic Properties
```python
def network_statistics(G):
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'clustering': nx.average_clustering(G),
        'path_length': nx.average_shortest_path_length(G),
        'diameter': nx.diameter(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }
    return stats
```

#### Degree Distribution Analysis
```python
def analyze_degree_distribution(G):
    degrees = [d for n, d in G.degree()]

    # Test for power law
    fit = powerlaw.Fit(degrees)

    return {
        'mean_degree': np.mean(degrees),
        'degree_variance': np.var(degrees),
        'power_law_alpha': fit.power_law.alpha,
        'power_law_xmin': fit.power_law.xmin,
        'power_law_likelihood': fit.power_law.loglikelihoods
    }
```

### Validation and Benchmarking

#### Null Models
- **Configuration Model**: Preserve degree sequence
- **Erdős-Rényi**: Random graphs with same density
- **Exponential Random Graphs**: Statistical models with constraints

#### Model Comparison
- **AIC/BIC**: Information criteria for model selection
- **Cross-Validation**: Predictive performance on held-out data
- **Goodness-of-Fit**: Statistical tests for model adequacy

This network analysis methodology provides the foundation for understanding information propagation patterns and designing effective interventions in complex social networks.