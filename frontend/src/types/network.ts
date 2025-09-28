// frontend/src/types/network.ts

/**
 * Type definitions for social network graph data and visualization.
 * These interfaces are essential for frontend visualization components
 * and ensure proper data structure for network analysis and rendering.
 */

import type { PlayerType } from './gameTheory';

/**
 * Network topology types
 */
export type NetworkType = 'random' | 'small_world' | 'scale_free' | 'complete' | 'ring' | 'star' | 'tree' | 'custom';

/**
 * Node (user) data in the network graph
 */
export interface NodeData {
  /** Unique identifier for the node */
  id: string;
  /** Display label for the node */
  label?: string;
  /** Node type/category */
  type: PlayerType | 'regular_user';
  /** User type for visualization */
  user_type: string;

  /** Influence and credibility metrics */
  influence_score: number;
  credibility_score: number;
  trust_score: number;
  authority_score: number;

  /** Network centrality measures */
  centrality: {
    degree: number;
    betweenness: number;
    closeness: number;
    eigenvector: number;
    pagerank: number;
  };

  /** Content and activity metrics */
  activity: {
    posts_count: number;
    shares_count: number;
    comments_count: number;
    reactions_count: number;
    last_active: string;
  };

  /** Content classification */
  content_stats: {
    fake_news_shared: number;
    real_news_shared: number;
    fact_checks_performed: number;
    misinformation_flagged: number;
  };

  /** Demographic and profile information */
  profile: {
    account_age: number;
    follower_count: number;
    following_count: number;
    verified: boolean;
    location?: string;
    categories: string[];
  };

  /** Visual positioning (for layout algorithms) */
  position?: {
    x: number;
    y: number;
    z?: number;
  };

  /** Visual styling properties */
  visual: {
    size: number;
    color: string;
    opacity: number;
    shape: 'circle' | 'square' | 'triangle' | 'diamond';
    strokeWidth: number;
    strokeColor: string;
  };

  /** State information */
  state: {
    selected: boolean;
    highlighted: boolean;
    filtered: boolean;
    clusterId?: string;
    communityId?: string;
  };

  /** Custom metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Edge (connection) data between nodes
 */
export interface LinkData {
  /** Unique identifier for the edge */
  id?: string;
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;

  /** Relationship strength and trust */
  weight: number;
  trust: number;
  interaction_strength: number;
  influence_flow: number;

  /** Relationship type and characteristics */
  relationship: {
    type: 'follow' | 'friend' | 'mention' | 'retweet' | 'reply' | 'share';
    bidirectional: boolean;
    established_date: string;
    last_interaction: string;
  };

  /** Communication metrics */
  communication: {
    message_count: number;
    shared_content: number;
    reaction_count: number;
    average_response_time: number;
  };

  /** Content flow analysis */
  content_flow: {
    information_shared: number;
    fake_news_transmitted: number;
    fact_checks_shared: number;
    corrections_made: number;
  };

  /** Visual styling properties */
  visual: {
    thickness: number;
    color: string;
    opacity: number;
    style: 'solid' | 'dashed' | 'dotted';
    animated: boolean;
  };

  /** State information */
  state: {
    selected: boolean;
    highlighted: boolean;
    filtered: boolean;
  };

  /** Custom metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Complete network data structure
 */
export interface NetworkData {
  /** Network metadata */
  id: string;
  name?: string;
  description?: string;
  created_at: string;
  updated_at: string;

  /** Node and edge collections */
  nodes: NodeData[];
  links: LinkData[];

  /** Network statistics */
  statistics: NetworkStatistics;

  /** Layout information */
  layout: {
    algorithm: string;
    parameters: Record<string, unknown>;
    bounds: {
      minX: number;
      maxX: number;
      minY: number;
      maxY: number;
    };
  };

  /** Filtering and grouping */
  filters: {
    active: Record<string, unknown>;
    available: Array<{
      name: string;
      type: 'categorical' | 'numerical' | 'boolean';
      values: unknown[];
    }>;
  };

  /** Community detection results */
  communities?: CommunityData[];

  /** Temporal information */
  temporal?: {
    snapshots: Array<{
      timestamp: string;
      nodes: string[];
      links: Array<{ source: string; target: string }>;
    }>;
  };

  /** Custom metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Network statistics and metrics
 */
export interface NetworkStatistics {
  /** Basic network properties */
  basic: {
    node_count: number;
    edge_count: number;
    density: number;
    is_connected: boolean;
    is_directed: boolean;
    is_weighted: boolean;
  };

  /** Structural properties */
  structure: {
    average_degree: number;
    max_degree: number;
    degree_distribution: Array<{ degree: number; count: number }>;
    clustering_coefficient: number;
    transitivity: number;
    assortativity: number;
  };

  /** Path and distance metrics */
  paths: {
    average_path_length: number;
    diameter: number;
    radius: number;
    efficiency: number;
    small_world_coefficient?: number;
  };

  /** Centralization measures */
  centralization: {
    degree: number;
    betweenness: number;
    closeness: number;
    eigenvector: number;
  };

  /** Community structure */
  community: {
    modularity: number;
    community_count: number;
    average_community_size: number;
    community_sizes: number[];
  };

  /** Robustness and resilience */
  robustness: {
    node_connectivity: number;
    edge_connectivity: number;
    algebraic_connectivity: number;
    vulnerability: number;
  };

  /** Information flow properties */
  information_flow: {
    bottlenecks: string[];
    bridges: Array<{ source: string; target: string }>;
    cut_vertices: string[];
    flow_capacity: number;
  };
}

/**
 * Community detection results
 */
export interface CommunityData {
  /** Community identifier */
  id: string;
  /** Community label/name */
  name?: string;
  /** Nodes in this community */
  nodes: string[];
  /** Community size */
  size: number;
  /** Community quality metrics */
  metrics: {
    modularity: number;
    conductance: number;
    density: number;
    clustering: number;
  };
  /** Community characteristics */
  characteristics: {
    dominant_user_types: Record<string, number>;
    activity_level: 'low' | 'medium' | 'high';
    content_focus: string[];
    influence_level: number;
  };
  /** Visual properties */
  visual: {
    color: string;
    center: { x: number; y: number };
    bounds: { x1: number; y1: number; x2: number; y2: number };
  };
}

/**
 * Network layout algorithm configuration
 */
export interface LayoutConfig {
  /** Algorithm type */
  algorithm: 'force_directed' | 'circular' | 'hierarchical' | 'grid' | 'spring' | 'fruchterman_reingold' | 'kamada_kawai';
  /** Algorithm parameters */
  parameters: {
    iterations?: number;
    strength?: number;
    distance?: number;
    theta?: number;
    cooling_factor?: number;
    initial_temperature?: number;
    gravity?: number;
    node_repulsion?: number;
    edge_attraction?: number;
  };
  /** Constraints */
  constraints?: {
    fixed_nodes?: string[];
    boundaries?: {
      x: [number, number];
      y: [number, number];
    };
    preserve_aspects?: boolean;
  };
}

/**
 * Network visualization configuration
 */
export interface NetworkVisualizationConfig {
  /** Canvas dimensions */
  dimensions: {
    width: number;
    height: number;
    depth?: number; // for 3D visualizations
  };

  /** Zoom and pan settings */
  camera: {
    zoom: number;
    center: { x: number; y: number };
    min_zoom: number;
    max_zoom: number;
  };

  /** Visual settings */
  visual: {
    /** Node rendering */
    nodes: {
      show_labels: boolean;
      label_threshold: number; // minimum zoom to show labels
      size_scaling: 'linear' | 'log' | 'sqrt';
      color_scheme: string;
      opacity: number;
    };

    /** Edge rendering */
    edges: {
      show_edges: boolean;
      thickness_scaling: 'linear' | 'log' | 'sqrt';
      opacity: number;
      color_scheme: string;
      curve_type: 'straight' | 'curved' | 'arc';
    };

    /** Community visualization */
    communities: {
      show_hulls: boolean;
      hull_opacity: number;
      show_labels: boolean;
    };
  };

  /** Interaction settings */
  interaction: {
    hover_enabled: boolean;
    selection_enabled: boolean;
    zoom_enabled: boolean;
    pan_enabled: boolean;
    tooltip_enabled: boolean;
  };

  /** Performance settings */
  performance: {
    max_nodes_for_labels: number;
    max_edges_for_curves: number;
    level_of_detail: boolean;
    fps_target: number;
  };
}

/**
 * Network analysis filter
 */
export interface NetworkFilter {
  /** Filter identifier */
  id: string;
  /** Filter name */
  name: string;
  /** Filter type */
  type: 'node' | 'edge' | 'community';
  /** Filter criteria */
  criteria: {
    field: string;
    operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'contains' | 'in' | 'not_in';
    value: unknown;
  };
  /** Whether filter is active */
  active: boolean;
  /** Whether filter excludes or includes matching items */
  mode: 'include' | 'exclude';
}

/**
 * Network search configuration
 */
export interface NetworkSearch {
  /** Search query */
  query: string;
  /** Search scope */
  scope: 'nodes' | 'edges' | 'both';
  /** Fields to search in */
  fields: string[];
  /** Search options */
  options: {
    case_sensitive: boolean;
    exact_match: boolean;
    regex: boolean;
  };
  /** Search results */
  results: {
    nodes: string[];
    edges: string[];
    total_matches: number;
  };
}

/**
 * Network comparison data
 */
export interface NetworkComparison {
  /** Networks being compared */
  networks: Array<{
    id: string;
    name: string;
    statistics: NetworkStatistics;
  }>;

  /** Comparison metrics */
  comparison: {
    /** Structural differences */
    structural: Record<string, {
      values: number[];
      differences: number[];
      significance: boolean;
    }>;

    /** Node-level comparisons */
    nodes: {
      common_nodes: string[];
      unique_nodes: Record<string, string[]>;
      changed_properties: Array<{
        node_id: string;
        property: string;
        values: unknown[];
      }>;
    };

    /** Edge-level comparisons */
    edges: {
      common_edges: Array<{ source: string; target: string }>;
      unique_edges: Record<string, Array<{ source: string; target: string }>>;
      changed_weights: Array<{
        source: string;
        target: string;
        weights: number[];
      }>;
    };
  };

  /** Evolution metrics */
  evolution?: {
    growth_rate: number;
    stability_index: number;
    change_points: Array<{
      timestamp: string;
      change_type: string;
      magnitude: number;
    }>;
  };
}

/**
 * Network export/import configuration
 */
export interface NetworkExportConfig {
  /** Export format */
  format: 'json' | 'graphml' | 'gexf' | 'csv' | 'pajek' | 'dot';
  /** What to include */
  include: {
    nodes: boolean;
    edges: boolean;
    attributes: boolean;
    layout: boolean;
    communities: boolean;
    statistics: boolean;
  };
  /** Export options */
  options: {
    compress: boolean;
    pretty_print: boolean;
    include_metadata: boolean;
  };
}

/**
 * Real-time network updates
 */
export interface NetworkUpdate {
  /** Update type */
  type: 'node_added' | 'node_removed' | 'node_updated' | 'edge_added' | 'edge_removed' | 'edge_updated';
  /** Timestamp */
  timestamp: string;
  /** Update data */
  data: {
    node?: Partial<NodeData>;
    edge?: Partial<LinkData>;
    changes?: Record<string, { old: unknown; new: unknown }>;
  };
  /** Update metadata */
  metadata?: {
    source: string;
    batch_id?: string;
    sequence: number;
  };
}

/**
 * Network simulation parameters
 */
export interface NetworkSimulationParams {
  /** Simulation type */
  type: 'information_spread' | 'opinion_dynamics' | 'epidemic' | 'cascade' | 'diffusion';
  /** Initial conditions */
  initial_conditions: {
    seed_nodes: string[];
    seed_values: Record<string, unknown>;
    transmission_probability: number;
    recovery_rate?: number;
  };
  /** Simulation parameters */
  parameters: {
    max_steps: number;
    step_size: number;
    threshold_dynamics: boolean;
    network_effects: boolean;
    external_influence: number;
  };
  /** Output configuration */
  output: {
    save_states: boolean;
    save_interval: number;
    track_metrics: string[];
  };
}