/**
 * Visualization Configuration
 * Configuration for charts, graphs, and visualizations
 */

export const VISUALIZATION_CONFIG = {
  // Chart dimensions
  dimensions: {
    small: { width: 400, height: 300 },
    medium: { width: 600, height: 400 },
    large: { width: 800, height: 600 },
    fullWidth: { width: '100%', height: 500 },
  },

  // Color schemes
  colors: {
    // Primary palette
    primary: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],

    // Player types
    players: {
      spreader: '#ef4444',
      factChecker: '#10b981',
      platform: '#3b82f6',
    },

    // Article labels
    labels: {
      fake: '#ef4444',
      true: '#10b981',
      unknown: '#6b7280',
    },

    // Network states
    network: {
      infected: '#ef4444',
      susceptible: '#6b7280',
      immune: '#10b981',
      active: '#3b82f6',
    },

    // Heatmap gradients
    heatmap: {
      diverging: ['#ef4444', '#fef3c7', '#10b981'],
      sequential: ['#eff6ff', '#3b82f6', '#1e3a8a'],
      cool: ['#f0fdfa', '#14b8a6', '#134e4a'],
      warm: ['#fef3c7', '#f59e0b', '#78350f'],
    },

    // Status colors
    status: {
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6',
      neutral: '#6b7280',
    },
  },

  // Chart types configuration
  charts: {
    // Line chart
    line: {
      strokeWidth: 2,
      pointRadius: 4,
      smooth: true,
      showGrid: true,
      showLegend: true,
    },

    // Bar chart
    bar: {
      barThickness: 30,
      borderWidth: 1,
      showGrid: true,
      showLegend: true,
    },

    // Scatter plot
    scatter: {
      pointRadius: 6,
      pointHoverRadius: 8,
      showGrid: true,
      showLegend: true,
    },

    // Heatmap
    heatmap: {
      cellSize: 40,
      showValues: true,
      colorScale: 'diverging',
      minOpacity: 0.3,
      maxOpacity: 1.0,
    },

    // Network graph
    network: {
      nodeRadius: { min: 5, max: 20 },
      edgeWidth: { min: 1, max: 5 },
      labelSize: 12,
      physics: {
        enabled: true,
        iterations: 100,
      },
      layout: {
        type: 'force-directed', // or 'circular', 'hierarchical'
        spacing: 100,
      },
    },

    // Payoff matrix
    payoffMatrix: {
      cellWidth: 100,
      cellHeight: 60,
      fontSize: 14,
      showBestResponse: true,
      highlightEquilibrium: true,
    },
  },

  // Animation settings
  animation: {
    enabled: true,
    duration: 750,
    easing: 'easeInOutCubic',
    delay: 0,
  },

  // Tooltip configuration
  tooltip: {
    enabled: true,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    textColor: '#ffffff',
    borderRadius: 4,
    padding: 8,
    fontSize: 12,
  },

  // Legend configuration
  legend: {
    position: 'bottom' as const,
    align: 'center' as const,
    fontSize: 12,
    padding: 10,
  },

  // Grid configuration
  grid: {
    show: true,
    color: '#e5e7eb',
    strokeDasharray: '3,3',
  },

  // Axis configuration
  axis: {
    fontSize: 12,
    tickSize: 6,
    tickPadding: 8,
    showLabel: true,
  },

  // Export settings
  export: {
    formats: ['png', 'svg', 'pdf'],
    quality: 1.0,
    scale: 2,
  },
} as const;

// Network visualization layouts
export const NETWORK_LAYOUTS = {
  forceDirected: {
    name: 'Force Directed',
    description: 'Physics-based layout with attractive and repulsive forces',
    iterations: 100,
  },
  circular: {
    name: 'Circular',
    description: 'Nodes arranged in a circle',
    radius: 200,
  },
  hierarchical: {
    name: 'Hierarchical',
    description: 'Tree-like structure with levels',
    direction: 'vertical',
  },
  grid: {
    name: 'Grid',
    description: 'Regular grid arrangement',
    spacing: 50,
  },
} as const;

// Chart library presets (for recharts, d3, etc.)
export const CHART_PRESETS = {
  recharts: {
    margin: { top: 20, right: 30, left: 20, bottom: 20 },
    animationDuration: 750,
  },
  d3: {
    margin: { top: 20, right: 30, left: 40, bottom: 40 },
    transitionDuration: 750,
  },
} as const;

// Responsive breakpoints
export const RESPONSIVE_BREAKPOINTS = {
  mobile: 640,
  tablet: 768,
  desktop: 1024,
  wide: 1280,
} as const;

// Helper function to get color by category
export function getColor(category: string, key: string): string {
  const colors = VISUALIZATION_CONFIG.colors as any;
  return colors[category]?.[key] || '#6b7280';
}

// Helper function to get dimension by size
export function getDimensions(size: keyof typeof VISUALIZATION_CONFIG.dimensions) {
  return VISUALIZATION_CONFIG.dimensions[size];
}

// Helper function to get network layout
export function getNetworkLayout(layout: keyof typeof NETWORK_LAYOUTS) {
  return NETWORK_LAYOUTS[layout];
}
