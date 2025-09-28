// frontend/src/app/(dashboard)/simulation/components/NetworkGraph.tsx

"use client";

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { ZoomIn, ZoomOut, RotateCcw, Download, Network, Info } from 'lucide-react';

// Types for network data
interface NodeData extends d3.SimulationNodeDatum {
  id: string;
  type: 'user' | 'spreader' | 'moderator' | 'bot';
  status: 'susceptible' | 'infected' | 'recovered' | 'immune';
  influence: number;
  connections: number;
  label?: string;
  group?: number;
}

interface LinkData extends d3.SimulationLinkDatum<NodeData> {
  source: string | NodeData;
  target: string | NodeData;
  strength: number;
  type?: 'trust' | 'follow' | 'share';
  weight?: number;
}

interface NetworkData {
  nodes: NodeData[];
  links: LinkData[];
}

interface NetworkGraphProps {
  data: NetworkData;
  width?: number;
  height?: number;
  showLabels?: boolean;
  highlightPaths?: boolean;
  onNodeClick?: (node: NodeData) => void;
  onNodeHover?: (node: NodeData | null) => void;
  className?: string;
}

// Color schemes for different node types and statuses
const NODE_COLORS = {
  type: {
    user: '#94a3b8',      // slate-400
    spreader: '#ef4444',   // red-500
    moderator: '#3b82f6',  // blue-500
    bot: '#8b5cf6'         // violet-500
  },
  status: {
    susceptible: '#e2e8f0', // slate-200
    infected: '#fca5a5',    // red-300
    recovered: '#86efac',   // green-300
    immune: '#93c5fd'       // blue-300
  }
};

const LINK_COLORS = {
  trust: '#10b981',    // emerald-500
  follow: '#6b7280',   // gray-500
  share: '#f59e0b'     // amber-500
};

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  data,
  width = 800,
  height = 600,
  showLabels = true,
  highlightPaths = false,
  onNodeClick,
  onNodeHover,
  className = ""
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<NodeData, LinkData> | null>(null);

  const [hoveredNode, setHoveredNode] = useState<NodeData | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [forceStrength, setForceStrength] = useState([30]);
  const [linkDistance, setLinkDistance] = useState([50]);
  const [showNodeInfo, setShowNodeInfo] = useState(true);

  // Initialize and update D3 simulation
  const initializeSimulation = useCallback(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    const container = svg.select('.network-container');

    // Clear existing elements
    container.selectAll('*').remove();

    // Create groups for different layers
    const linkGroup = container.append('g').attr('class', 'links');
    const nodeGroup = container.append('g').attr('class', 'nodes');
    const labelGroup = container.append('g').attr('class', 'labels');

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    svg.call(zoom);

    // Create simulation
    const simulation = d3.forceSimulation<NodeData>(data.nodes)
      .force('link', d3.forceLink<NodeData, LinkData>(data.links)
        .id(d => d.id)
        .distance(linkDistance[0])
        .strength(0.1))
      .force('charge', d3.forceManyBody().strength(-forceStrength[0]))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.influence || 1) * 8 + 5));

    simulationRef.current = simulation;

    // Create links
    const links = linkGroup
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('class', 'network-link')
      .attr('stroke', d => LINK_COLORS[d.type as keyof typeof LINK_COLORS] || '#6b7280')
      .attr('stroke-width', d => Math.sqrt(d.strength || 1) * 2)
      .attr('stroke-opacity', 0.6);

    // Create nodes
    const nodes = nodeGroup
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('class', 'network-node')
      .attr('r', d => Math.sqrt(d.influence || 1) * 8 + 5)
      .attr('fill', d => NODE_COLORS.type[d.type])
      .attr('stroke', d => NODE_COLORS.status[d.status])
      .attr('stroke-width', 3)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, NodeData>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    // Add node labels
    const labels = labelGroup
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .attr('class', 'network-label')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', '#1f2937')
      .attr('pointer-events', 'none')
      .style('display', showLabels ? 'block' : 'none')
      .text(d => d.label || d.id);

    // Node interaction handlers
    nodes
      .on('mouseover', (event, d) => {
        setHoveredNode(d);
        onNodeHover?.(d);

        // Highlight connected nodes and links
        if (highlightPaths) {
          const connectedNodes = new Set<string>();

          links
            .style('stroke-opacity', (l: any) => {
              const isConnected = (l.source.id === d.id || l.target.id === d.id);
              if (isConnected) {
                connectedNodes.add(l.source.id);
                connectedNodes.add(l.target.id);
              }
              return isConnected ? 1 : 0.1;
            })
            .style('stroke-width', (l: any) =>
              (l.source.id === d.id || l.target.id === d.id) ? 4 : 1);

          nodes
            .style('opacity', (n: any) =>
              connectedNodes.has(n.id) || n.id === d.id ? 1 : 0.3);
        }
      })
      .on('mouseout', () => {
        setHoveredNode(null);
        onNodeHover?.(null);

        if (highlightPaths) {
          links.style('stroke-opacity', 0.6).style('stroke-width', null);
          nodes.style('opacity', 1);
        }
      })
      .on('click', (event, d) => {
        setSelectedNode(d);
        onNodeClick?.(d);
        event.stopPropagation();
      });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodes
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      labels
        .attr('x', d => d.x!)
        .attr('y', d => d.y!);
    });

    // Clear selection when clicking on empty space
    svg.on('click', () => {
      setSelectedNode(null);
    });

  }, [data, width, height, showLabels, highlightPaths, forceStrength, linkDistance, onNodeClick, onNodeHover]);

  // Update simulation forces when parameters change
  useEffect(() => {
    if (simulationRef.current) {
      simulationRef.current
        .force('charge', d3.forceManyBody().strength(-forceStrength[0]))
        .force('link', d3.forceLink<NodeData, LinkData>(data.links)
          .id(d => d.id)
          .distance(linkDistance[0])
          .strength(0.1))
        .alpha(0.3)
        .restart();
    }
  }, [forceStrength, linkDistance, data.links]);

  // Initialize simulation on component mount and data change
  useEffect(() => {
    initializeSimulation();

    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [initializeSimulation]);

  // Update label visibility
  useEffect(() => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .selectAll('.network-label')
        .style('display', showLabels ? 'block' : 'none');
    }
  }, [showLabels]);

  // Zoom controls
  const handleZoomIn = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(
          d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
          1.5
        );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(
          d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
          1 / 1.5
        );
    }
  };

  const handleResetZoom = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(500)
        .call(
          d3.zoom<SVGSVGElement, unknown>().transform as any,
          d3.zoomIdentity
        );
    }
  };

  const handleExportGraph = () => {
    if (svgRef.current) {
      const svgData = new XMLSerializer().serializeToString(svgRef.current);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const svgUrl = URL.createObjectURL(svgBlob);

      const downloadLink = document.createElement('a');
      downloadLink.href = svgUrl;
      downloadLink.download = 'network-graph.svg';
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
      URL.revokeObjectURL(svgUrl);
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Network className="w-5 h-5" />
              Network Visualization
            </CardTitle>
            <CardDescription>
              Interactive force-directed graph showing network structure and information flow
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleResetZoom}>
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleExportGraph}>
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Control Panel */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div className="space-y-2">
            <Label className="text-sm font-medium">Force Strength</Label>
            <Slider
              value={forceStrength}
              onValueChange={setForceStrength}
              min={10}
              max={100}
              step={5}
            />
            <div className="text-xs text-slate-600 dark:text-slate-400 text-center">
              {forceStrength[0]}
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Link Distance</Label>
            <Slider
              value={linkDistance}
              onValueChange={setLinkDistance}
              min={20}
              max={150}
              step={10}
            />
            <div className="text-xs text-slate-600 dark:text-slate-400 text-center">
              {linkDistance[0]}px
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Switch
                id="show-labels"
                checked={showLabels}
                onCheckedChange={() => {}} // Controlled by parent
              />
              <Label htmlFor="show-labels" className="text-sm">Show Labels</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="show-info"
                checked={showNodeInfo}
                onCheckedChange={setShowNodeInfo}
              />
              <Label htmlFor="show-info" className="text-sm">Node Info</Label>
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Zoom Level</Label>
            <div className="text-sm font-mono text-center py-2">
              {(zoomLevel * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div>
            <h4 className="font-medium mb-2">Node Types</h4>
            <div className="space-y-1">
              {Object.entries(NODE_COLORS.type).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2 text-sm">
                  <div
                    className="w-3 h-3 rounded-full border"
                    style={{ backgroundColor: color }}
                  />
                  <span className="capitalize">{type}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Node Status</h4>
            <div className="space-y-1">
              {Object.entries(NODE_COLORS.status).map(([status, color]) => (
                <div key={status} className="flex items-center gap-2 text-sm">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color, border: '2px solid #374151' }}
                  />
                  <span className="capitalize">{status}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Graph Area */}
        <div
          ref={containerRef}
          className="relative border rounded-lg bg-white dark:bg-slate-900 overflow-hidden"
          style={{ height: height }}
        >
          <svg
            ref={svgRef}
            width={width}
            height={height}
            className="w-full h-full"
          >
            <defs>
              {/* Gradient definitions for better visual appeal */}
              <radialGradient id="nodeGradient" cx="30%" cy="30%">
                <stop offset="0%" stopColor="rgba(255,255,255,0.8)" />
                <stop offset="100%" stopColor="rgba(255,255,255,0)" />
              </radialGradient>

              {/* Arrow markers for directed links */}
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon
                  points="0 0, 10 3.5, 0 7"
                  fill="#6b7280"
                />
              </marker>
            </defs>

            <g className="network-container" />
          </svg>

          {/* Node Information Panel */}
          {showNodeInfo && (hoveredNode || selectedNode) && (
            <div className="absolute top-4 right-4 p-3 bg-white dark:bg-slate-800 rounded-lg shadow-lg border max-w-xs">
              <div className="flex items-center gap-2 mb-2">
                <Info className="w-4 h-4" />
                <span className="font-medium">Node Information</span>
              </div>

              {(hoveredNode || selectedNode) && (
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>ID:</span>
                    <span className="font-mono">{(hoveredNode || selectedNode)!.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Type:</span>
                    <Badge variant="outline" className="text-xs">
                      {(hoveredNode || selectedNode)!.type}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <Badge variant="outline" className="text-xs">
                      {(hoveredNode || selectedNode)!.status}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Influence:</span>
                    <span className="font-mono">{(hoveredNode || selectedNode)!.influence?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Connections:</span>
                    <span className="font-mono">{(hoveredNode || selectedNode)!.connections || 0}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Network Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className="font-bold text-lg">{data.nodes.length}</div>
            <div className="text-slate-600 dark:text-slate-400">Nodes</div>
          </div>
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className="font-bold text-lg">{data.links.length}</div>
            <div className="text-slate-600 dark:text-slate-400">Edges</div>
          </div>
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className="font-bold text-lg">
              {((data.links.length * 2) / (data.nodes.length * (data.nodes.length - 1))).toFixed(3)}
            </div>
            <div className="text-slate-600 dark:text-slate-400">Density</div>
          </div>
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className="font-bold text-lg">
              {((data.links.length * 2) / data.nodes.length).toFixed(1)}
            </div>
            <div className="text-slate-600 dark:text-slate-400">Avg Degree</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};