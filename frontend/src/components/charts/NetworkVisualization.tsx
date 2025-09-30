// frontend/src/components/charts/NetworkVisualization.tsx

"use client";

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { NetworkData, NodeData, LinkData } from '@/types/network';

interface NetworkVisualizationProps {
  data: NetworkData;
  width?: number;
  height?: number;
  onNodeClick?: (node: NodeData) => void;
  onNodeHover?: (node: NodeData | null, event?: MouseEvent) => void;
}

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  data,
  width = 800,
  height = 600,
  onNodeClick,
  onNodeHover,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create main group for zoomable content
    const g = svg.append('g');

    // Create force simulation
    const simulation = d3.forceSimulation<NodeData>(data.nodes)
      .force('link', d3.forceLink<NodeData, LinkData>(data.links).id(d => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(20));

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('stroke', d => d.visual?.color || '#999')
      .attr('stroke-opacity', d => d.visual?.opacity || 0.6)
      .attr('stroke-width', d => d.visual?.thickness || Math.sqrt(d.weight || 1))
      .style('transition', 'all 0.3s ease');

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', d => d.visual?.size || 8)
      .attr('fill', d => d.visual?.color || '#69b3a2')
      .attr('stroke', d => d.visual?.strokeColor || '#fff')
      .attr('stroke-width', d => d.visual?.strokeWidth || 2)
      .attr('opacity', d => d.visual?.opacity || 1)
      .style('cursor', 'pointer')
      .style('transition', 'all 0.3s ease')
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

    // Create labels
    const label = g.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .text(d => d.label || d.id)
      .attr('font-size', 12)
      .attr('dx', 15)
      .attr('dy', 4)
      .style('font-family', 'sans-serif')
      .style('fill', '#333')
      .style('pointer-events', 'none');

    // Add hover and click interactions
    node
      .on('mouseenter', (event, d) => {
        setHoveredNode(d.id);

        // Call parent hover handler with mouse event
        if (onNodeHover) {
          onNodeHover(d, event as MouseEvent);
        }

        // Highlight connected nodes and links
        const connectedNodes = new Set([d.id]);
        data.links.forEach(link => {
          if (link.source === d.id || (typeof link.source === 'object' && link.source.id === d.id)) {
            connectedNodes.add(typeof link.target === 'string' ? link.target : link.target.id);
          }
          if (link.target === d.id || (typeof link.target === 'object' && link.target.id === d.id)) {
            connectedNodes.add(typeof link.source === 'string' ? link.source : link.source.id);
          }
        });

        // Dim non-connected elements with smooth transition
        node
          .transition()
          .duration(200)
          .style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.15);

        link
          .transition()
          .duration(200)
          .style('opacity', l => {
            const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
            const targetId = typeof l.target === 'string' ? l.target : l.target.id;
            return (sourceId === d.id || targetId === d.id) ? 0.8 : 0.05;
          })
          .style('stroke-width', l => {
            const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
            const targetId = typeof l.target === 'string' ? l.target : l.target.id;
            const currentWidth = l.visual?.thickness || Math.sqrt(l.weight || 1);
            return (sourceId === d.id || targetId === d.id) ? currentWidth * 1.5 : currentWidth;
          });

        label
          .transition()
          .duration(200)
          .style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.15);
      })
      .on('mouseleave', () => {
        setHoveredNode(null);

        // Call parent hover handler to clear
        if (onNodeHover) {
          onNodeHover(null);
        }

        // Reset opacity with smooth transition
        node
          .transition()
          .duration(200)
          .style('opacity', 1);

        link
          .transition()
          .duration(200)
          .style('opacity', l => l.visual?.opacity || 0.6)
          .style('stroke-width', l => l.visual?.thickness || Math.sqrt(l.weight || 1));

        label
          .transition()
          .duration(200)
          .style('opacity', 1);
      })
      .on('click', (event, d) => {
        if (onNodeClick) {
          onNodeClick(d);
        }
      });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as NodeData).x || 0)
        .attr('y1', d => (d.source as NodeData).y || 0)
        .attr('x2', d => (d.target as NodeData).x || 0)
        .attr('y2', d => (d.target as NodeData).y || 0);

      node
        .attr('cx', d => d.x || 0)
        .attr('cy', d => d.y || 0);

      label
        .attr('x', d => d.x || 0)
        .attr('y', d => d.y || 0);
    });

    // Cleanup function
    return () => {
      simulation.stop();
    };

  }, [data, width, height, onNodeClick]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="border border-slate-200 dark:border-slate-800 rounded-lg bg-white dark:bg-slate-950"
    >
      {/* Tooltip or additional UI elements can be added here */}
    </svg>
  );
};