// frontend/src/app/(dashboard)/simulation/components/NetworkGraph.tsx

"use client";

import React, { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { NetworkVisualization } from "@/components/charts/NetworkVisualization";
import { NetworkData, NodeData } from "@/types/network";
import { Spinner } from "@/components/ui/spinner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Info, Activity, Users, Network as NetworkIcon } from "lucide-react";

/**
 * Simulation step data representing the state at a specific time point
 */
export interface SimulationStep {
  time: number;
  nodeStates: Record<
    string,
    {
      status: "susceptible" | "infected" | "recovered" | "immune";
      influence_score?: number;
      credibility_score?: number;
    }
  >;
}

/**
 * Props for the NetworkGraph wrapper component
 */
interface NetworkGraphProps {
  /** Initial network structure with nodes and edges */
  data: NetworkData | null;
  /** Stream of simulation updates over time */
  simulationResults?: SimulationStep[];
  /** Whether the simulation is currently running */
  isRunning?: boolean;
  /** Callback when a node is clicked */
  onNodeClick?: (node: NodeData) => void;
  /** Optional width override */
  width?: number;
  /** Optional height override */
  height?: number;
  /** Optional className for container styling */
  className?: string;
}

/**
 * Color schemes for node visualization
 */
const STATUS_COLORS = {
  susceptible: "#e2e8f0", // slate-200 - not yet exposed
  infected: "#ef4444", // red-500 - currently spreading misinformation
  recovered: "#22c55e", // green-500 - no longer susceptible
  immune: "#3b82f6", // blue-500 - resistant to misinformation
} as const;

const USER_TYPE_COLORS = {
  spreader: "#f59e0b", // amber-500
  fact_checker: "#22c55e", // green-500
  platform: "#6366f1", // indigo-500
  regular_user: "#94a3b8", // slate-400
} as const;

/**
 * NetworkGraph: Smart wrapper component that manages data and state
 * for the underlying NetworkVisualization engine.
 *
 * Responsibilities:
 * - Data management and transformation
 * - Loading and empty states
 * - Live simulation animation
 * - State tracking and updates
 * - Prop delegation to visualization engine
 */
export function NetworkGraph({
  data,
  simulationResults = [],
  isRunning = false,
  onNodeClick,
  width = 800,
  height = 600,
  className,
}: NetworkGraphProps) {
  // Current simulation time step index
  const [currentStep, setCurrentStep] = useState(0);

  // Hovered node for tooltip display
  const [hoveredNode, setHoveredNode] = useState<NodeData | null>(null);

  // Tooltip position
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  /**
   * Reset simulation step when new results arrive
   */
  useEffect(() => {
    setCurrentStep(0);
  }, [simulationResults]);

  /**
   * Animate through simulation steps when running
   */
  useEffect(() => {
    if (!isRunning || simulationResults.length === 0) {
      return;
    }

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        // Loop back to start when reaching the end
        if (prev >= simulationResults.length - 1) {
          return 0;
        }
        return prev + 1;
      });
    }, 500); // 500ms per step for smooth animation

    return () => clearInterval(interval);
  }, [isRunning, simulationResults]);

  /**
   * Transform network data based on current simulation step
   * This creates dynamic node updates during simulation playback
   */
  const transformedData = useMemo<NetworkData | null>(() => {
    if (!data) return null;

    // If no simulation results, return original data
    if (simulationResults.length === 0 || !isRunning) {
      return data;
    }

    // Get current step data
    const stepData = simulationResults[currentStep];
    if (!stepData) return data;

    // Create new nodes array with updated states
    const updatedNodes = data.nodes.map((node) => {
      const nodeState = stepData.nodeStates[node.id];

      if (!nodeState) return node;

      // Calculate color based on status during simulation
      const statusColor = STATUS_COLORS[nodeState.status];

      // Update node visual properties
      return {
        ...node,
        visual: {
          ...node.visual,
          color: statusColor,
          size:
            nodeState.influence_score !== undefined
              ? Math.sqrt(nodeState.influence_score) * 10
              : node.visual.size,
          strokeColor:
            nodeState.status === "infected" ? "#dc2626" : node.visual.strokeColor,
          strokeWidth: nodeState.status === "infected" ? 3 : node.visual.strokeWidth,
        },
      };
    });

    return {
      ...data,
      nodes: updatedNodes,
    };
  }, [data, simulationResults, currentStep, isRunning]);

  /**
   * Handle node hover for tooltip
   */
  const handleNodeHover = React.useCallback((node: NodeData | null, event?: MouseEvent) => {
    setHoveredNode(node);
    if (node && event) {
      setTooltipPosition({ x: event.clientX, y: event.clientY });
    }
  }, []);

  /**
   * Loading State: Show spinner while data is being loaded
   */
  if (!data) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-xl border border-gray-200",
          "bg-white p-12",
          className
        )}
        style={{ height }}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col items-center gap-4"
        >
          <Spinner className="h-12 w-12 text-primary-600" />
          <div className="text-center space-y-2">
            <h3 className="text-lg font-medium text-gray-900">
              Loading Network
            </h3>
            <p className="text-sm text-gray-500">
              Generating network structure...
            </p>
          </div>
        </motion.div>
      </div>
    );
  }

  /**
   * Empty State: Show message when no nodes are available
   */
  if (data.nodes.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-xl border border-gray-200",
          "bg-gradient-to-br from-gray-50 to-gray-100 p-12",
          className
        )}
        style={{ height }}
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col items-center gap-4 max-w-md text-center"
        >
          <div className="p-4 rounded-full bg-gray-200">
            <NetworkIcon className="h-12 w-12 text-gray-400" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-gray-900">
              No Network Data
            </h3>
            <p className="text-sm text-gray-500">
              Configure simulation parameters and generate a network to visualize
              the social network structure and information flow.
            </p>
          </div>
        </motion.div>
      </div>
    );
  }

  /**
   * Main Visualization: Render the network graph
   */
  return (
    <TooltipProvider>
      <div className={cn("relative", className)}>
        {/* Simulation Progress Indicator */}
        <AnimatePresence>
          {isRunning && simulationResults.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-4 left-4 z-20 flex items-center gap-3 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-lg"
            >
              <Activity className="h-4 w-4 text-green-500 animate-pulse" />
              <div className="flex flex-col">
                <span className="text-xs font-medium text-gray-900">
                  Simulation Running
                </span>
                <span className="text-xs text-gray-500">
                  Step {currentStep + 1} / {simulationResults.length}
                </span>
              </div>
              <div className="ml-2 h-1.5 w-32 bg-gray-200 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-green-500 to-blue-500"
                  initial={{ width: 0 }}
                  animate={{
                    width: `${((currentStep + 1) / simulationResults.length) * 100}%`,
                  }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Network Statistics Panel */}
        <motion.div
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="absolute top-4 right-4 z-20 px-4 py-3 bg-white border border-gray-200 rounded-lg shadow-lg"
        >
          <div className="flex items-center gap-2 mb-2">
            <Users className="h-4 w-4 text-gray-500" />
            <span className="text-xs font-medium text-gray-900">
              Network Stats
            </span>
          </div>
          <div className="space-y-1 text-xs text-gray-600">
            <div className="flex justify-between gap-4">
              <span>Nodes:</span>
              <span className="font-mono font-medium text-gray-900">
                {data.nodes.length}
              </span>
            </div>
            <div className="flex justify-between gap-4">
              <span>Edges:</span>
              <span className="font-mono font-medium text-gray-900">
                {data.links.length}
              </span>
            </div>
            {data.statistics?.basic?.density !== undefined && (
              <div className="flex justify-between gap-4">
                <span>Density:</span>
                <span className="font-mono font-medium text-gray-900">
                  {data.statistics.basic.density.toFixed(3)}
                </span>
              </div>
            )}
          </div>
        </motion.div>

        {/* Node Hover Tooltip */}
        <AnimatePresence>
          {hoveredNode && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.15 }}
              className="absolute z-30 pointer-events-none"
              style={{
                left: tooltipPosition.x + 15,
                top: tooltipPosition.y + 15,
              }}
            >
              <div className="px-4 py-3 bg-gray-900 dark:bg-gray-800 text-white rounded-lg shadow-xl border border-gray-700 max-w-xs">
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-700">
                  <Info className="h-3.5 w-3.5 text-blue-400" />
                  <span className="text-xs font-semibold">Node Information</span>
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between gap-3">
                    <span className="text-gray-400">ID:</span>
                    <span className="font-mono text-gray-100">
                      {hoveredNode.id}
                    </span>
                  </div>
                  {hoveredNode.user_type && (
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-400">Type:</span>
                      <span
                        className="font-medium capitalize px-2 py-0.5 rounded text-xs"
                        style={{
                          backgroundColor:
                            USER_TYPE_COLORS[
                              hoveredNode.user_type as keyof typeof USER_TYPE_COLORS
                            ] + "30",
                          color:
                            USER_TYPE_COLORS[
                              hoveredNode.user_type as keyof typeof USER_TYPE_COLORS
                            ],
                        }}
                      >
                        {hoveredNode.user_type.replace("_", " ")}
                      </span>
                    </div>
                  )}
                  {hoveredNode.influence_score !== undefined && (
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-400">Influence:</span>
                      <span className="font-mono text-gray-100">
                        {hoveredNode.influence_score.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {hoveredNode.credibility_score !== undefined && (
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-400">Credibility:</span>
                      <span className="font-mono text-gray-100">
                        {hoveredNode.credibility_score.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {hoveredNode.centrality?.degree !== undefined && (
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-400">Connections:</span>
                      <span className="font-mono text-gray-100">
                        {hoveredNode.centrality.degree}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Visualization Component */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="rounded-xl overflow-hidden border border-gray-200 bg-white"
        >
          <NetworkVisualization
            data={transformedData}
            width={width}
            height={height}
            onNodeClick={onNodeClick}
            onNodeHover={handleNodeHover}
          />
        </motion.div>

        {/* Legend */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="absolute bottom-4 left-4 z-20 px-4 py-3 bg-white border border-gray-200 rounded-lg shadow-lg"
        >
          <div className="text-xs font-medium text-gray-900 mb-2">
            {isRunning ? "Node Status" : "User Types"}
          </div>
          <div className="space-y-1.5">
            {isRunning
              ? Object.entries(STATUS_COLORS).map(([status, color]) => (
                  <div key={status} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full border border-gray-300"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs text-gray-600 capitalize">
                      {status}
                    </span>
                  </div>
                ))
              : Object.entries(USER_TYPE_COLORS).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full border border-gray-300"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs text-gray-600 capitalize">
                      {type.replace("_", " ")}
                    </span>
                  </div>
                ))}
          </div>
        </motion.div>
      </div>
    </TooltipProvider>
  );
}