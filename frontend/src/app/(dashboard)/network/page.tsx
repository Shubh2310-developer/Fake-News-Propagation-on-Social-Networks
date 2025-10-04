"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Network,
  BarChart3,
  Users,
  TrendingUp,
  GitBranch,
  Zap,
  Activity,
  Target,
  Info,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { BarChart } from '@/components/charts/BarChart';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';

interface NetworkNode {
  id: string;
  user_type: string;
  influence_score: number;
  degree: number;
  betweenness: number;
  eigenvector: number;
  pagerank: number;
  community_id: number;
}

interface NetworkStats {
  nodes: number;
  edges: number;
  density: number;
  avgClustering: number;
  diameter: number;
  avgDegree: number;
  communities: number;
  modularity: number;
}

interface Community {
  id: number;
  size: number;
  dominantType: string;
}

function generateMockNetwork(type: string, nodeCount: number) {
  const nodes: NetworkNode[] = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node-${i}`,
    user_type: i < 5 ? 'spreader' : i < 15 ? 'moderator' : i < 20 ? 'bot' : 'user',
    influence_score: Math.random() * 100,
    degree: Math.floor(Math.random() * 20) + 1,
    betweenness: Math.random(),
    eigenvector: Math.random(),
    pagerank: Math.random(),
    community_id: Math.floor(i / 20),
  }));

  const stats: NetworkStats = {
    nodes: nodeCount,
    edges: Math.floor(nodeCount * 1.5),
    density: (nodeCount * 1.5) / (nodeCount * (nodeCount - 1) / 2),
    avgClustering: 0.45 + Math.random() * 0.2,
    diameter: Math.floor(Math.log(nodeCount) * 2),
    avgDegree: 3 + Math.random() * 5,
    communities: Math.floor(nodeCount / 20) + 1,
    modularity: 0.65 + Math.random() * 0.2,
  };

  return { nodes, stats };
}

function getCommunities(nodes: NetworkNode[]): Community[] {
  const communityMap = new Map<number, NetworkNode[]>();

  nodes.forEach(node => {
    if (!communityMap.has(node.community_id)) {
      communityMap.set(node.community_id, []);
    }
    communityMap.get(node.community_id)?.push(node);
  });

  return Array.from(communityMap.entries()).map(([id, members]) => {
    const typeCounts = members.reduce((acc, node) => {
      acc[node.user_type] = (acc[node.user_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const dominantType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0][0];

    return {
      id,
      size: members.length,
      dominantType,
    };
  }).sort((a, b) => b.size - a.size);
}

function useCountUp(end: number, duration: number = 1500) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime: number | null = null;
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setCount(end * easeOutQuart);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };
    requestAnimationFrame(animate);
  }, [end, duration]);

  return count;
}

interface MetricsCardProps {
  title: string;
  value: number;
  suffix?: string;
  icon: React.ReactNode;
  description: string;
  decimals?: number;
  delay?: number;
}

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  suffix = '',
  icon,
  description,
  decimals = 0,
  delay = 0,
}) => {
  const animatedValue = useCountUp(value, 1500);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay }}
    >
      <Card className="hover:shadow-lg transition-shadow duration-200">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-700">
            {title}
          </CardTitle>
          <div className="p-2 bg-blue-100 rounded-lg">
            {icon}
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-gray-900">
            {animatedValue.toFixed(decimals)}{suffix}
          </div>
          <p className="text-xs text-gray-600 mt-2">
            {description}
          </p>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default function NetworkAnalysisPage() {
  const [selectedNetwork, setSelectedNetwork] = useState<string>('existing-1');
  const [networkType, setNetworkType] = useState<string>('barabasi-albert');
  const [nodeCount, setNodeCount] = useState<number>(100);
  const [isGenerating, setIsGenerating] = useState(false);
  const [colorBy, setColorBy] = useState<string>('user_type');
  const [sizeBy, setSizeBy] = useState<string>('degree');
  const [centralityMetric, setCentralityMetric] = useState<string>('degree');

  const [networkData, setNetworkData] = useState(generateMockNetwork('barabasi-albert', 100));

  const handleGenerateNetwork = async () => {
    setIsGenerating(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    const newNetwork = generateMockNetwork(networkType, nodeCount);
    setNetworkData(newNetwork);
    setSelectedNetwork('generated');
    setIsGenerating(false);
  };

  const handleNetworkSelect = (value: string) => {
    setSelectedNetwork(value);
    if (value !== 'generate-new') {
      const count = value === 'existing-1' ? 100 : value === 'existing-2' ? 150 : 200;
      setNetworkData(generateMockNetwork('barabasi-albert', count));
    }
  };

  const getCentralityData = () => {
    return networkData.nodes
      .map(node => ({
        node: node.id,
        value: node[centralityMetric as keyof NetworkNode] as number,
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 20);
  };

  const centralitySeries = [
    { dataKey: 'value', name: 'Score', color: '#3b82f6' },
  ];

  const getTopNodes = (metric: string) => {
    return networkData.nodes
      .sort((a, b) => {
        const aVal = a[metric as keyof NetworkNode] as number;
        const bVal = b[metric as keyof NetworkNode] as number;
        return bVal - aVal;
      })
      .slice(0, 10);
  };

  const communities = getCommunities(networkData.nodes);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-3 bg-blue-100 rounded-lg">
            <Network className="h-6 w-6 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Social Network Analysis
          </h1>
        </div>
        <p className="text-gray-600 text-lg">
          Generate, visualize, and analyze the topology and properties of complex networks
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-purple-600" />
              Network Controls
            </CardTitle>
            <CardDescription>Select or generate a network to analyze</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Network
                </label>
                <Select value={selectedNetwork} onValueChange={handleNetworkSelect}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="existing-1">Sample Network 1 (100 nodes)</SelectItem>
                    <SelectItem value="existing-2">Sample Network 2 (150 nodes)</SelectItem>
                    <SelectItem value="existing-3">Sample Network 3 (200 nodes)</SelectItem>
                    <SelectItem value="generate-new">Generate New Network</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <AnimatePresence>
                {selectedNetwork === 'generate-new' && (
                  <>
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      <label className="text-sm font-medium text-gray-700 mb-2 block">
                        Network Type
                      </label>
                      <Select value={networkType} onValueChange={setNetworkType}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="barabasi-albert">Barabasi-Albert</SelectItem>
                          <SelectItem value="watts-strogatz">Watts-Strogatz</SelectItem>
                          <SelectItem value="erdos-renyi">Erdos-Renyi</SelectItem>
                        </SelectContent>
                      </Select>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3, delay: 0.1 }}
                      className="space-y-2"
                    >
                      <label className="text-sm font-medium text-gray-700 mb-2 block">
                        Number of Nodes: {nodeCount}
                      </label>
                      <Slider
                        min={50}
                        max={200}
                        step={10}
                        value={[nodeCount]}
                        onValueChange={(value) => setNodeCount(value[0])}
                      />
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3, delay: 0.2 }}
                      className="flex items-end"
                    >
                      <Button
                        onClick={handleGenerateNetwork}
                        disabled={isGenerating}
                        className="w-full"
                      >
                        {isGenerating ? (
                          <>
                            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Zap className="mr-2 h-4 w-4" />
                            Generate Network
                          </>
                        )}
                      </Button>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="xl:col-span-7"
        >
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-green-600" />
                Network Visualization
              </CardTitle>
              <CardDescription>Interactive graph with dynamic visual encoding</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">
                    Color Nodes By
                  </label>
                  <Select value={colorBy} onValueChange={setColorBy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="user_type">User Type</SelectItem>
                      <SelectItem value="community_id">Community ID</SelectItem>
                      <SelectItem value="influence_score">Influence Score</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">
                    Size Nodes By
                  </label>
                  <Select value={sizeBy} onValueChange={setSizeBy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="degree">Degree</SelectItem>
                      <SelectItem value="betweenness">Betweenness Centrality</SelectItem>
                      <SelectItem value="influence_score">Influence Score</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="relative w-full h-[600px] bg-gradient-to-br from-slate-50 to-blue-50 rounded-lg border-2 border-dashed border-slate-300 flex items-center justify-center">
                <div className="text-center">
                  <Network className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                  <p className="text-lg font-semibold text-slate-600">Network Visualization</p>
                  <p className="text-sm text-slate-500 mt-2">
                    Interactive D3.js force-directed graph would render here
                  </p>
                  <div className="mt-4 flex items-center justify-center gap-2">
                    <Badge variant="outline">Color: {colorBy}</Badge>
                    <Badge variant="outline">Size: {sizeBy}</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="xl:col-span-5"
        >
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-orange-600" />
                Statistical Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="centrality">Centrality</TabsTrigger>
                  <TabsTrigger value="communities">Communities</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4 mt-4">
                  <MetricsCard
                    title="Nodes"
                    value={networkData.stats.nodes}
                    icon={<Users className="h-5 w-5 text-blue-600" />}
                    description="Total number of nodes"
                    delay={0.1}
                  />
                  <MetricsCard
                    title="Edges"
                    value={networkData.stats.edges}
                    icon={<GitBranch className="h-5 w-5 text-green-600" />}
                    description="Total number of connections"
                    delay={0.2}
                  />
                  <MetricsCard
                    title="Density"
                    value={networkData.stats.density}
                    decimals={3}
                    icon={<Activity className="h-5 w-5 text-purple-600" />}
                    description="Network connectivity ratio"
                    delay={0.3}
                  />
                  <MetricsCard
                    title="Avg Clustering"
                    value={networkData.stats.avgClustering}
                    decimals={3}
                    icon={<Target className="h-5 w-5 text-orange-600" />}
                    description="Average clustering coefficient"
                    delay={0.4}
                  />
                  <MetricsCard
                    title="Diameter"
                    value={networkData.stats.diameter}
                    icon={<TrendingUp className="h-5 w-5 text-red-600" />}
                    description="Longest shortest path"
                    delay={0.5}
                  />
                </TabsContent>

                <TabsContent value="centrality" className="space-y-6 mt-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <label className="text-sm font-semibold text-gray-900 mb-3 block">
                      Select Centrality Metric
                    </label>
                    <Select value={centralityMetric} onValueChange={setCentralityMetric}>
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="degree">Degree Centrality</SelectItem>
                        <SelectItem value="betweenness">Betweenness Centrality</SelectItem>
                        <SelectItem value="eigenvector">Eigenvector Centrality</SelectItem>
                        <SelectItem value="pagerank">PageRank</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-600 mt-2">
                      {centralityMetric === 'degree' && 'Number of direct connections'}
                      {centralityMetric === 'betweenness' && 'Importance as a bridge between nodes'}
                      {centralityMetric === 'eigenvector' && 'Influence based on connected nodes'}
                      {centralityMetric === 'pagerank' && 'Google PageRank algorithm score'}
                    </p>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 mb-4">
                      Distribution of {centralityMetric.charAt(0).toUpperCase() + centralityMetric.slice(1)} Scores
                    </h4>
                    <div className="w-full h-[450px] bg-white p-6 rounded-lg border-2 border-gray-200 overflow-hidden">
                      <BarChart
                        data={getCentralityData()}
                        series={centralitySeries}
                        xAxisKey="node"
                      />
                    </div>
                  </div>

                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-900 mb-4">
                      Top 10 Most Central Nodes
                    </h4>
                    <div className="border-2 border-gray-200 rounded-lg overflow-hidden bg-white">
                      <Table>
                        <TableHeader>
                          <TableRow className="bg-gray-50">
                            <TableHead className="text-gray-900 font-bold py-3">Node ID</TableHead>
                            <TableHead className="text-gray-900 font-bold py-3 text-right">Score</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {getTopNodes(centralityMetric).map((node, index) => (
                            <TableRow key={node.id} className="border-b border-gray-100">
                              <TableCell className="font-mono text-sm text-gray-900 py-3">
                                <div className="flex items-center gap-2">
                                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold">
                                    {index + 1}
                                  </span>
                                  {node.id}
                                </div>
                              </TableCell>
                              <TableCell className="font-bold text-gray-900 py-3 text-right">
                                {(node[centralityMetric as keyof NetworkNode] as number).toFixed(3)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="communities" className="space-y-4 mt-4">
                  <MetricsCard
                    title="Communities"
                    value={networkData.stats.communities}
                    icon={<Users className="h-5 w-5 text-blue-600" />}
                    description="Number of detected communities"
                    delay={0.1}
                  />
                  <MetricsCard
                    title="Modularity"
                    value={networkData.stats.modularity}
                    decimals={3}
                    icon={<Target className="h-5 w-5 text-green-600" />}
                    description="Community structure quality"
                    delay={0.2}
                  />

                  <div className="mt-4">
                    <h4 className="text-sm font-semibold text-gray-900 mb-3">
                      Largest Communities
                    </h4>
                    <div className="border rounded-lg overflow-hidden">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="text-gray-900">ID</TableHead>
                            <TableHead className="text-gray-900">Size</TableHead>
                            <TableHead className="text-gray-900">Type</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {communities.slice(0, 10).map((community) => (
                            <TableRow key={community.id}>
                              <TableCell className="font-mono">{community.id}</TableCell>
                              <TableCell className="font-semibold">{community.size}</TableCell>
                              <TableCell>
                                <Badge variant="outline">{community.dominantType}</Badge>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200 mt-4">
                    <div className="flex items-start gap-2">
                      <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-900">
                          Community Detection
                        </p>
                        <p className="text-sm text-gray-600 mt-1">
                          Communities detected using Louvain algorithm. Higher modularity indicates stronger community structure.
                        </p>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
