"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface Dataset {
  id: string;
  name: string;
  description: string;
  size: string;
  features: string[];
  primaryUsage: string;
  sourceUrl?: string;
  methodologyUrl?: string;
}

const DatasetsPage = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch("/api/data/datasets");
        if (!response.ok) {
          throw new Error("Failed to fetch datasets");
        }
        const data = await response.json();
        setDatasets(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100,
      },
    },
  };

  const hoverEffect = {
    scale: 1.03,
    boxShadow: "0px 10px 20px rgba(0, 0, 0, 0.1)",
  };

  return (
    <div className="container mx-auto py-12">
      <PageHeader
        title="Project Datasets"
        subtitle="A comprehensive overview of the data sources powering our research and analysis."
      />

      {loading && (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-8"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {Array.from({ length: 6 }).map((_, index) => (
            <motion.div key={index} variants={itemVariants}>
              <Card>
                <CardHeader>
                  <Skeleton className="h-6 w-3/4" />
                </CardHeader>
                <CardContent className="space-y-4">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <div className="flex flex-wrap gap-2">
                    <Skeleton className="h-6 w-20" />
                    <Skeleton className="h-6 w-24" />
                    <Skeleton className="h-6 w-16" />
                  </div>
                </CardContent>
                <CardFooter>
                  <Skeleton className="h-10 w-28" />
                </CardFooter>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}

      {error && (
        <div className="mt-8">
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              Failed to load dataset information. Please try again later.
            </AlertDescription>
          </Alert>
        </div>
      )}

      {!loading && !error && (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-8"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {datasets.map((dataset) => (
            <motion.div
              key={dataset.id}
              variants={itemVariants}
              whileHover={hoverEffect}
            >
              <Card className="flex flex-col h-full">
                <CardHeader>
                  <CardTitle>{dataset.name}</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow space-y-4">
                  <p className="text-muted-foreground">{dataset.description}</p>
                  <div>
                    <h4 className="font-semibold">Size</h4>
                    <p>{dataset.size}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold">Features</h4>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {dataset.features.map((feature) => (
                        <Badge key={feature} variant="secondary">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold">Primary Usage</h4>
                    <p>{dataset.primaryUsage}</p>
                  </div>
                </CardContent>
                <CardFooter>
                  {dataset.sourceUrl && (
                    <Button asChild variant="outline">
                      <a href={dataset.sourceUrl} target="_blank" rel="noopener noreferrer">
                        View Source
                      </a>
                    </Button>
                  )}
                  {dataset.methodologyUrl && (
                    <Button asChild variant="link">
                      <a href={dataset.methodologyUrl}>Learn More</a>
                    </Button>
                  )}
                </CardFooter>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
};

export default DatasetsPage;
