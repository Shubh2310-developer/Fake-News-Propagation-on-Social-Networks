// frontend/src/components/layout/Breadcrumbs.tsx

"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronRight, Home } from 'lucide-react';
import { Fragment } from 'react';
import { cn } from '@/lib/utils';

// Map of route segments to readable labels
const routeLabels: Record<string, string> = {
  simulation: 'Simulation',
  analytics: 'Analytics',
  classifier: 'Classifier',
  network: 'Network Analysis',
  equilibrium: 'Nash Equilibrium',
  datasets: 'Datasets',
  about: 'About',
  research: 'Research',
  methodology: 'Methodology',
};

export function Breadcrumbs() {
  const pathname = usePathname();

  // Split pathname and filter out empty segments
  const segments = pathname.split('/').filter(Boolean);

  // Don't show breadcrumbs on home page
  if (segments.length === 0) {
    return null;
  }

  // Build breadcrumb items
  const breadcrumbs = segments.map((segment, index) => {
    const href = '/' + segments.slice(0, index + 1).join('/');
    const label = routeLabels[segment] || segment.charAt(0).toUpperCase() + segment.slice(1);
    const isLast = index === segments.length - 1;

    return {
      href,
      label,
      isLast,
    };
  });

  return (
    <nav
      aria-label="Breadcrumb"
      className="mb-6 flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400"
    >
      {/* Home Link */}
      <Link
        href="/"
        className={cn(
          "flex items-center gap-1.5 px-2 py-1 rounded-md",
          "transition-colors duration-150",
          "hover:bg-gray-100 dark:hover:bg-gray-800",
          "hover:text-gray-900 dark:hover:text-gray-100"
        )}
        aria-label="Home"
      >
        <Home className="h-4 w-4" />
        <span className="hidden sm:inline">Home</span>
      </Link>

      {/* Breadcrumb Items */}
      {breadcrumbs.map((crumb, index) => (
        <Fragment key={crumb.href}>
          <ChevronRight className="h-4 w-4 text-gray-400 dark:text-gray-600" aria-hidden="true" />

          {crumb.isLast ? (
            <span
              className="px-2 py-1 font-medium text-gray-900 dark:text-gray-100"
              aria-current="page"
            >
              {crumb.label}
            </span>
          ) : (
            <Link
              href={crumb.href}
              className={cn(
                "px-2 py-1 rounded-md",
                "transition-colors duration-150",
                "hover:bg-gray-100 dark:hover:bg-gray-800",
                "hover:text-gray-900 dark:hover:text-gray-100"
              )}
            >
              {crumb.label}
            </Link>
          )}
        </Fragment>
      ))}
    </nav>
  );
}