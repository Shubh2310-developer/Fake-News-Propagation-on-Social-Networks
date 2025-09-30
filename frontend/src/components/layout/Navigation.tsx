// frontend/src/components/layout/Navigation.tsx

"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BarChart, BrainCircuit, Bot, Network, Scale } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

const navLinks = [
  { href: '/simulation', label: 'Simulation', icon: BrainCircuit, description: 'Run game theory simulations' },
  { href: '/analytics', label: 'Analytics', icon: BarChart, description: 'View analytics dashboard' },
  { href: '/classifier', label: 'Classifier', icon: Bot, description: 'Fake news detection' },
  { href: '/equilibrium', label: 'Equilibrium', icon: Scale, description: 'Nash equilibrium analysis' },
  { href: '/network', label: 'Network', icon: Network, description: 'Network analysis tools' },
];

interface NavigationProps {
  isCollapsed: boolean;
}

export function Navigation({ isCollapsed }: NavigationProps) {
  const pathname = usePathname();

  if (isCollapsed) {
    return (
      <TooltipProvider delayDuration={0}>
        <nav className="flex flex-col gap-2">
          {navLinks.map((link) => {
            const isActive = pathname.startsWith(link.href);
            const Icon = link.icon;

            return (
              <Tooltip key={link.href}>
                <TooltipTrigger asChild>
                  <Link
                    href={link.href}
                    className={cn(
                      'relative flex items-center justify-center w-10 h-10 rounded-lg',
                      'transition-all duration-200',
                      'hover:bg-gray-100 dark:hover:bg-gray-800',
                      isActive
                        ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
                    )}
                    aria-current={isActive ? 'page' : undefined}
                    aria-label={link.label}
                  >
                    {/* Active Indicator */}
                    {isActive && (
                      <motion.div
                        layoutId="activeIndicatorCollapsed"
                        className="absolute left-0 w-1 h-6 bg-blue-600 rounded-r-full"
                        transition={{
                          type: 'spring',
                          stiffness: 380,
                          damping: 30,
                        }}
                      />
                    )}
                    <Icon className="h-5 w-5" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right" className="font-medium">
                  <p className="font-semibold">{link.label}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{link.description}</p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </nav>
      </TooltipProvider>
    );
  }

  return (
    <nav className="flex flex-col gap-1" role="navigation" aria-label="Main navigation">
      {navLinks.map((link) => {
        const isActive = pathname.startsWith(link.href);
        const Icon = link.icon;

        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              'group relative flex items-center gap-3 rounded-lg px-3 py-2.5',
              'transition-all duration-200',
              'hover:bg-gray-100 dark:hover:bg-gray-800',
              'hover:translate-x-0.5',
              isActive
                ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400 font-medium'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
            )}
            aria-current={isActive ? 'page' : undefined}
          >
            {/* Active Indicator - Left Border */}
            {isActive && (
              <motion.div
                layoutId="activeIndicator"
                className="absolute left-0 w-1 h-8 bg-blue-600 dark:bg-blue-400 rounded-r-full"
                transition={{
                  type: 'spring',
                  stiffness: 380,
                  damping: 30,
                }}
              />
            )}

            {/* Icon */}
            <Icon
              className={cn(
                "h-5 w-5 transition-transform duration-200",
                "group-hover:scale-110",
                isActive && "scale-110"
              )}
            />

            {/* Label */}
            <span className="text-sm">{link.label}</span>

            {/* Hover Indicator - Right Highlight */}
            <span
              className={cn(
                "absolute right-3 opacity-0 group-hover:opacity-100",
                "transition-opacity duration-200",
                "w-1 h-1 rounded-full bg-gray-400 dark:bg-gray-600",
                isActive && "bg-blue-600 dark:bg-blue-400"
              )}
            />
          </Link>
        );
      })}
    </nav>
  );
}