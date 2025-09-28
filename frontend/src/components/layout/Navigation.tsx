// frontend/src/components/layout/Navigation.tsx

"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BarChart, BrainCircuit, Bot, Network, Scale } from 'lucide-react';
import { cn } from '@/lib/utils';

const navLinks = [
  { href: '/simulation', label: 'Simulation', icon: BrainCircuit },
  { href: '/analytics', label: 'Analytics', icon: BarChart },
  { href: '/classifier', label: 'Classifier', icon: Bot },
  { href: '/network', label: 'Network Analysis', icon: Network },
  { href: '/equilibrium', label: 'Equilibrium', icon: Scale },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-col gap-2">
      {navLinks.map((link) => {
        const isActive = pathname.startsWith(link.href);
        const Icon = link.icon;

        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              'flex items-center gap-3 rounded-lg px-3 py-2 text-slate-600 transition-all hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-50',
              isActive && 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-50'
            )}
            aria-current={isActive ? 'page' : undefined}
          >
            <Icon className="h-4 w-4" />
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}