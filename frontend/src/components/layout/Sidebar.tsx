// frontend/src/components/layout/Sidebar.tsx

"use client";

import { useUIStore } from '@/store';
import { cn } from '@/lib/utils';
import { Navigation } from './Navigation';

export function Sidebar() {
  const { isSidebarOpen } = useUIStore();

  return (
    <>
      {/* Permanent sidebar for larger screens */}
      <aside className="hidden md:flex md:w-64 flex-col border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
        <div className="p-4">
          <Navigation />
        </div>
      </aside>

      {/* Mobile sidebar overlay */}
      {isSidebarOpen && (
        <div
          className={cn(
            'fixed inset-0 z-30 bg-black/60 transition-opacity md:hidden',
            isSidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
          )}
        >
          <aside className="fixed inset-y-0 left-0 z-40 w-64 bg-white dark:bg-slate-950 p-4">
            <Navigation />
          </aside>
        </div>
      )}
    </>
  );
}