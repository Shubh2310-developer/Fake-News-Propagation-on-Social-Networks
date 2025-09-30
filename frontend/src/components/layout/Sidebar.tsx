// frontend/src/components/layout/Sidebar.tsx

"use client";

import { motion, AnimatePresence } from 'framer-motion';
import { useUIStore } from '@/store';
import { cn } from '@/lib/utils';
import { Navigation } from './Navigation';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Sidebar() {
  const { isSidebarOpen, toggleSidebar, isSidebarCollapsed, toggleSidebarCollapse } = useUIStore();

  return (
    <>
      {/* Desktop Sidebar - Collapsible */}
      <motion.aside
        initial={false}
        animate={{
          width: isSidebarCollapsed ? '4rem' : '16rem', // 64px : 256px
        }}
        transition={{
          duration: 0.3,
          ease: [0.4, 0, 0.2, 1], // Tailwind's ease-in-out
        }}
        className={cn(
          "hidden md:flex flex-col border-r border-gray-200 bg-white",
          "dark:border-gray-800 dark:bg-gray-900",
          "relative overflow-hidden",
          "transition-colors duration-200"
        )}
      >
        {/* Collapse Toggle Button */}
        <div className="absolute top-4 -right-3 z-10">
          <Button
            variant="outline"
            size="icon"
            onClick={toggleSidebarCollapse}
            className={cn(
              "h-6 w-6 rounded-full bg-white dark:bg-gray-900",
              "shadow-md border-gray-300 dark:border-gray-700",
              "hover:bg-gray-100 dark:hover:bg-gray-800"
            )}
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isSidebarCollapsed ? (
              <ChevronRight className="h-3 w-3" />
            ) : (
              <ChevronLeft className="h-3 w-3" />
            )}
          </Button>
        </div>

        {/* Sidebar Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <AnimatePresence mode="wait">
            {!isSidebarCollapsed ? (
              <motion.div
                key="expanded"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                {/* Logo/Title */}
                <div className="mb-6 px-2">
                  <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    GTDS Platform
                  </h2>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Game Theory Dashboard
                  </p>
                </div>

                {/* Navigation */}
                <Navigation isCollapsed={false} />
              </motion.div>
            ) : (
              <motion.div
                key="collapsed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col items-center"
              >
                {/* Collapsed Logo */}
                <div className="mb-6">
                  <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
                    <span className="text-white font-bold text-sm">GT</span>
                  </div>
                </div>

                {/* Collapsed Navigation */}
                <Navigation isCollapsed={true} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer Section (optional) */}
        {!isSidebarCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="border-t border-gray-200 dark:border-gray-800 p-4"
          >
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <p>&copy; 2025 GTDS Project</p>
            </div>
          </motion.div>
        )}
      </motion.aside>

      {/* Mobile Sidebar - Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={toggleSidebar}
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
              aria-hidden="true"
            />

            {/* Mobile Sidebar Panel */}
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{
                duration: 0.3,
                ease: [0.4, 0, 0.2, 1],
              }}
              className={cn(
                "fixed inset-y-0 left-0 z-50 w-64 md:hidden",
                "bg-white dark:bg-gray-900",
                "border-r border-gray-200 dark:border-gray-800",
                "shadow-xl overflow-hidden"
              )}
            >
              {/* Mobile Header */}
              <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-800 p-4">
                <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                  GTDS Platform
                </h2>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleSidebar}
                  aria-label="Close sidebar"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              {/* Mobile Navigation */}
              <div className="overflow-y-auto p-4">
                <Navigation isCollapsed={false} />
              </div>

              {/* Mobile Footer */}
              <div className="absolute bottom-0 left-0 right-0 border-t border-gray-200 dark:border-gray-800 p-4 bg-white dark:bg-gray-900">
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  <p>&copy; 2025 GTDS Project</p>
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}