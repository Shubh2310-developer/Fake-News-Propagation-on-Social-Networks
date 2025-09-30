// frontend/src/app/(dashboard)/layout.tsx

"use client";

import { motion, AnimatePresence } from 'framer-motion';
import { usePathname } from 'next/navigation';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { Breadcrumbs } from '@/components/layout/Breadcrumbs';
import { useUIStore } from '@/store';
import { cn } from '@/lib/utils';
import { useEffect, useState } from 'react';
import * as React from 'react';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

// Page transition variants
const pageVariants = {
  initial: {
    opacity: 0,
    y: 8,
    scale: 0.99,
  },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: [0.4, 0, 0.2, 1],
    },
  },
  exit: {
    opacity: 0,
    y: -8,
    scale: 0.99,
    transition: {
      duration: 0.2,
      ease: [0.4, 0, 0.2, 1],
    },
  },
};

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const pathname = usePathname();
  const { isSidebarCollapsed, theme } = useUIStore();

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors duration-200">
      {/* Header - Fixed at top */}
      <Header />

      {/* Main Layout Container */}
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar - Collapsible */}
        <Sidebar />

        {/* Main Content Area */}
        <motion.main
          initial={false}
          animate={{
            marginLeft: 0, // Sidebar is absolute/fixed, no margin needed
            transition: {
              duration: 0.3,
              ease: [0.4, 0, 0.2, 1],
            },
          }}
          className={cn(
            "flex-1 overflow-y-auto transition-all duration-300",
            "bg-gradient-to-br from-gray-50 via-white to-gray-100",
            "dark:from-gray-950 dark:via-gray-900 dark:to-gray-950"
          )}
        >
          {/* Content Container with max-width and padding */}
          <div className="container mx-auto px-4 py-6 lg:px-6 lg:py-8 max-w-7xl">
            {/* Breadcrumbs Navigation */}
            <Breadcrumbs />

            {/* Page Content with AnimatePresence for transitions */}
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={pathname} // Animate when route changes
                variants={pageVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="min-h-[calc(100vh-12rem)]"
              >
                {/* Glass-morphism content wrapper */}
                <div
                  className={cn(
                    "rounded-xl transition-all duration-200",
                    "bg-white/70 dark:bg-gray-900/70",
                    "backdrop-blur-sm",
                    "shadow-sm border border-gray-200/50 dark:border-gray-800/50",
                    "p-6 lg:p-8"
                  )}
                >
                  {children}
                </div>
              </motion.div>
            </AnimatePresence>

            {/* Footer Section */}
            <motion.footer
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="mt-12 py-6 border-t border-gray-200 dark:border-gray-800"
            >
              <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-2">
                  <span>© 2025 GTDS Project</span>
                  <span className="hidden sm:inline">•</span>
                  <span className="hidden sm:inline">Powered by Game Theory</span>
                </div>
                <div className="flex items-center gap-4">
                  <a
                    href="/about"
                    className="hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
                  >
                    About
                  </a>
                  <a
                    href="/docs"
                    className="hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
                  >
                    Documentation
                  </a>
                  <a
                    href="/research"
                    className="hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
                  >
                    Research
                  </a>
                </div>
              </div>
            </motion.footer>
          </div>

          {/* Scroll to top button (optional enhancement) */}
          <ScrollToTopButton />
        </motion.main>
      </div>
    </div>
  );
}

// Scroll to Top Button Component
function ScrollToTopButton() {
  const [isVisible, setIsVisible] = React.useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      if (window.scrollY > 300) {
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    window.addEventListener('scroll', toggleVisibility);
    return () => window.removeEventListener('scroll', toggleVisibility);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          onClick={scrollToTop}
          className={cn(
            "fixed bottom-8 right-8 z-30",
            "p-3 rounded-full",
            "bg-blue-600 hover:bg-blue-700",
            "text-white shadow-lg",
            "transition-colors duration-200",
            "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          )}
          aria-label="Scroll to top"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <svg
            className="h-5 w-5"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </motion.button>
      )}
    </AnimatePresence>
  );
}