// /home/ghost/fake-news-game-theory/frontend/src/app/not-found.tsx
'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Home, BarChart3, ArrowRight } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-[calc(100vh-200px)] flex items-center justify-center px-6 py-12 relative overflow-hidden">
      {/* Large Background "404" Text */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
        animate={{ opacity: [0.1, 0.15, 0.1] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
      >
        <span className="text-[20rem] md:text-[30rem] lg:text-[40rem] font-bold text-gray-200 dark:text-gray-800 leading-none">
          404
        </span>
      </motion.div>

      {/* Main Content */}
      <motion.div
        className="relative z-10 text-center max-w-2xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, staggerChildren: 0.15 }}
      >
        {/* Headline */}
        <motion.h1
          className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Lost in the Network
        </motion.h1>

        {/* Description */}
        <motion.p
          className="text-lg md:text-xl text-gray-600 dark:text-gray-400 mb-10 leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          It seems the page you are looking for has been moved, deleted, or does 
          not exist. Let&apos;s get you back on track.
        </motion.p>

        {/* Call to Action Buttons */}
        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <Link href="/">
            <Button
              size="lg"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 group min-w-[200px]"
            >
              <Home className="mr-2 h-5 w-5" />
              Return to Homepage
            </Button>
          </Link>

          <Link href="/simulation">
            <Button
              size="lg"
              variant="outline"
              className="border-2 border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 px-8 py-6 group min-w-[200px]"
            >
              <BarChart3 className="mr-2 h-5 w-5" />
              Explore the Dashboard
              <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
        </motion.div>

        {/* Optional: Additional helpful links */}
        <motion.div
          className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.9 }}
        >
          <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
            Looking for something specific?
          </p>
          <div className="flex flex-wrap gap-3 justify-center">
            <Link href="/classifier">
              <Button
                variant="link"
                className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 p-0 h-auto"
              >
                Classifier
              </Button>
            </Link>
            <span className="text-gray-300 dark:text-gray-700">•</span>
            <Link href="/analytics">
              <Button
                variant="link"
                className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 p-0 h-auto"
              >
                Analytics
              </Button>
            </Link>
            <span className="text-gray-300 dark:text-gray-700">•</span>
            <Link href="/equilibrium">
              <Button
                variant="link"
                className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 p-0 h-auto"
              >
                Equilibrium
              </Button>
            </Link>
            <span className="text-gray-300 dark:text-gray-700">•</span>
            <Link href="/research">
              <Button
                variant="link"
                className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 p-0 h-auto"
              >
                Research
              </Button>
            </Link>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}