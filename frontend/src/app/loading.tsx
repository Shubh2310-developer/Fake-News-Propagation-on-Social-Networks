// /home/ghost/fake-news-game-theory/frontend/src/app/loading.tsx
'use client';

import { motion } from 'framer-motion';

export default function Loading() {
  return (
    <div className="fixed top-0 left-0 right-0 z-50 pointer-events-none">
      {/* Semi-transparent track */}
      <div className="h-[3px] w-full bg-blue-500/20" />
      
      {/* Animated progress bar */}
      <motion.div
        className="absolute top-0 left-0 h-[3px] bg-gradient-to-r from-blue-600 to-blue-400 shadow-lg shadow-blue-500/50"
        initial={{ width: '0%' }}
        animate={{ 
          width: ['0%', '40%', '70%', '90%'],
        }}
        transition={{
          duration: 2,
          times: [0, 0.3, 0.7, 1],
          ease: 'easeOut',
        }}
        exit={{
          width: '100%',
          opacity: 0,
          transition: {
            width: { duration: 0.3, ease: 'easeIn' },
            opacity: { duration: 0.2, delay: 0.3 }
          }
        }}
      />
      
      {/* Animated glow effect */}
      <motion.div
        className="absolute top-0 right-0 h-[3px] w-32 bg-gradient-to-l from-blue-400/50 to-transparent blur-sm"
        initial={{ x: '-100vw' }}
        animate={{ 
          x: ['0%', '30%', '60%', '85%'],
        }}
        transition={{
          duration: 2,
          times: [0, 0.3, 0.7, 1],
          ease: 'easeOut',
        }}
        exit={{
          x: '100%',
          opacity: 0,
          transition: {
            duration: 0.3,
            ease: 'easeIn'
          }
        }}
      />
    </div>
  );
}