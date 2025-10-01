// frontend/src/components/layout/Header.tsx

"use client";

import { Menu } from 'lucide-react';
import { useUIStore } from '@/store';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export function Header() {
  const { toggleSidebar } = useUIStore();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md">
      <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0 px-6">
        {/* Left Section */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="md:hidden"
          >
            <Menu className="h-5 w-5" />
            <span className="sr-only">Toggle Sidebar</span>
          </Button>
          <Link href="/">
            <h1 className="text-xl font-bold text-gray-900 hover:text-blue-600 transition-colors cursor-pointer">
              GTDS Platform
            </h1>
          </Link>
        </div>

        {/* Right Section */}
        <div className="flex flex-1 items-center justify-end space-x-4">
          {/* Theme toggle removed as requested */}
        </div>
      </div>
    </header>
  );
}