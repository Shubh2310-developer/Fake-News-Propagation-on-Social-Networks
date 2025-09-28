// frontend/src/components/layout/Header.tsx

"use client";

import { Menu, Sun, Moon } from 'lucide-react';
import { useUIStore } from '@/store';
import { Button } from '@/components/ui/button';

export function Header() {
  const { toggleSidebar, theme, toggleTheme } = useUIStore();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
      <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
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
          <h1 className="text-xl font-bold">GTDS Platform</h1>
        </div>

        {/* Right Section */}
        <div className="flex flex-1 items-center justify-end space-x-4">
          <Button variant="ghost" size="icon" onClick={toggleTheme}>
            {theme === 'dark' ? (
              <Sun className="h-[1.2rem] w-[1.2rem]" />
            ) : (
              <Moon className="h-[1.2rem] w-[1.2rem]" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>
          {/* <UserProfileDropdown /> */}
        </div>
      </div>
    </header>
  );
}