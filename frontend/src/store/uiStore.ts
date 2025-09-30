// frontend/src/store/uiStore.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type NotificationType = 'success' | 'error' | 'info' | 'warning';

interface UIStore {
  // State
  theme: 'light' | 'dark';
  isSidebarOpen: boolean;
  isSidebarCollapsed: boolean;
  notification: { message: string; type: NotificationType } | null;

  // Actions
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  setSidebarOpen: (isOpen: boolean) => void;
  toggleSidebarCollapse: () => void;
  setSidebarCollapsed: (isCollapsed: boolean) => void;
  showNotification: (message: string, type: NotificationType) => void;
  hideNotification: () => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'dark',
      isSidebarOpen: false, // Mobile sidebar closed by default
      isSidebarCollapsed: false, // Desktop sidebar expanded by default
      notification: null,

      // Actions
      toggleTheme: () =>
        set((state) => ({
          theme: state.theme === 'dark' ? 'light' : 'dark'
        })),

      setTheme: (theme) => set({ theme }),

      toggleSidebar: () =>
        set((state) => ({
          isSidebarOpen: !state.isSidebarOpen
        })),

      setSidebarOpen: (isOpen) => set({ isSidebarOpen: isOpen }),

      toggleSidebarCollapse: () =>
        set((state) => ({
          isSidebarCollapsed: !state.isSidebarCollapsed
        })),

      setSidebarCollapsed: (isCollapsed) => set({ isSidebarCollapsed: isCollapsed }),

      showNotification: (message, type) =>
        set({ notification: { message, type } }),

      hideNotification: () => set({ notification: null }),
    }),
    {
      name: 'ui-storage', // Key for localStorage
      partialize: (state) => ({
        theme: state.theme,
        isSidebarCollapsed: state.isSidebarCollapsed, // Persist collapsed state
      }), // Only persist theme and sidebar collapsed state
    }
  )
);