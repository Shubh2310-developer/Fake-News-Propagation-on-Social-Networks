// frontend/src/store/uiStore.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type NotificationType = 'success' | 'error' | 'info' | 'warning';

interface UIStore {
  // State
  theme: 'light' | 'dark';
  isSidebarOpen: boolean;
  notification: { message: string; type: NotificationType } | null;

  // Actions
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  setSidebarOpen: (isOpen: boolean) => void;
  showNotification: (message: string, type: NotificationType) => void;
  hideNotification: () => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'dark',
      isSidebarOpen: true,
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

      showNotification: (message, type) =>
        set({ notification: { message, type } }),

      hideNotification: () => set({ notification: null }),
    }),
    {
      name: 'ui-storage', // Key for localStorage
      partialize: (state) => ({
        theme: state.theme,
        isSidebarOpen: state.isSidebarOpen
      }), // Only persist theme and sidebar state
    }
  )
);