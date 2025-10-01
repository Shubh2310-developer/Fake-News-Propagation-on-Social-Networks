// frontend/src/components/providers/SessionProvider.tsx

"use client";

import { SessionProvider as NextAuthSessionProvider } from 'next-auth/react';
import type { Session } from 'next-auth';

interface SessionProviderProps {
  children: React.ReactNode;
  session?: Session | null;
}

/**
 * SessionProvider wrapper for NextAuth.js
 *
 * This component wraps the application and provides session context
 * to all child components, enabling client-side session management.
 */
export function SessionProvider({ children, session }: SessionProviderProps) {
  return (
    <NextAuthSessionProvider session={session} refetchInterval={5 * 60}>
      {children}
    </NextAuthSessionProvider>
  );
}
