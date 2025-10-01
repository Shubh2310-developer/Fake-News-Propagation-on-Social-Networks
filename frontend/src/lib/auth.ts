// frontend/src/lib/auth.ts

import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

/**
 * Server-side utility to get the current session
 * Use this in Server Components, API routes, and Server Actions
 */
export async function getCurrentUser() {
  const session = await getServerSession(authOptions);
  return session?.user;
}

/**
 * Server-side utility to check if user is authenticated
 */
export async function isAuthenticated(): Promise<boolean> {
  const session = await getServerSession(authOptions);
  return !!session?.user;
}

/**
 * Server-side utility to check if user has a specific role
 */
export async function hasRole(role: string | string[]): Promise<boolean> {
  const user = await getCurrentUser();

  if (!user) return false;

  const roles = Array.isArray(role) ? role : [role];
  return roles.includes(user.role);
}

/**
 * Server-side utility to require authentication
 * Throws an error if user is not authenticated
 */
export async function requireAuth() {
  const user = await getCurrentUser();

  if (!user) {
    throw new Error('Unauthorized');
  }

  return user;
}

/**
 * Server-side utility to require specific role
 * Throws an error if user doesn't have the required role
 */
export async function requireRole(role: string | string[]) {
  const user = await requireAuth();
  const roles = Array.isArray(role) ? role : [role];

  if (!roles.includes(user.role)) {
    throw new Error('Forbidden: Insufficient permissions');
  }

  return user;
}
