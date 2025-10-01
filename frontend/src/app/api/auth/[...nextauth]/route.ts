// frontend/src/app/api/auth/[...nextauth]/route.ts

import NextAuth, { NextAuthOptions, Session, User } from 'next-auth';
import { JWT } from 'next-auth/jwt';
import GoogleProvider from 'next-auth/providers/google';
import GitHubProvider from 'next-auth/providers/github';
import CredentialsProvider from 'next-auth/providers/credentials';
import { PrismaAdapter } from '@auth/prisma-adapter';
import bcrypt from 'bcryptjs';
import { prisma } from '@/lib/prisma';

/**
 * Extended User type with custom fields
 */
interface ExtendedUser extends User {
  id: string;
  role?: string;
  emailVerified?: Date | null;
}

/**
 * Extended JWT type with custom claims
 */
interface ExtendedJWT extends JWT {
  id: string;
  role: string;
  email: string;
  name?: string | null;
}

/**
 * Extended Session type with custom fields
 */
interface ExtendedSession extends Session {
  user: {
    id: string;
    role: string;
    email: string;
    name?: string | null;
    image?: string | null;
  };
}

/**
 * NextAuth Configuration
 *
 * This configuration implements a mature, secure authentication system with:
 * - Multiple OAuth providers (Google, GitHub)
 * - Credentials-based authentication (email/password)
 * - JWT-based session management
 * - Database persistence via Prisma
 * - Custom callbacks for session enrichment
 * - Role-based access control (RBAC) support
 */
export const authOptions: NextAuthOptions = {
  // ================================================================
  // Adapter Configuration
  // ================================================================
  adapter: PrismaAdapter(prisma) as any,

  // ================================================================
  // Session Strategy
  // ================================================================
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
    updateAge: 24 * 60 * 60, // 24 hours
  },

  // ================================================================
  // Authentication Providers
  // ================================================================
  providers: [
    // Google OAuth Provider
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      authorization: {
        params: {
          prompt: 'consent',
          access_type: 'offline',
          response_type: 'code',
        },
      },
      profile(profile) {
        return {
          id: profile.sub,
          name: profile.name,
          email: profile.email,
          image: profile.picture,
          role: 'researcher', // Default role for OAuth users
          emailVerified: profile.email_verified ? new Date() : null,
        };
      },
    }),

    // GitHub OAuth Provider
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
      profile(profile) {
        return {
          id: profile.id.toString(),
          name: profile.name || profile.login,
          email: profile.email,
          image: profile.avatar_url,
          role: 'researcher', // Default role for OAuth users
          emailVerified: profile.email ? new Date() : null,
        };
      },
    }),

    // Credentials Provider (Email/Password)
    CredentialsProvider({
      id: 'credentials',
      name: 'Email and Password',
      credentials: {
        email: {
          label: 'Email',
          type: 'email',
          placeholder: 'user@example.com',
        },
        password: {
          label: 'Password',
          type: 'password',
        },
      },
      async authorize(credentials): Promise<ExtendedUser | null> {
        if (!credentials?.email || !credentials?.password) {
          throw new Error('Email and password are required');
        }

        try {
          // Find user in database
          const user = await prisma.user.findUnique({
            where: { email: credentials.email },
            select: {
              id: true,
              email: true,
              name: true,
              image: true,
              password: true,
              role: true,
              emailVerified: true,
            },
          });

          if (!user || !user.password) {
            throw new Error('Invalid email or password');
          }

          // Verify password
          const isPasswordValid = await bcrypt.compare(
            credentials.password,
            user.password
          );

          if (!isPasswordValid) {
            throw new Error('Invalid email or password');
          }

          // Check if email is verified (optional, depending on your requirements)
          if (!user.emailVerified) {
            throw new Error('Please verify your email before signing in');
          }

          // Return user object (password excluded)
          return {
            id: user.id,
            email: user.email,
            name: user.name,
            image: user.image,
            role: user.role || 'researcher',
            emailVerified: user.emailVerified,
          };
        } catch (error) {
          console.error('Authentication error:', error);
          throw error;
        }
      },
    }),
  ],

  // ================================================================
  // Callbacks
  // ================================================================
  callbacks: {
    /**
     * JWT Callback
     *
     * Augments the JWT with custom user data immediately after sign-in.
     * This callback runs whenever a JWT is created (sign in) or updated.
     */
    async jwt({ token, user, account, trigger }): Promise<ExtendedJWT> {
      // Initial sign in
      if (user) {
        const extendedUser = user as ExtendedUser;

        // Add custom claims to token
        token.id = extendedUser.id;
        token.email = extendedUser.email;
        token.name = extendedUser.name;
        token.role = extendedUser.role || 'researcher';

        // If signing in via OAuth, ensure user has a role in database
        if (account?.provider !== 'credentials') {
          try {
            const dbUser = await prisma.user.findUnique({
              where: { id: extendedUser.id },
              select: { role: true },
            });

            if (dbUser) {
              token.role = dbUser.role || 'researcher';
            }
          } catch (error) {
            console.error('Error fetching user role:', error);
          }
        }
      }

      // Handle token updates (e.g., when user updates profile)
      if (trigger === 'update' && token.id) {
        try {
          const updatedUser = await prisma.user.findUnique({
            where: { id: token.id as string },
            select: {
              name: true,
              email: true,
              image: true,
              role: true,
            },
          });

          if (updatedUser) {
            token.name = updatedUser.name;
            token.email = updatedUser.email;
            token.image = updatedUser.image;
            token.role = updatedUser.role || 'researcher';
          }
        } catch (error) {
          console.error('Error updating token:', error);
        }
      }

      return token as ExtendedJWT;
    },

    /**
     * Session Callback
     *
     * Transfers custom data from JWT to the client-side session object.
     * This makes user information available throughout the frontend application.
     */
    async session({ session, token }): Promise<ExtendedSession> {
      const extendedToken = token as ExtendedJWT;

      // Enrich session with user data from token
      return {
        ...session,
        user: {
          id: extendedToken.id,
          email: extendedToken.email,
          name: extendedToken.name || null,
          image: extendedToken.image || null,
          role: extendedToken.role,
        },
      } as ExtendedSession;
    },

    /**
     * Sign In Callback
     *
     * Controls whether a user is allowed to sign in.
     * Useful for implementing custom access control logic.
     */
    async signIn({ user, account, profile }): Promise<boolean> {
      try {
        // For OAuth providers, ensure user email is verified
        if (account?.provider === 'google' || account?.provider === 'github') {
          // Google and GitHub verify emails, so we trust them
          return true;
        }

        // For credentials provider, verification is handled in authorize()
        return true;
      } catch (error) {
        console.error('Sign in error:', error);
        return false;
      }
    },

    /**
     * Redirect Callback
     *
     * Controls where users are redirected after sign in/out.
     */
    async redirect({ url, baseUrl }): Promise<string> {
      // Allows relative callback URLs
      if (url.startsWith('/')) return `${baseUrl}${url}`;

      // Allows callback URLs on the same origin
      else if (new URL(url).origin === baseUrl) return url;

      // Default redirect to dashboard
      return `${baseUrl}/dashboard`;
    },
  },

  // ================================================================
  // Custom Pages
  // ================================================================
  pages: {
    signIn: '/auth/signin',
    signOut: '/auth/signout',
    error: '/auth/error',
    verifyRequest: '/auth/verify-request',
    newUser: '/auth/new-user',
  },

  // ================================================================
  // Events
  // ================================================================
  events: {
    async signIn({ user, account, isNewUser }) {
      console.log(`User signed in: ${user.email} via ${account?.provider}`);

      if (isNewUser) {
        console.log(`New user registered: ${user.email}`);
        // Here you could send welcome email, analytics event, etc.
      }
    },
    async signOut({ token }) {
      console.log(`User signed out: ${token?.email}`);
    },
  },

  // ================================================================
  // Security Configuration
  // ================================================================
  secret: process.env.NEXTAUTH_SECRET,

  // Enable debug logs in development
  debug: process.env.NODE_ENV === 'development',

  // JWT configuration
  jwt: {
    secret: process.env.NEXTAUTH_SECRET,
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },

  // Cookies configuration
  cookies: {
    sessionToken: {
      name: `${process.env.NODE_ENV === 'production' ? '__Secure-' : ''}next-auth.session-token`,
      options: {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: process.env.NODE_ENV === 'production',
      },
    },
  },
};

// ================================================================
// Export NextAuth Handler
// ================================================================
const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
