# Authentication System Documentation

## Overview

The Fake News Game Theory platform implements a mature, secure authentication system using **NextAuth.js** with support for multiple authentication providers and JWT-based session management.

## Features

- ✅ **Multiple Authentication Providers**
  - Google OAuth 2.0
  - GitHub OAuth
  - Email/Password (Credentials)

- ✅ **JWT-Based Sessions**
  - Decoupled architecture
  - Secure token encryption
  - Automatic session refresh

- ✅ **Database Persistence**
  - PostgreSQL via Prisma ORM
  - User accounts and OAuth links
  - Session management

- ✅ **Role-Based Access Control (RBAC)**
  - User roles: `researcher`, `admin`, `moderator`
  - Server-side authorization utilities
  - Client-side session management

- ✅ **Custom Authentication Pages**
  - Branded sign-in/sign-out pages
  - Error handling with user-friendly messages
  - Responsive design with Framer Motion animations

## Setup Instructions

### 1. Install Dependencies

```bash
cd frontend
npm install next-auth @auth/prisma-adapter @prisma/client bcryptjs
npm install -D @types/bcryptjs prisma
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env.local` and fill in your credentials:

```bash
cp .env.example .env.local
```

Required variables:

```env
# NextAuth.js Secret (Generate with: openssl rand -base64 32)
NEXTAUTH_SECRET=your-super-secret-key

# Application URL
NEXTAUTH_URL=http://localhost:3000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/fake_news_db

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### 3. Set Up OAuth Providers

#### Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URI: `http://localhost:3000/api/auth/callback/google`
6. Copy Client ID and Secret to `.env.local`

#### GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Create a new OAuth App
3. Set Authorization callback URL: `http://localhost:3000/api/auth/callback/github`
4. Copy Client ID and Secret to `.env.local`

### 4. Initialize Database

```bash
# Generate Prisma Client
npx prisma generate

# Run database migrations
npx prisma migrate dev --name init

# (Optional) Seed initial data
npx prisma db seed
```

### 5. Start Development Server

```bash
npm run dev
```

Navigate to `http://localhost:3000/auth/signin` to test authentication.

## Usage

### Client-Side Authentication

Use the `useSession` hook in client components:

```tsx
"use client";

import { useSession, signIn, signOut } from 'next-auth/react';

export default function MyComponent() {
  const { data: session, status } = useSession();

  if (status === 'loading') {
    return <div>Loading...</div>;
  }

  if (!session) {
    return (
      <button onClick={() => signIn()}>Sign In</button>
    );
  }

  return (
    <div>
      <p>Welcome, {session.user.email}!</p>
      <p>Role: {session.user.role}</p>
      <button onClick={() => signOut()}>Sign Out</button>
    </div>
  );
}
```

### Server-Side Authentication

Use the auth utilities in Server Components and API routes:

```tsx
// Server Component
import { getCurrentUser, requireAuth, requireRole } from '@/lib/auth';

export default async function ProtectedPage() {
  // Option 1: Get current user (returns null if not authenticated)
  const user = await getCurrentUser();

  // Option 2: Require authentication (throws error if not authenticated)
  const authenticatedUser = await requireAuth();

  // Option 3: Require specific role (throws error if insufficient permissions)
  const adminUser = await requireRole('admin');

  return (
    <div>
      <h1>Welcome, {user?.name}!</h1>
      <p>Your role: {user?.role}</p>
    </div>
  );
}
```

### API Route Protection

```typescript
// app/api/protected/route.ts
import { NextResponse } from 'next/server';
import { getCurrentUser } from '@/lib/auth';

export async function GET() {
  const user = await getCurrentUser();

  if (!user) {
    return NextResponse.json(
      { error: 'Unauthorized' },
      { status: 401 }
    );
  }

  return NextResponse.json({
    message: 'Protected data',
    user,
  });
}
```

### Middleware Protection

Create `middleware.ts` in the root:

```typescript
import { withAuth } from 'next-auth/middleware';

export default withAuth({
  pages: {
    signIn: '/auth/signin',
  },
});

export const config = {
  matcher: ['/dashboard/:path*', '/admin/:path*'],
};
```

## User Roles

The system supports three default roles:

- **`researcher`**: Default role for all users. Access to simulation tools and analysis features.
- **`moderator`**: Enhanced permissions for content moderation and user management.
- **`admin`**: Full system access including configuration and user role management.

### Changing User Roles

Use Prisma Studio or database CLI:

```bash
# Open Prisma Studio
npx prisma studio

# Or use SQL
psql $DATABASE_URL
UPDATE users SET role = 'admin' WHERE email = 'user@example.com';
```

## Security Best Practices

### Production Configuration

1. **Use Strong Secrets**
   ```bash
   # Generate a secure secret
   openssl rand -base64 32
   ```

2. **Enable HTTPS**
   - Set `NEXTAUTH_URL` to your HTTPS domain
   - Cookies will be marked as `secure`

3. **Configure CORS**
   - Limit OAuth redirect URIs to your domain
   - Set up proper Content Security Policy

4. **Environment Variables**
   - Never commit `.env.local` to version control
   - Use secure secret management in production (AWS Secrets Manager, Vault, etc.)

5. **Rate Limiting**
   - Implement rate limiting on auth routes
   - Monitor failed login attempts

### Password Security

When using credentials provider:

- Passwords are hashed using `bcryptjs` with 12 rounds
- Never log or expose password hashes
- Implement password complexity requirements
- Consider adding password reset flow

## Custom Pages

The authentication system uses custom branded pages:

- **Sign In**: `/auth/signin` - Multiple authentication options
- **Sign Out**: `/auth/signout` - Confirmation before signing out
- **Error**: `/auth/error` - User-friendly error messages with recovery options

All pages feature:
- Responsive design
- Dark mode support
- Smooth animations (Framer Motion)
- Accessibility compliance
- Loading states

## Troubleshooting

### Common Issues

**"Invalid OAuth credentials"**
- Verify `CLIENT_ID` and `CLIENT_SECRET` in `.env.local`
- Check OAuth callback URLs match exactly
- Ensure OAuth app is not in development mode (for production)

**"Database connection failed"**
- Verify `DATABASE_URL` is correct
- Ensure PostgreSQL is running
- Check database exists and migrations are applied

**"Session not persisting"**
- Clear cookies and try again
- Verify `NEXTAUTH_SECRET` is set
- Check browser allows cookies

**"CSRF token mismatch"**
- Clear browser cache
- Verify `NEXTAUTH_URL` matches your domain
- Check for reverse proxy configuration issues

### Debug Mode

Enable debug logging in development:

```typescript
// app/api/auth/[...nextauth]/route.ts
export const authOptions: NextAuthOptions = {
  // ...
  debug: true, // Already enabled in development
};
```

## Database Schema

The Prisma schema includes these auth-related models:

```prisma
model User {
  id            String    @id @default(cuid())
  name          String?
  email         String    @unique
  emailVerified DateTime?
  image         String?
  password      String?   // Hashed
  role          String    @default("researcher")
  accounts      Account[]
  sessions      Session[]
}

model Account {
  // OAuth account linking
}

model Session {
  // User sessions
}

model VerificationToken {
  // Email verification
}
```

## API Reference

### Server-Side Functions

```typescript
// Get current user
const user = await getCurrentUser();
// Returns: { id, email, name, image, role } | null

// Check if authenticated
const isAuth = await isAuthenticated();
// Returns: boolean

// Check role
const hasAdminRole = await hasRole('admin');
// Returns: boolean

// Require authentication (throws if not authenticated)
const user = await requireAuth();
// Returns: { id, email, name, image, role }

// Require role (throws if insufficient permissions)
const admin = await requireRole('admin');
// Returns: { id, email, name, image, role }
```

### Client-Side Hooks

```typescript
import { useSession, signIn, signOut } from 'next-auth/react';

// Get session
const { data: session, status } = useSession();

// Sign in
await signIn('google');
await signIn('github');
await signIn('credentials', { email, password });

// Sign out
await signOut({ callbackUrl: '/' });
```

## Contributing

When extending the auth system:

1. Follow the existing callback patterns
2. Update TypeScript types for session/JWT
3. Add tests for new auth flows
4. Update this documentation

## License

MIT License - See LICENSE file for details
