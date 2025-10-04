# Authentication System Installation Guide

This guide provides step-by-step instructions to install and configure the authentication system for the Fake News Game Theory platform.

## Prerequisites

- Node.js 18+ installed
- PostgreSQL database running
- npm or yarn package manager

## Installation Steps

### 1. Install Required Dependencies

Run the following command in the `frontend` directory:

```bash
npm install next-auth@^4.24.0 \
  @auth/prisma-adapter@^1.0.0 \
  @prisma/client@^5.6.0 \
  bcryptjs@^2.4.3

npm install -D \
  prisma@^5.6.0 \
  @types/bcryptjs@^2.4.6
```

Or with yarn:

```bash
yarn add next-auth @auth/prisma-adapter @prisma/client bcryptjs
yarn add -D prisma @types/bcryptjs
```

### 2. Create Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env.local
```

Generate a secure NextAuth secret:

```bash
openssl rand -base64 32
```

Update `.env.local` with your configuration:

```env
# NextAuth.js Configuration
NEXTAUTH_SECRET=<paste-generated-secret-here>
NEXTAUTH_URL=http://localhost:3000

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/fake_news_db?schema=public

# OAuth Providers (obtain from respective platforms)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### 3. Set Up OAuth Applications

#### Google OAuth Setup

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. Choose "Web application"
6. Add authorized JavaScript origins:
   - `http://localhost:3000`
   - (Add production URL when deploying)
7. Add authorized redirect URIs:
   - `http://localhost:3000/api/auth/callback/google`
   - (Add production callback URL when deploying)
8. Copy the Client ID and Client Secret to `.env.local`

#### GitHub OAuth Setup

1. Visit [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in application details:
   - **Application name**: Fake News Game Theory
   - **Homepage URL**: `http://localhost:3000`
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
4. Click "Register application"
5. Generate a new client secret
6. Copy the Client ID and Client Secret to `.env.local`

### 4. Initialize Prisma Database

Generate Prisma client and create database schema:

```bash
# Generate Prisma client
npx prisma generate

# Create database tables
npx prisma migrate dev --name init_auth

# (Optional) Open Prisma Studio to view data
npx prisma studio
```

### 5. Verify Installation

Start the development server:

```bash
npm run dev
```

Navigate to:
- Sign In Page: http://localhost:3000/auth/signin
- Test authentication with Google, GitHub, or credentials

### 6. Create First Admin User

After signing in for the first time, upgrade your account to admin:

```bash
# Using Prisma Studio (recommended)
npx prisma studio
# Navigate to Users table, find your user, change role to "admin"

# Or using psql
psql $DATABASE_URL
UPDATE users SET role = 'admin' WHERE email = 'your-email@example.com';
```

## File Structure

The authentication system adds the following files:

```
frontend/
├── prisma/
│   └── schema.prisma                          # Database schema with auth models
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── auth/
│   │   │       └── [...nextauth]/
│   │   │           └── route.ts               # NextAuth.js configuration
│   │   └── auth/
│   │       ├── signin/
│   │       │   └── page.tsx                   # Custom sign-in page
│   │       ├── signout/
│   │       │   └── page.tsx                   # Custom sign-out page
│   │       └── error/
│   │           └── page.tsx                   # Custom error page
│   ├── components/
│   │   └── providers/
│   │       └── SessionProvider.tsx            # NextAuth session wrapper
│   └── lib/
│       └── auth.ts                            # Server-side auth utilities
├── docs/
│   └── AUTHENTICATION.md                      # Full authentication docs
└── .env.local                                 # Environment configuration
```

## Testing Authentication

### Test OAuth Providers

1. Click "Continue with Google" or "Continue with GitHub"
2. Complete the OAuth flow
3. Verify you're redirected to the dashboard
4. Check your session: `await getCurrentUser()` in server components

### Test Credentials Provider

1. First, create a user with hashed password:

```typescript
// Create a script: scripts/create-user.ts
import bcrypt from 'bcryptjs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function createUser() {
  const hashedPassword = await bcrypt.hash('YourPassword123!', 12);

  await prisma.user.create({
    data: {
      email: 'test@example.com',
      name: 'Test User',
      password: hashedPassword,
      role: 'researcher',
      emailVerified: new Date(),
    },
  });
}

createUser();
```

2. Run the script: `npx ts-node scripts/create-user.ts`
3. Sign in with email and password
4. Verify authentication works

## Protecting Routes

### Protect Dashboard Routes

Create `middleware.ts` in the app root:

```typescript
export { default } from 'next-auth/middleware';

export const config = {
  matcher: ['/dashboard/:path*', '/admin/:path*'],
};
```

### Protect API Routes

```typescript
// app/api/protected/route.ts
import { NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';

export async function GET() {
  try {
    const user = await requireAuth();
    return NextResponse.json({ user });
  } catch {
    return NextResponse.json(
      { error: 'Unauthorized' },
      { status: 401 }
    );
  }
}
```

## Common Issues and Solutions

### Issue: "Cannot find module '@prisma/client'"

**Solution**: Run `npx prisma generate` to generate the Prisma client.

### Issue: OAuth callback fails

**Solution**:
- Verify callback URLs in OAuth provider settings
- Ensure `NEXTAUTH_URL` matches your development URL
- Check that OAuth credentials are correct

### Issue: "Invalid session token"

**Solution**:
- Clear browser cookies
- Restart development server
- Verify `NEXTAUTH_SECRET` is set correctly

### Issue: Database connection errors

**Solution**:
- Verify PostgreSQL is running
- Check `DATABASE_URL` format is correct
- Ensure database exists: `createdb fake_news_db`

### Issue: Credentials login fails

**Solution**:
- Verify user exists in database with hashed password
- Check email is verified (`emailVerified` is not null)
- Verify password is hashed with bcryptjs (not plain text)

## Production Deployment Checklist

- [ ] Generate new `NEXTAUTH_SECRET` for production
- [ ] Update `NEXTAUTH_URL` to production domain (with HTTPS)
- [ ] Update OAuth redirect URIs to production URLs
- [ ] Enable HTTPS-only cookies in production
- [ ] Set up database backups
- [ ] Configure rate limiting on auth endpoints
- [ ] Enable session logging and monitoring
- [ ] Set up email verification for new users
- [ ] Implement password reset flow
- [ ] Add 2FA (optional but recommended)

## Next Steps

1. **Customize Auth Pages**: Modify sign-in/sign-out pages to match your brand
2. **Add Email Verification**: Implement email verification for new accounts
3. **Password Reset**: Add password reset flow
4. **Role Management**: Create admin UI for managing user roles
5. **Audit Logging**: Log authentication events for security

## Support

For issues or questions:
- Check the [AUTHENTICATION.md](./docs/AUTHENTICATION.md) documentation
- Review [NextAuth.js documentation](https://next-auth.js.org/)
- Open an issue on the GitHub repository

## Security Notes

⚠️ **Important Security Reminders**:

1. Never commit `.env.local` or expose secrets
2. Use strong, unique secrets in production
3. Regularly update dependencies for security patches
4. Monitor failed login attempts
5. Implement rate limiting on auth endpoints
6. Use HTTPS in production
7. Regularly audit user permissions and roles

---

**Authentication system successfully installed! 🎉**

You can now sign in at http://localhost:3000/auth/signin
