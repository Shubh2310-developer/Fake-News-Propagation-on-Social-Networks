# Authentication Quick Start Guide

## ✅ Installation Complete!

The authentication system has been successfully implemented with:
- ✓ NextAuth.js v4.24.11
- ✓ Prisma ORM v5.22.0
- ✓ Google & GitHub OAuth
- ✓ Email/Password authentication
- ✓ JWT sessions
- ✓ Role-based access control

## 🚀 Quick Setup (3 Steps)

### Step 1: Configure Environment

```bash
# Run the automated setup script
cd frontend
./scripts/setup-auth.sh
```

Or manually:

```bash
# Copy environment file
cp .env.example .env.local

# Generate secret
openssl rand -base64 32

# Add to .env.local:
NEXTAUTH_SECRET=<generated-secret>
DATABASE_URL=postgresql://user:pass@localhost:5432/fake_news_db
```

### Step 2: Set Up OAuth (Optional but Recommended)

**Google OAuth:**
1. Visit: https://console.cloud.google.com/
2. Create OAuth credentials
3. Add callback: `http://localhost:3000/api/auth/callback/google`
4. Copy Client ID & Secret to `.env.local`

**GitHub OAuth:**
1. Visit: https://github.com/settings/developers
2. Create OAuth App
3. Add callback: `http://localhost:3000/api/auth/callback/github`
4. Copy Client ID & Secret to `.env.local`

### Step 3: Initialize Database

```bash
# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev --name init_auth

# (Optional) View database
npx prisma studio
```

## 🎯 Usage Examples

### Client Components

```tsx
"use client";
import { useSession, signIn, signOut } from 'next-auth/react';

export default function MyComponent() {
  const { data: session } = useSession();

  if (!session) {
    return <button onClick={() => signIn()}>Sign In</button>;
  }

  return (
    <div>
      <p>Welcome {session.user.email}</p>
      <p>Role: {session.user.role}</p>
      <button onClick={() => signOut()}>Sign Out</button>
    </div>
  );
}
```

### Server Components

```tsx
import { getCurrentUser, requireAuth, requireRole } from '@/lib/auth';

// Get user (returns null if not authenticated)
const user = await getCurrentUser();

// Require authentication (throws if not authenticated)
const user = await requireAuth();

// Require specific role (throws if insufficient permissions)
const admin = await requireRole('admin');
```

### API Routes

```typescript
import { NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';

export async function GET() {
  try {
    const user = await requireAuth();
    return NextResponse.json({ user });
  } catch {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
}
```

## 🔐 Available Routes

- **Sign In**: http://localhost:3000/auth/signin
- **Sign Out**: http://localhost:3000/auth/signout
- **Error**: http://localhost:3000/auth/error

## 👥 User Roles

- `researcher` - Default role, access to simulation tools
- `moderator` - Enhanced permissions
- `admin` - Full system access

**Change user role:**
```bash
npx prisma studio
# Navigate to Users → Select user → Change 'role' field
```

## 📝 Environment Variables

```env
# Required
NEXTAUTH_SECRET=<32-character-secret>
NEXTAUTH_URL=http://localhost:3000
DATABASE_URL=postgresql://...

# OAuth (Optional)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
```

## 🔧 Common Commands

```bash
# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev

# Open database viewer
npx prisma studio

# Reset database (⚠️ deletes all data)
npx prisma migrate reset

# Start development server
npm run dev
```

## 📚 Documentation

- **Full Setup Guide**: [INSTALL_AUTH.md](./INSTALL_AUTH.md)
- **Comprehensive Docs**: [docs/AUTHENTICATION.md](./docs/AUTHENTICATION.md)
- **NextAuth Docs**: https://next-auth.js.org/
- **Prisma Docs**: https://www.prisma.io/docs

## 🐛 Troubleshooting

**Module not found errors?**
```bash
npm install next-auth @auth/prisma-adapter @prisma/client bcryptjs
npx prisma generate
```

**Database connection failed?**
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists

**OAuth errors?**
- Verify callback URLs match exactly
- Check credentials in .env.local
- Ensure OAuth app is active

**Session not persisting?**
- Clear browser cookies
- Verify NEXTAUTH_SECRET is set
- Restart development server

## ✨ Features

✅ Multiple auth providers (Google, GitHub, Email/Password)
✅ JWT-based sessions (30-day expiry)
✅ Secure password hashing (bcryptjs)
✅ Role-based access control
✅ Custom branded UI pages
✅ Dark mode support
✅ Responsive design
✅ TypeScript support
✅ Database persistence
✅ Automatic session refresh

## 🎉 You're All Set!

Start the dev server and visit the sign-in page:

```bash
npm run dev
```

Navigate to: **http://localhost:3000/auth/signin**

---

**Need help?** Check the detailed documentation:
- [INSTALL_AUTH.md](./INSTALL_AUTH.md) - Step-by-step installation
- [docs/AUTHENTICATION.md](./docs/AUTHENTICATION.md) - Complete API reference
