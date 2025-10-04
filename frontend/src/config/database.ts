/**
 * Database Configuration
 * Configuration for database connections and settings
 * Note: Frontend typically doesn't connect directly to the database,
 * but this config is used for NextAuth and server-side operations
 */

export const DATABASE_CONFIG = {
  // Database connection
  url: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/fakenews_db',

  // Connection pool settings
  pool: {
    min: 2,
    max: 10,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
  },

  // NextAuth specific
  nextAuth: {
    adapter: 'prisma', // or 'postgres'
    sessionStrategy: 'jwt' as const, // or 'database'
    sessionMaxAge: 30 * 24 * 60 * 60, // 30 days
    sessionUpdateAge: 24 * 60 * 60, // 24 hours
  },

  // Query settings
  query: {
    timeout: 10000, // 10 seconds
    retries: 3,
  },

  // Feature flags
  features: {
    ssl: process.env.NODE_ENV === 'production',
    logging: process.env.NODE_ENV === 'development',
    migrations: true,
  },
} as const;

// Table names (for reference)
export const TABLES = {
  users: 'users',
  newsArticles: 'news_articles',
  classifications: 'classifications',
  socialNodes: 'social_nodes',
  socialEdges: 'social_edges',
  simulations: 'simulations',
  simulationResults: 'simulation_results',
  payoffs: 'payoffs',
  accounts: 'accounts', // NextAuth
  sessions: 'sessions', // NextAuth
  verificationTokens: 'verification_tokens', // NextAuth
} as const;

// User roles
export const USER_ROLES = {
  SPREADER: 'spreader',
  FACT_CHECKER: 'fact_checker',
  PLATFORM: 'platform',
  ADMIN: 'admin',
} as const;

// Article labels
export const ARTICLE_LABELS = {
  FAKE: 'fake',
  TRUE: 'true',
  UNKNOWN: 'unknown',
} as const;

// Connection type
export const CONNECTION_TYPES = {
  TRUST: 'trust',
  FOLLOW: 'follow',
  FRIEND: 'friend',
} as const;

export type UserRole = (typeof USER_ROLES)[keyof typeof USER_ROLES];
export type ArticleLabel = (typeof ARTICLE_LABELS)[keyof typeof ARTICLE_LABELS];
export type ConnectionType = (typeof CONNECTION_TYPES)[keyof typeof CONNECTION_TYPES];
