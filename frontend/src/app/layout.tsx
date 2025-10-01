// /home/ghost/fake-news-game-theory/frontend/src/app/layout.tsx
import type { Metadata, Viewport } from 'next';
import './globals.css';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
};

// Metadata configuration
export const metadata: Metadata = {
  title: {
    default: 'Fake News Game Theory | Misinformation Analysis Platform',
    template: '%s | Fake News Game Theory',
  },
  description:
    'An innovative research platform integrating game theory, machine learning, and network analysis to predict fake news propagation and inform policy decisions.',
  keywords: [
    'fake news',
    'game theory',
    'machine learning',
    'network analysis',
    'misinformation',
    'social networks',
    'nash equilibrium',
    'content moderation',
    'fact checking',
    'information propagation',
  ],
  authors: [{ name: 'Fake News Game Theory Research Team' }],
  creator: 'Fake News Game Theory Research Team',
  publisher: 'Fake News Game Theory Research Team',
  metadataBase: new URL(process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000'),

  // Open Graph metadata for social sharing
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000',
    siteName: 'Fake News Game Theory',
    title: 'Fake News Game Theory | Misinformation Analysis Platform',
    description:
      'An innovative research platform integrating game theory, machine learning, and network analysis to predict fake news propagation and inform policy decisions.',
    images: [
      {
        url: '/images/network-visualization.png',
        width: 1200,
        height: 630,
        alt: 'Fake News Game Theory Platform',
      },
    ],
  },

  // Twitter Card metadata
  twitter: {
    card: 'summary_large_image',
    title: 'Fake News Game Theory | Misinformation Analysis Platform',
    description:
      'An innovative research platform integrating game theory, machine learning, and network analysis to predict fake news propagation.',
    images: ['/images/network-visualization.png'],
    creator: '@fakenewsgametheory',
  },

  // Verification and indexing
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },

  manifest: '/icons/manifest.json',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen bg-white text-gray-900 antialiased font-sans" style={{ fontFamily: 'Inter, sans-serif' }}>
        {/* Skip to main content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-4 focus:left-4 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          Skip to main content
        </a>

        {/* Global header */}
        <Header />

        {/* Main content */}
        <main id="main-content" className="min-h-[calc(100vh-200px)]">
          {children}
        </main>

        {/* Global footer */}
        <Footer />
      </body>
    </html>
  );
}