// frontend/src/app/auth/error/page.tsx

"use client";

import React from 'react';
import { useSearchParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { AlertCircle, Home, ArrowLeft } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import Link from 'next/link';

export default function AuthErrorPage() {
  const searchParams = useSearchParams();
  const error = searchParams?.get('error');

  const getErrorDetails = (error: string | null) => {
    switch (error) {
      case 'Configuration':
        return {
          title: 'Configuration Error',
          description: 'There is a problem with the server configuration.',
          action: 'Please contact the administrator.',
        };
      case 'AccessDenied':
        return {
          title: 'Access Denied',
          description: 'You do not have permission to sign in.',
          action: 'Please check your account status or contact support.',
        };
      case 'Verification':
        return {
          title: 'Verification Failed',
          description: 'The verification token has expired or is invalid.',
          action: 'Please request a new verification email.',
        };
      case 'OAuthSignin':
      case 'OAuthCallback':
      case 'OAuthCreateAccount':
      case 'EmailCreateAccount':
      case 'Callback':
        return {
          title: 'Authentication Error',
          description: 'An error occurred during the authentication process.',
          action: 'Please try again or use a different sign-in method.',
        };
      case 'OAuthAccountNotLinked':
        return {
          title: 'Account Already Exists',
          description: 'An account with this email already exists with a different provider.',
          action: 'Please sign in using your original authentication method.',
        };
      case 'EmailSignin':
        return {
          title: 'Email Sign In Failed',
          description: 'Unable to send verification email.',
          action: 'Please check your email address and try again.',
        };
      case 'CredentialsSignin':
        return {
          title: 'Invalid Credentials',
          description: 'The email or password you entered is incorrect.',
          action: 'Please check your credentials and try again.',
        };
      case 'SessionRequired':
        return {
          title: 'Session Required',
          description: 'You must be signed in to access this page.',
          action: 'Please sign in to continue.',
        };
      default:
        return {
          title: 'Authentication Error',
          description: 'An unexpected error occurred during authentication.',
          action: 'Please try signing in again.',
        };
    }
  };

  const errorDetails = getErrorDetails(error);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-xl">
          <CardHeader className="space-y-1">
            <div className="flex justify-center mb-4">
              <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-full">
                <AlertCircle className="h-12 w-12 text-red-600 dark:text-red-400" />
              </div>
            </div>
            <CardTitle className="text-2xl font-bold text-center">
              {errorDetails.title}
            </CardTitle>
            <CardDescription className="text-center">
              Something went wrong during authentication
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>What happened?</AlertTitle>
              <AlertDescription>{errorDetails.description}</AlertDescription>
            </Alert>

            <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
              <p className="text-sm text-slate-700 dark:text-slate-300">
                <span className="font-semibold">Next steps:</span> {errorDetails.action}
              </p>
            </div>

            <div className="space-y-3">
              <Button asChild className="w-full">
                <Link href="/auth/signin">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Sign In
                </Link>
              </Button>

              <Button asChild variant="outline" className="w-full">
                <Link href="/">
                  <Home className="mr-2 h-4 w-4" />
                  Go to Homepage
                </Link>
              </Button>
            </div>

            {error && (
              <div className="pt-4 border-t">
                <p className="text-xs text-slate-500 dark:text-slate-400 text-center">
                  Error code: <code className="font-mono">{error}</code>
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
