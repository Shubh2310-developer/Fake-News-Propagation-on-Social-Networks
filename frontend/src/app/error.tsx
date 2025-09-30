'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="flex min-h-[calc(100vh-200px)] flex-col items-center justify-center px-4">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-red-600 dark:text-red-400">
          Something went wrong!
        </h2>
        <p className="text-gray-600 dark:text-gray-400 max-w-md">
          {error.message || 'An unexpected error occurred. Please try again.'}
        </p>
        <div className="flex gap-4 justify-center">
          <Button
            onClick={reset}
            variant="default"
          >
            Try again
          </Button>
          <Button
            onClick={() => window.location.href = '/'}
            variant="outline"
          >
            Go home
          </Button>
        </div>
        {error.digest && (
          <p className="text-sm text-gray-500 dark:text-gray-600 mt-4">
            Error ID: {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}