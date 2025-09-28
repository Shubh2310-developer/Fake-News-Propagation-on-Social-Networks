// frontend/src/components/layout/Footer.tsx

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
      <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
        <p className="text-center text-sm leading-loose text-slate-600 dark:text-slate-400">
          © {currentYear} GTDS Project. All Rights Reserved.
        </p>
        <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
          <a href="/privacy" className="hover:underline">Privacy Policy</a>
          <a href="/terms" className="hover:underline">Terms of Service</a>
        </div>
      </div>
    </footer>
  );
}