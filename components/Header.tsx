import Link from "next/link";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        {/* Brand logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent/15 border border-accent/30 text-accent font-bold text-xs group-hover:bg-accent group-hover:text-white transition-colors">
            §
          </div>
          <span className="text-base font-bold tracking-tight text-foreground">
            LegaLese
          </span>
          <span className="rounded bg-accent/10 border border-accent/20 px-1.5 py-0.2 text-[10px] font-mono text-accent">
            AI 3.6
          </span>
        </Link>

        {/* Navigation links */}
        <nav aria-label="Main" className="flex items-center gap-4 text-xs font-medium">
          <Link
            href="/login"
            className="text-muted hover:text-foreground transition-colors px-3 py-1.5 rounded-md hover:bg-surface"
          >
            Sign in
          </Link>
          <Link
            href="/signup"
            className="rounded-lg bg-accent px-3.5 py-1.5 font-semibold text-white hover:bg-accent-hover transition-all shadow-xs"
          >
            Get Started
          </Link>
        </nav>
      </div>
    </header>
  );
}
