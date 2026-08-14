import Link from "next/link";

export function Header() {
  return (
    <header className="border-b border-border bg-surface/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
        <Link
          href="/"
          className="text-xl font-semibold tracking-tight text-foreground"
        >
          LegaLese
        </Link>
        <nav aria-label="Main" className="flex items-center gap-4 text-sm">
          <Link href="/login" className="text-muted hover:text-foreground">
            Sign in
          </Link>
          <Link
            href="/signup"
            className="font-medium text-accent hover:text-accent-hover"
          >
            Sign up
          </Link>
        </nav>
      </div>
    </header>
  );
}
