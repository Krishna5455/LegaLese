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
        <nav aria-label="Main">
          <span className="text-sm text-muted">Legal contract analysis</span>
        </nav>
      </div>
    </header>
  );
}
