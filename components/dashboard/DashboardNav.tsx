import Link from "next/link";

import { SignOutButton } from "@/components/auth/SignOutButton";

type DashboardNavProps = {
  userEmail?: string | null;
  active?: "dashboard" | "create";
};

export function DashboardNav({ userEmail, active }: DashboardNavProps) {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent/15 border border-accent/30 text-accent font-bold text-xs group-hover:bg-accent group-hover:text-white transition-colors">
            §
          </div>
          <span className="text-base font-bold tracking-tight text-foreground">
            LegaLese
          </span>
        </Link>

        <nav className="flex items-center gap-4 text-xs">
          <Link
            href="/dashboard"
            className={
              active === "dashboard"
                ? "font-semibold text-accent"
                : "text-muted hover:text-foreground"
            }
          >
            Dashboard
          </Link>
          <Link
            href="/dashboard/create"
            className={
              active === "create"
                ? "font-semibold text-accent"
                : "text-muted hover:text-foreground"
            }
          >
            Create Document
          </Link>
          {userEmail ? (
            <span className="font-mono text-muted hidden sm:inline-block">
              {userEmail}
            </span>
          ) : null}
          <SignOutButton />
        </nav>
      </div>
    </header>
  );
}
