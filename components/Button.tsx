import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary" | "outline" | "ghost";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  children: ReactNode;
};

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-white shadow-xs hover:bg-accent-hover focus-visible:ring-accent/40 active:scale-[0.99]",
  secondary:
    "border border-border bg-surface text-foreground hover:bg-surface-hover hover:border-border-strong focus-visible:ring-accent/30 active:scale-[0.99]",
  outline:
    "border border-border/80 bg-transparent text-muted hover:border-border-strong hover:text-foreground active:scale-[0.99]",
  ghost:
    "bg-transparent text-muted hover:bg-surface-hover hover:text-foreground active:scale-[0.99]",
};

export function Button({
  variant = "primary",
  className = "",
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold tracking-wide transition-all focus-visible:outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-50 ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

type ButtonLinkProps = {
  href: string;
  variant?: ButtonVariant;
  className?: string;
  children: ReactNode;
};

export function ButtonLink({
  href,
  variant = "primary",
  className = "",
  children,
}: ButtonLinkProps) {
  return (
    <Link
      href={href}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold tracking-wide transition-all focus-visible:outline-none focus-visible:ring-2 ${variantStyles[variant]} ${className}`}
    >
      {children}
    </Link>
  );
}
