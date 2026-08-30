"use client";

import { useActionState } from "react";

import { Button } from "@/components/Button";
import type { AuthActionState } from "@/lib/actions/auth";

type AuthFormProps = {
  title: string;
  description: string;
  submitLabel: string;
  action: (
    prevState: AuthActionState,
    formData: FormData,
  ) => Promise<AuthActionState>;
  initialError?: string;
  footer?: React.ReactNode;
  children?: React.ReactNode;
  passwordAutoComplete?: "current-password" | "new-password";
};

export function AuthForm({
  title,
  description,
  submitLabel,
  action,
  initialError,
  footer,
  children,
  passwordAutoComplete = "current-password",
}: AuthFormProps) {
  const [state, formAction, pending] = useActionState(action, {
    error: initialError,
  });

  return (
    <div className="rounded-xl border border-border bg-surface p-8 shadow-sm max-w-md w-full mx-auto">
      <div className="mb-6 text-center sm:text-left">
        <h1 className="text-2xl font-bold text-foreground">{title}</h1>
        <p className="mt-2 text-xs text-muted leading-relaxed">{description}</p>
      </div>

      <form action={formAction} className="space-y-5">
        {children}

        <div>
          <label
            htmlFor="email"
            className="mb-2 block text-xs font-semibold text-foreground"
          >
            Email address
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            placeholder="you@company.com"
            className="w-full rounded-lg border border-border bg-surface-inset px-3.5 py-2.5 text-xs text-foreground placeholder:text-subtle outline-none focus:border-accent focus:ring-1 focus:ring-accent/40 transition-colors"
          />
        </div>

        <div>
          <label
            htmlFor="password"
            className="mb-2 block text-xs font-semibold text-foreground"
          >
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete={passwordAutoComplete}
            required
            minLength={6}
            placeholder="••••••••••••"
            className="w-full rounded-lg border border-border bg-surface-inset px-3.5 py-2.5 text-xs text-foreground placeholder:text-subtle outline-none focus:border-accent focus:ring-1 focus:ring-accent/40 transition-colors"
          />
        </div>

        {state.error ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs font-medium text-red-500 dark:text-red-400 leading-relaxed">
            {state.error}
          </div>
        ) : null}

        {state.message ? (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-xs font-medium text-emerald-600 dark:text-emerald-400 leading-relaxed">
            {state.message}
          </div>
        ) : null}

        <Button
          type="submit"
          variant="primary"
          className="w-full py-3 text-xs"
          disabled={pending}
        >
          {pending ? "Authenticating..." : submitLabel}
        </Button>
      </form>

      {footer ? (
        <div className="mt-6 text-center text-xs text-muted">{footer}</div>
      ) : null}
    </div>
  );
}
