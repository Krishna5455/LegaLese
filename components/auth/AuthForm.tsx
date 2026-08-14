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
    <div className="rounded-xl border border-border bg-surface p-8 shadow-sm">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground">{title}</h1>
        <p className="mt-2 text-sm text-muted">{description}</p>
      </div>

      <form action={formAction} className="space-y-5">
        {children}

        <div>
          <label htmlFor="email" className="mb-2 block text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            className="w-full rounded-lg border border-border bg-background px-4 py-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
          />
        </div>

        <div>
          <label htmlFor="password" className="mb-2 block text-sm font-medium">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete={passwordAutoComplete}
            required
            minLength={6}
            className="w-full rounded-lg border border-border bg-background px-4 py-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
          />
        </div>

        {state.error ? (
          <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {state.error}
          </p>
        ) : null}

        <Button type="submit" className="w-full" disabled={pending}>
          {pending ? "Please wait..." : submitLabel}
        </Button>
      </form>

      {footer ? <div className="mt-6">{footer}</div> : null}
    </div>
  );
}
