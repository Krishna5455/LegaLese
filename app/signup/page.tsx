import Link from "next/link";

import { AuthForm } from "@/components/auth/AuthForm";
import { signUp } from "@/lib/actions/auth";

export default function SignUpPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-border bg-surface/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
          <Link
            href="/"
            className="text-xl font-semibold tracking-tight text-foreground"
          >
            LegaLese
          </Link>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-6 py-16">
        <AuthForm
          title="Create account"
          description="Start reviewing contracts with LegaLese."
          submitLabel="Sign up"
          action={signUp}
          passwordAutoComplete="new-password"
          footer={
            <p className="text-sm text-muted">
              Already have an account?{" "}
              <Link href="/login" className="font-medium text-accent">
                Sign in
              </Link>
            </p>
          }
        />
      </main>
    </div>
  );
}
