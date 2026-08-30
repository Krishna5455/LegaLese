import Link from "next/link";

import { AuthForm } from "@/components/auth/AuthForm";
import { signIn } from "@/lib/actions/auth";

type LoginPageProps = {
  searchParams: Promise<{
    error?: string;
    next?: string;
  }>;
};

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams;
  const callbackError =
    params.error === "auth_callback_error"
      ? "Authentication failed. Please try signing in again."
      : undefined;

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
          title="Sign in"
          description="Access your LegaLese dashboard."
          submitLabel="Sign in"
          action={signIn}
          initialError={callbackError}
          footer={
            <p className="text-sm text-muted">
              Don&apos;t have an account?{" "}
              <Link href="/signup" className="font-medium text-accent">
                Sign up
              </Link>
            </p>
          }
        >
          <input type="hidden" name="next" value={params.next ?? "/dashboard"} />
        </AuthForm>
      </main>
    </div>
  );
}
