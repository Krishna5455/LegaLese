import Link from "next/link";
import { redirect } from "next/navigation";

import { SignOutButton } from "@/components/auth/SignOutButton";
import { createClient } from "@/lib/supabase/server";
import { type Document, getDocumentLabel } from "@/types/database";

export const dynamic = "force-dynamic";

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { data: documents, error: documentsError } = await supabase
    .from("documents")
    .select("*")
    .order("created_at", { ascending: false });

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
          <SignOutButton />
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">
        <div className="mb-10">
          <p className="text-sm font-medium uppercase tracking-widest text-accent">
            Dashboard
          </p>
          <h1 className="mt-2 text-3xl font-semibold text-foreground">
            Welcome back
          </h1>
          <p className="mt-2 text-muted">Signed in as {user.email}</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-xl border border-border bg-surface p-6">
            <h2 className="text-lg font-semibold text-foreground">
              Uploaded contracts
            </h2>
            <p className="mt-2 text-sm text-muted">
              Contract upload will be available in a future release.
            </p>

            <div className="mt-6 rounded-lg border border-dashed border-border bg-background p-6">
              {documentsError ? (
                <p className="text-sm text-red-700">
                  Unable to load documents: {documentsError.message}
                </p>
              ) : documents && documents.length > 0 ? (
                <ul className="space-y-3">
                  {(documents as Document[]).map((document) => (
                    <li
                      key={document.id}
                      className="rounded-lg border border-border bg-surface px-4 py-3"
                    >
                      <p className="text-sm font-medium text-foreground">
                        {getDocumentLabel(document)}
                      </p>
                      <p className="mt-1 text-xs text-muted">
                        Added {formatDate(document.created_at)}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">
                  No contracts uploaded yet. Your documents will appear here once
                  upload is enabled.
                </p>
              )}
            </div>
          </section>

          <section className="rounded-xl border border-border bg-surface p-6">
            <h2 className="text-lg font-semibold text-foreground">
              Recent analyses
            </h2>
            <p className="mt-2 text-sm text-muted">
              Analysis results will appear here after you upload and analyze a
              contract.
            </p>

            <div className="mt-6 rounded-lg border border-dashed border-border bg-background p-6">
              <p className="text-sm text-muted">No analyses yet.</p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
