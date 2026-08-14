import Link from "next/link";
import { redirect } from "next/navigation";

import { SignOutButton } from "@/components/auth/SignOutButton";
import { ContractUpload } from "@/components/dashboard/ContractUpload";
import { DocumentList } from "@/components/dashboard/DocumentList";
import { createClient } from "@/lib/supabase/server";
import type { Document } from "@/types/database";

export const dynamic = "force-dynamic";

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

        <div className="grid gap-8 lg:grid-cols-2">
          <div className="space-y-8">
            <ContractUpload />
          </div>

          <div className="space-y-8">
            <section className="rounded-xl border border-border bg-surface p-6">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-foreground">
                    Uploaded contracts
                  </h2>
                  <p className="mt-1 text-sm text-muted">
                    Your stored agreements and legal documents.
                  </p>
                </div>
                {documents && documents.length > 0 ? (
                  <span className="rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-semibold text-accent">
                    {documents.length}
                  </span>
                ) : null}
              </div>

              <DocumentList
                documents={documents as Document[] | null}
                error={documentsError?.message}
              />
            </section>

            <section className="rounded-xl border border-border bg-surface p-6">
              <h2 className="text-lg font-semibold text-foreground">
                Recent analyses
              </h2>
              <p className="mt-2 text-sm text-muted">
                Analysis results will appear here after you analyze an uploaded
                contract in a future release.
              </p>

              <div className="mt-6 rounded-lg border border-dashed border-border bg-background p-6">
                <p className="text-sm text-muted">No analyses yet.</p>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
