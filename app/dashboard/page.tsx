import Link from "next/link";
import { redirect } from "next/navigation";

import { SignOutButton } from "@/components/auth/SignOutButton";
import { ContractUpload } from "@/components/dashboard/ContractUpload";
import { DocumentList } from "@/components/dashboard/DocumentList";
import { createClient } from "@/lib/supabase/server";
import type { AnalysisWithDetails } from "@/types/analysis";
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

  // Fetch documents
  const { data: documents, error: documentsError } = await supabase
    .from("documents")
    .select("*")
    .order("created_at", { ascending: false });

  // Fetch the most recent complete analysis for each document.
  // We query all complete analyses for the user and build a map keyed by document_id.
  // Using a single query is more efficient than one query per document.
  let analysesMap: Record<string, AnalysisWithDetails> = {};

  if (documents && documents.length > 0) {
    const documentIds = documents.map((d: Document) => d.id);

    const [
      { data: analyses },
      { data: findings },
      { data: keyTerms },
      { data: obligations },
      { data: questions },
    ] = await Promise.all([
      supabase
        .from("analyses")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .eq("status", "complete")
        .order("created_at", { ascending: false }),
      supabase
        .from("findings")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .order("sort_order", { ascending: true }),
      supabase
        .from("key_terms")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .order("sort_order", { ascending: true }),
      supabase
        .from("obligations")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .order("sort_order", { ascending: true }),
      supabase
        .from("questions")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .order("sort_order", { ascending: true }),
    ]);

    if (analyses && analyses.length > 0) {
      // Keep only the most recent complete analysis per document
      // (analyses are ordered newest-first so the first per document_id wins)
      const seenDocIds = new Set<string>();
      for (const analysis of analyses) {
        if (seenDocIds.has(analysis.document_id)) continue;
        seenDocIds.add(analysis.document_id);

        analysesMap[analysis.document_id] = {
          ...analysis,
          findings: (findings ?? []).filter(
            (f) => f.analysis_id === analysis.id,
          ),
          key_terms: (keyTerms ?? []).filter(
            (kt) => kt.analysis_id === analysis.id,
          ),
          obligations: (obligations ?? []).filter(
            (o) => o.analysis_id === analysis.id,
          ),
          questions: (questions ?? []).filter(
            (q) => q.analysis_id === analysis.id,
          ),
        } as AnalysisWithDetails;
      }
    }
  }

  const analyzedCount = Object.keys(analysesMap).length;

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
                <div className="flex items-center gap-2">
                  {documents && documents.length > 0 ? (
                    <span className="rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-semibold text-accent">
                      {documents.length}
                    </span>
                  ) : null}
                  {analyzedCount > 0 && (
                    <span className="rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-semibold text-green-700">
                      {analyzedCount} analyzed
                    </span>
                  )}
                </div>
              </div>

              <DocumentList
                documents={documents as Document[] | null}
                error={documentsError?.message}
                analysesMap={analysesMap}
              />
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
