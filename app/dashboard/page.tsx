import Link from "next/link";
import { redirect } from "next/navigation";

import { SignOutButton } from "@/components/auth/SignOutButton";
import { ContractUpload } from "@/components/dashboard/ContractUpload";
import { DocumentList } from "@/components/dashboard/DocumentList";
import { createClient } from "@/lib/supabase/server";
import type {
  AnalysisRow,
  ClauseRow,
  DetailedAnalysis,
  FindingRow,
  FindingWithClause,
  KeyTermRow,
  KeyTermWithClause,
  ObligationRow,
  ObligationWithClause,
} from "@/types/analysis";
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

  // Fetch user documents
  const { data: documents, error: documentsError } = await supabase
    .from("documents")
    .select("*")
    .order("created_at", { ascending: false });

  // Fetch analyses and related child records
  const analysesMap: Record<string, DetailedAnalysis> = {};

  if (documents && documents.length > 0) {
    const documentIds = documents.map((d: Document) => d.id);

    const [
      { data: analyses },
      { data: clauses },
      { data: findings },
      { data: keyTerms },
      { data: obligations },
    ] = await Promise.all([
      supabase
        .from("analyses")
        .select("*")
        .in("document_id", documentIds)
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      supabase
        .from("clauses")
        .select("*")
        .in("document_id", documentIds)
        .order("created_at", { ascending: true }),
      supabase
        .from("findings")
        .select("*")
        .in("document_id", documentIds)
        .order("created_at", { ascending: true }),
      supabase
        .from("key_terms")
        .select("*")
        .in("document_id", documentIds)
        .order("created_at", { ascending: true }),
      supabase
        .from("obligations")
        .select("*")
        .in("document_id", documentIds)
        .order("created_at", { ascending: true }),
    ]);

    if (analyses && analyses.length > 0) {
      const clauseMap = new Map<string, ClauseRow>();
      ((clauses as ClauseRow[]) ?? []).forEach((c) => clauseMap.set(c.id, c));

      const seenDocIds = new Set<string>();
      for (const analysis of analyses as AnalysisRow[]) {
        if (seenDocIds.has(analysis.document_id)) continue;
        seenDocIds.add(analysis.document_id);

        const docClauses = ((clauses as ClauseRow[]) ?? []).filter(
          (c) => c.document_id === analysis.document_id,
        );

        const docFindings: FindingWithClause[] = (
          (findings as FindingRow[]) ?? []
        )
          .filter((f) => f.document_id === analysis.document_id)
          .map((f) => ({
            ...f,
            clause: f.clause_id ? clauseMap.get(f.clause_id) ?? null : null,
          }));

        const docKeyTerms: KeyTermWithClause[] = (
          (keyTerms as KeyTermRow[]) ?? []
        )
          .filter((kt) => kt.document_id === analysis.document_id)
          .map((kt) => ({
            ...kt,
            clause: kt.source_clause_id
              ? clauseMap.get(kt.source_clause_id) ?? null
              : null,
          }));

        const docObligations: ObligationWithClause[] = (
          (obligations as ObligationRow[]) ?? []
        )
          .filter((o) => o.document_id === analysis.document_id)
          .map((o) => ({
            ...o,
            clause: o.source_clause_id
              ? clauseMap.get(o.source_clause_id) ?? null
              : null,
          }));

        analysesMap[analysis.document_id] = {
          ...analysis,
          clauses: docClauses,
          findings: docFindings,
          key_terms: docKeyTerms,
          obligations: docObligations,
        };
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
