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

  const totalCount = documents?.length ?? 0;
  const analyzedCount = Object.keys(analysesMap).length;
  const highRiskCount = Object.values(analysesMap).filter(
    (a) => a.risk_score === 3,
  ).length;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
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
          <div className="flex items-center gap-4">
            <span className="text-xs font-mono text-muted hidden sm:inline-block">
              {user.email}
            </span>
            <SignOutButton />
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8 space-y-8">
        {/* Page Title Header */}
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              Contract Dashboard
            </h1>
            <p className="text-xs text-muted">
              Manage your legal agreements, extractions, and AI analyses.
            </p>
          </div>
        </div>

        {/* 1. METRICS TOP BANNER */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="rounded-xl border border-border bg-surface p-4">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider">
              Total Contracts
            </p>
            <p className="mt-2 text-2xl font-bold text-foreground">
              {totalCount}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface p-4">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider">
              Analyzed
            </p>
            <p className="mt-2 text-2xl font-bold text-accent">
              {analyzedCount}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface p-4">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider">
              High Risk
            </p>
            <p className="mt-2 text-2xl font-bold text-red-400">
              {highRiskCount}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface p-4">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider">
              Engine Status
            </p>
            <div className="mt-2 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs font-mono text-foreground font-semibold">
                Gemini 3.6 Ready
              </span>
            </div>
          </div>
        </div>

        {/* 2. COMPACT CONTRACT UPLOAD */}
        <section className="space-y-3">
          <ContractUpload />
        </section>

        {/* 3. CONTRACT TABLE */}
        <section className="rounded-xl border border-border bg-surface p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-border/60 pb-4">
            <div>
              <h2 className="text-base font-bold text-foreground">
                Your Contracts
              </h2>
              <p className="text-xs text-muted">
                View stored agreements and access deep-dive contract reviews.
              </p>
            </div>
            {documents && documents.length > 0 && (
              <span className="rounded-full bg-accent/10 border border-accent/20 px-2.5 py-0.5 text-xs font-mono font-semibold text-accent">
                {documents.length} Total
              </span>
            )}
          </div>

          <DocumentList
            documents={documents as Document[] | null}
            error={documentsError?.message}
            analysesMap={analysesMap}
          />
        </section>
      </main>
    </div>
  );
}
