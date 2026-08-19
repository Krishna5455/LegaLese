import Link from "next/link";
import { redirect } from "next/navigation";

import { DashboardNav } from "@/components/dashboard/DashboardNav";
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
  const attentionCount = Object.values(analysesMap).filter(
    (a) => a.risk_score === 3 || a.risk_score === 2,
  ).length;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <DashboardNav userEmail={user.email} active="dashboard" />

      {/* Main Workspace Container */}
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 sm:px-8 py-10 space-y-10">
        {/* Hero Header */}
        <div className="space-y-1.5">
          <h1 className="text-3xl font-extrabold tracking-tight text-foreground sm:text-4xl">
            Welcome back
          </h1>
          <p className="text-base text-muted font-normal">
            What would you like to do today?
          </p>
        </div>

        {/* Primary Action Cards */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Card 1: Create a Legal Document */}
          <div className="rounded-2xl border border-border bg-surface p-8 space-y-6 flex flex-col justify-between card-hover shadow-xs">
            <div className="space-y-3.5">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/10 text-accent font-bold text-xl shadow-xs">
                📝
              </div>
              <h2 className="text-xl font-bold text-foreground sm:text-2xl">
                Create a Legal Document
              </h2>
              <p className="text-base text-muted leading-relaxed">
                Answer a few simple questions and create your agreement in minutes.
              </p>
            </div>
            <div>
              <Link
                href="/dashboard/create"
                className="inline-flex items-center gap-1.5 text-base font-bold text-white bg-accent hover:bg-accent-hover px-5 py-2.5 rounded-xl transition-all shadow-xs"
              >
                Create document →
              </Link>
            </div>
          </div>

          {/* Card 2: Analyze a Contract */}
          <div className="rounded-2xl border border-border bg-surface p-8 space-y-6 flex flex-col justify-between card-hover shadow-xs">
            <div className="space-y-3.5">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-600 font-bold text-xl shadow-xs">
                🔍
              </div>
              <h2 className="text-xl font-bold text-foreground sm:text-2xl">
                Analyze a Contract
              </h2>
              <p className="text-base text-muted leading-relaxed">
                Upload an existing contract and understand what needs attention before signing.
              </p>
            </div>
            <div>
              <a
                href="#upload"
                className="inline-flex items-center gap-1.5 text-base font-bold text-accent border border-accent/30 bg-accent/5 hover:bg-accent/10 px-5 py-2.5 rounded-xl transition-all"
              >
                Analyze document →
              </a>
            </div>
          </div>
        </div>

        {/* Workspace Metrics Summary */}
        <div className="grid grid-cols-2 gap-5 sm:grid-cols-3">
          <div className="rounded-2xl border border-border bg-surface p-6 shadow-xs card-hover">
            <p className="text-xs font-bold uppercase tracking-wider text-muted">
              Total Contracts
            </p>
            <p className="mt-2 text-3xl font-extrabold text-foreground">
              {totalCount}
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-surface p-6 shadow-xs card-hover">
            <p className="text-xs font-bold uppercase tracking-wider text-muted">
              Analyzed Documents
            </p>
            <p className="mt-2 text-3xl font-extrabold text-accent">
              {analyzedCount}
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-surface p-6 shadow-xs card-hover col-span-2 sm:col-span-1">
            <p className="text-xs font-bold uppercase tracking-wider text-muted">
              Needs Attention
            </p>
            <p className="mt-2 text-3xl font-extrabold text-amber-600">
              {attentionCount}
            </p>
          </div>
        </div>

        {/* Contract Upload Section */}
        <section id="upload" className="space-y-3">
          <ContractUpload />
        </section>

        {/* Recent Documents Table */}
        <section className="rounded-2xl border border-border bg-surface p-8 space-y-6 shadow-xs card-hover">
          <div className="flex items-center justify-between border-b border-border pb-5">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                Recent Documents
              </h2>
              <p className="text-base text-muted">
                View stored agreements and access deep-dive contract reviews.
              </p>
            </div>
            {documents && documents.length > 0 && (
              <span className="rounded-full bg-accent/10 border border-accent/20 px-3.5 py-1 text-xs font-bold text-accent">
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
