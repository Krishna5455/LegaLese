import Link from "next/link";
import { redirect } from "next/navigation";
import { Plus, Upload, FilePlus, Search, ShieldAlert, FileText, CheckCircle2 } from "lucide-react";

import { ContractUpload } from "@/components/dashboard/ContractUpload";
import { DocumentList } from "@/components/dashboard/DocumentList";
import { SpotlightCard } from "@/components/ui/SpotlightCard";
import { createClient } from "@/lib/supabase/server";

import type { DetailedAnalysis } from "@/types/analysis";
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

  // Fetch user documents and analyses summary in parallel
  const [
    { data: documents, error: documentsError },
    { data: analyses },
  ] = await Promise.all([
    supabase
      .from("documents")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false }),
    supabase
      .from("analyses")
      .select("id, document_id, risk_score, summary, created_at, model")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false }),
  ]);

  // Build lightweight analyses summary map for instant rendering
  const analysesMap: Record<string, DetailedAnalysis> = {};

  if (analyses && analyses.length > 0) {
    for (const analysis of analyses) {
      if (!analysesMap[analysis.document_id]) {
        analysesMap[analysis.document_id] = {
          id: analysis.id,
          document_id: analysis.document_id,
          user_id: user.id,
          risk_score: analysis.risk_score,
          summary: analysis.summary,
          result: {},
          model: analysis.model,
          created_at: analysis.created_at,
          clauses: [],
          findings: [],
          key_terms: [],
          obligations: [],
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
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8 space-y-8">
      {/* Welcome Workspace Greeting Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-[#E7E5E2] pb-6">
        <div className="space-y-1">
          <h1 className="heading-page text-[#171717]">
            Workspace Overview
          </h1>
          <p className="text-xs sm:text-[14px] text-[#5F6368]">
            Create customized legal agreements and audit contracts before signing.
          </p>
        </div>

        {/* Quick Primary Actions Toolbar */}
        <div className="flex items-center gap-2.5 shrink-0">
          <Link
            href="/dashboard/create"
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-medium text-white hover:bg-[#262626] transition-all shadow-xs active:scale-98"
          >
            <Plus className="w-3.5 h-3.5 text-[#059669]" />
            <span>Create document</span>
          </Link>
          <Link
            href="#upload"
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-4 py-2 text-xs font-medium text-[#171717] hover:bg-[#F7F7F5] hover:border-[#D4D2CD] transition-all shadow-2xs active:scale-98"
          >
            <Upload className="w-3.5 h-3.5 text-[#5F6368]" />
            <span>Analyze contract</span>
          </Link>
        </div>
      </div>

      {/* Primary Action Cards Grid with SpotlightCard Effect */}
      <div className="grid gap-5 md:grid-cols-2">
        {/* Card 1: Create a Legal Document */}
        <SpotlightCard
          spotlightColor="rgba(5, 150, 105, 0.08)"
          className="rounded-2xl border-[#E7E5E2] bg-white p-6 space-y-4 shadow-xs"
        >
          <div className="flex items-start justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#059669]/10 text-[#059669] border border-[#059669]/20 shadow-2xs">
              <FilePlus className="w-5 h-5" />
            </div>
            <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
              Guided Generator
            </span>
          </div>
          <div className="space-y-1">
            <h2 className="text-base font-semibold text-[#171717]">
              Create a Legal Document
            </h2>
            <p className="text-xs sm:text-[13px] text-[#5F6368] leading-relaxed">
              Answer simple guided questions to generate customized, legally sound freelance and service agreements with plain English summaries.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href="/dashboard/create"
              className="inline-flex items-center gap-1 text-xs font-semibold text-[#059669] hover:underline"
            >
              <span>Start creating document</span>
              <span>→</span>
            </Link>
          </div>
        </SpotlightCard>

        {/* Card 2: Analyze an Existing Contract */}
        <SpotlightCard
          spotlightColor="rgba(5, 150, 105, 0.08)"
          className="rounded-2xl border-[#E7E5E2] bg-white p-6 space-y-4 shadow-xs"
        >
          <div className="flex items-start justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F7F7F5] text-[#171717] border border-[#E7E5E2] shadow-2xs">
              <Search className="w-5 h-5 text-[#059669]" />
            </div>
            <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
              Pre-Sign Audit
            </span>
          </div>
          <div className="space-y-1">
            <h2 className="text-base font-semibold text-[#171717]">
              Analyze Existing Contract
            </h2>
            <p className="text-xs sm:text-[13px] text-[#5F6368] leading-relaxed">
              Upload a PDF or DOCX file to spot high-risk liability traps, evaluate milestone fairness, and review plain English clause explanations.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href="#upload"
              className="inline-flex items-center gap-1 text-xs font-semibold text-[#059669] hover:underline"
            >
              <span>Upload file for analysis</span>
              <span>↓</span>
            </Link>
          </div>
        </SpotlightCard>
      </div>

      {/* Workspace Metrics Summary Bar */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-[#E7E5E2] bg-white p-5 card-hover shadow-2xs">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
              Total Contracts
            </p>
            <FileText className="w-4 h-4 text-[#8A8F98]" />
          </div>
          <p className="mt-2 text-2xl font-bold text-[#171717]">
            {totalCount}
          </p>
        </div>

        <div className="rounded-xl border border-[#E7E5E2] bg-white p-5 card-hover shadow-2xs">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
              Audited Documents
            </p>
            <CheckCircle2 className="w-4 h-4 text-[#059669]" />
          </div>
          <p className="mt-2 text-2xl font-bold text-[#059669]">
            {analyzedCount}
          </p>
        </div>

        <div className="rounded-xl border border-[#E7E5E2] bg-white p-5 card-hover shadow-2xs col-span-2 sm:col-span-1">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
              Needs Attention
            </p>
            <ShieldAlert className="w-4 h-4 text-[#B45309]" />
          </div>
          <p className="mt-2 text-2xl font-bold text-[#B45309]">
            {attentionCount}
          </p>
        </div>
      </div>

      {/* Contract Upload Section */}
      <section id="upload" className="space-y-3">
        <div className="space-y-1">
          <h2 className="heading-section text-[#171717]">
            Contract Analysis Dropzone
          </h2>
          <p className="text-xs sm:text-[13px] text-[#5F6368]">
            Upload agreements up to 10MB to detect risks, obligations, and key dates.
          </p>
        </div>
        <ContractUpload />
      </section>

      {/* Recent Documents Table Container */}
      <section id="documents" className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-5 shadow-sm">
        <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-4">
          <div>
            <h2 className="heading-section text-[#171717]">
              Recent Documents
            </h2>
            <p className="text-xs text-[#5F6368]">
              Manage saved agreements and access deep-dive contract audit workspaces.
            </p>
          </div>
          {documents && documents.length > 0 && (
            <span className="rounded-md bg-[#059669]/10 border border-[#059669]/20 px-2.5 py-0.5 text-xs font-semibold text-[#059669]">
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
  );
}
