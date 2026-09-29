import Link from "next/link";
import { redirect } from "next/navigation";
import {
  Plus,
  Upload,
  ShieldAlert,
  Calendar,
  ArrowRight,
  FolderLock,
  DollarSign,
  Clock,
} from "lucide-react";

import { ContractUpload } from "@/components/dashboard/ContractUpload";
import { DocumentList } from "@/components/dashboard/DocumentList";
import { IntentEntry } from "@/components/legalflow/IntentEntry";
import { ActiveFlowsList } from "@/components/legalflow/ActiveFlowsList";
import { getCalendarObligations } from "@/lib/actions/legalflow";
import { createClient } from "@/lib/supabase/server";

import type { DetailedAnalysis } from "@/types/analysis";
import type { Document } from "@/types/database";
import type { LegalFlowRow } from "@/types/legalflow";

import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";
import { isDemoMode, listDemoFlows, DEMO_CALENDAR_EVENTS } from "@/lib/demo/demo-state";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  const isDemo = isDemoMode() && user.id === "00000000-0000-0000-0000-000000000000";

  let flows = isDemo ? listDemoFlows() : [];
  let documents: Document[] | null = isDemo ? [] : null;
  let documentsError: unknown = null;
  let analyses: Array<{ id: string; document_id: string; risk_score: number | null; summary: string; created_at: string; model?: string }> | null = isDemo ? [] : null;
  let calendarEvents = isDemo ? DEMO_CALENDAR_EVENTS : [];

  if (!isDemo) {
    // Fetch user legal flows, documents, analyses, and calendar obligations concurrently
    const [
      { data: dbFlows },
      { data: dbDocs, error: docErr },
      { data: dbAnalyses },
      { events: dbEvents },
    ] = await Promise.all([
      supabase
        .from("legal_flows")
        .select("id, user_id, title, intent_query, scenario_key, current_stage, status, steps, trust_summary, document_id, generated_doc_id, created_at, updated_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      supabase
        .from("documents")
        .select("id, user_id, filename, file_size, mime_type, storage_path, document_type, status, extracted_text, created_at, updated_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      supabase
        .from("analyses")
        .select("id, document_id, risk_score, summary, created_at, model")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      getCalendarObligations(),
    ]);

    flows = dbFlows || [];
    documents = dbDocs || [];
    documentsError = docErr;
    analyses = dbAnalyses || [];
    calendarEvents = dbEvents || [];
  }

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
          model: analysis.model || "gemini-3.5-flash-lite",
          created_at: analysis.created_at,
          clauses: [],
          findings: [],
          key_terms: [],
          obligations: [],
        };
      }
    }
  }

  const attentionCount = Object.values(analysesMap).filter(
    (a) => a.risk_score === 3 || a.risk_score === 2,
  ).length;

  const upcomingDates = (calendarEvents || []).slice(0, 2);

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8 space-y-8">
      {/* Welcome Workspace Greeting Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-[#E7E5E2] pb-6 stagger-1">
        <div className="space-y-1">
          <h1 className="heading-page text-[#171717]">
            Workspace Overview
          </h1>
          <p className="text-xs sm:text-[14px] text-[#5F6368]">
            LegaLese turns fragmented legal tasks into guided journeys from intent to complete execution.
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

      {/* 1. Primary Intent-First Entry Point */}
      <section className="stagger-1">
        <IntentEntry />
      </section>

      {/* 2. Active LegalFlows Command Center (Dominant Secondary Element) */}
      <section className="stagger-2">
        <ActiveFlowsList flows={(flows as LegalFlowRow[]) ?? []} />
      </section>

      {/* 3. Needs Attention Section (If any matters have high risk or flagged items) */}
      {attentionCount > 0 && (
        <section className="rounded-2xl border border-amber-200 bg-amber-50/50 p-5 sm:p-6 space-y-3 shadow-2xs stagger-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-100 text-amber-800">
                <ShieldAlert className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-[#171717]">Items Needing Your Attention</h3>
            </div>
            <span className="rounded-full bg-amber-100 text-amber-900 border border-amber-300 px-2.5 py-0.5 text-xs font-semibold">
              {attentionCount} {attentionCount === 1 ? "Matter" : "Matters"} Flagged
            </span>
          </div>
          <p className="text-xs text-[#5F6368]">
            The following documents contain clauses or risk findings where professional review or careful modification is recommended before signing.
          </p>
          <div className="grid sm:grid-cols-2 gap-3 pt-1">
            {Object.values(analysesMap)
              .filter((a) => (a.risk_score ?? 0) >= 2)
              .slice(0, 2)
              .map((analysis) => {
                const doc = (documents || []).find((d) => d.id === analysis.document_id);
                return (
                  <div
                    key={analysis.id}
                    className="flex items-center justify-between gap-3 rounded-xl bg-white border border-amber-200 p-3.5 text-xs shadow-2xs"
                  >
                    <div className="min-w-0 space-y-0.5">
                      <p className="font-bold text-[#171717] truncate">{doc?.filename || "Contract Document"}</p>
                      <p className="text-[11px] text-amber-800 line-clamp-1">{analysis.summary || "High risk liability or indemnity terms detected."}</p>
                    </div>
                    <Link
                      href={`/dashboard/documents/${analysis.document_id}`}
                      className="shrink-0 inline-flex items-center gap-1 rounded-lg bg-[#171717] px-3 py-1.5 text-[11px] font-semibold text-white hover:bg-[#262626] transition-all"
                    >
                      <span>Review</span>
                      <ArrowRight className="w-3 h-3 text-[#059669]" />
                    </Link>
                  </div>
                );
              })}
          </div>
        </section>
      )}

      {/* 4. Upcoming Obligations & Dates Ribbon */}
      {upcomingDates.length > 0 && (
        <section className="rounded-2xl border border-[#E7E5E2] bg-white p-5 sm:p-6 space-y-3 shadow-xs stagger-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#059669]/10 text-[#059669]">
                <Calendar className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-[#171717]">Upcoming Deadlines &amp; Obligations</h3>
            </div>
            <Link
              href="/dashboard/calendar"
              className="inline-flex items-center gap-1 text-xs font-semibold text-[#059669] hover:underline"
            >
              <span>View All Dates</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid sm:grid-cols-2 gap-3">
            {upcomingDates.map((evt) => (
              <div
                key={evt.id}
                className="flex items-center justify-between gap-3 rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-3 text-xs"
              >
                <div className="flex items-center gap-2 min-w-0">
                  {evt.type === "payment" ? (
                    <DollarSign className="w-4 h-4 text-[#059669] shrink-0" />
                  ) : (
                    <Clock className="w-4 h-4 text-[#8A8F98] shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="font-bold text-[#171717] truncate">{evt.title}</p>
                    <p className="text-[11px] text-[#5F6368]">{evt.description}</p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span className="font-mono font-semibold text-[#171717]">{evt.date}</span>
                  {evt.amount && (
                    <p className="text-[11px] font-bold text-[#059669]">{evt.amount}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 5. Recent Documents Repository & Dropzone */}
      <section id="documents" className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-6 shadow-sm stagger-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-[#E7E5E2] pb-4">
          <div>
            <h2 className="heading-section text-[#171717]">
              Documents Repository
            </h2>
            <p className="text-xs text-[#5F6368]">
              Manage generated agreements and audited contracts linked to your LegalFlows.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/dashboard/vault"
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#059669] hover:underline"
            >
              <FolderLock className="w-3.5 h-3.5" />
              <span>All Documents</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* Contract Analysis Dropzone */}
        <div id="upload" className="space-y-2">
          <p className="text-xs font-semibold text-[#171717]">Quick Contract Pre-Sign Audit</p>
          <ContractUpload />
        </div>

        {/* Document List */}
        <div className="pt-2">
          <DocumentList
            documents={documents as Document[] | null}
            error={(documentsError as { message?: string } | null)?.message}
            analysesMap={analysesMap}
          />
        </div>
      </section>
    </main>
  );
}
