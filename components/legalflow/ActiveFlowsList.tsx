"use client";

import Link from "next/link";
import { ArrowRight, Compass, CircleDot, AlertTriangle, ShieldCheck } from "lucide-react";
import { getFlowNextAction } from "@/lib/legalflow/intent-engine";
import type { LegalFlowRow } from "@/types/legalflow";

interface ActiveFlowsListProps {
  flows: LegalFlowRow[];
}

export function ActiveFlowsList({ flows }: ActiveFlowsListProps) {
  if (!flows || flows.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-[#E5E5E3] bg-white p-8 text-center space-y-4">
        <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] text-[#6B7280]">
          <Compass className="w-5 h-5" />
        </div>
        <div className="max-w-md mx-auto space-y-1">
          <h3 className="text-sm font-bold text-[#111827]">
            No active LegalFlows
          </h3>
          <p className="text-xs text-[#4B5563]">
            Type your goal above to start a guided legal journey.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <h2 className="text-base font-bold text-[#111827]">Continue LegalFlow</h2>
          <p className="text-xs text-[#6B7280]">
            Track progress and next actions for your active journeys.
          </p>
        </div>
        <span className="rounded-full bg-[#ECFDF5] border border-[#A7F3D0] px-2.5 py-0.5 text-xs font-semibold text-[#065F46]">
          {flows.length} Active
        </span>
      </div>

      <div className="space-y-4">
        {flows.map((flow) => {
          const { completedCount, totalCount, progressPercent, nextStep } =
            getFlowNextAction(flow.steps || []);

          const isCompleted = flow.status === "completed" || progressPercent === 100;

          return (
            <div
              key={flow.id}
              className="group rounded-2xl border border-[#E5E5E3] bg-white p-6 space-y-5 shadow-xs hover:border-[#D1D5DB] transition-all"
            >
              {/* Header: Title and Overall Progress */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#E5E5E3]">
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#059669] bg-[#ECFDF5] px-2 py-0.5 rounded border border-[#A7F3D0]">
                      Active Journey
                    </span>
                    <h3 className="text-base font-bold text-[#111827] truncate">
                      {flow.title}
                    </h3>
                  </div>
                  <p className="text-xs text-[#6B7280]">
                    Intent: &quot;{flow.intent_query}&quot;
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[#111827]">
                    {progressPercent}% Complete
                  </span>
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[10px] font-mono font-semibold uppercase ${
                      isCompleted
                        ? "bg-[#ECFDF5] text-[#065F46] border border-[#A7F3D0]"
                        : "bg-[#F9F9F8] text-[#4B5563] border border-[#E5E5E3]"
                    }`}
                  >
                    {isCompleted ? "Complete" : `${completedCount}/${totalCount} Steps`}
                  </span>
                </div>
              </div>

              {/* 6-Stage Progress Indicator */}
              <div className="space-y-2">
                <div className="h-2 w-full overflow-hidden rounded-full bg-[#F3F4F6]">
                  <div
                    className="h-full rounded-full bg-[#059669] transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <div className="grid grid-cols-6 gap-1 text-center text-[10px] text-[#6B7280] font-medium hidden sm:grid">
                  <span className="text-[#111827] font-semibold">1. Understand</span>
                  <span className="text-[#111827] font-semibold">2. Prepare</span>
                  <span className="text-[#059669] font-bold">3. Review</span>
                  <span>4. Trust</span>
                  <span>5. Track</span>
                  <span>6. Complete</span>
                </div>
              </div>

              {/* Dominant Next Action Prompt */}
              <div className="rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-[#111827]">
                    <CircleDot className="w-3.5 h-3.5 text-[#059669]" />
                    <span>Next Action: {nextStep ? nextStep.title : "Journey requirements fulfilled"}</span>
                  </div>
                  {nextStep?.description && (
                    <p className="text-xs text-[#4B5563]">
                      {nextStep.description}
                    </p>
                  )}
                </div>

                <Link
                  href={`/dashboard/flow/${flow.id}`}
                  className="shrink-0 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#111827] text-white text-xs font-semibold hover:bg-[#1F2937] transition-all shadow-xs cursor-pointer active:scale-98"
                >
                  <span>Continue LegalFlow</span>
                  <ArrowRight className="w-3.5 h-3.5 text-[#10B981]" />
                </Link>
              </div>

              {/* Trust Signal Footer */}
              <div className="flex items-center justify-between text-xs text-[#6B7280] pt-1">
                <div className="flex items-center gap-2">
                  {flow.trust_summary?.level === "professional_review_recommended" ? (
                    <span className="flex items-center gap-1 text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded text-[11px] font-medium">
                      <AlertTriangle className="w-3 h-3" />
                      <span>Professional Review Recommended for Liability & IP</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-[#059669] bg-[#ECFDF5] border border-[#A7F3D0] px-2 py-0.5 rounded text-[11px] font-medium">
                      <ShieldCheck className="w-3 h-3" />
                      <span>AI-Assisted Drafting Ready</span>
                    </span>
                  )}
                </div>

                <span className="text-[11px] text-[#9CA3AF]">
                  Last updated {new Date(flow.updated_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
