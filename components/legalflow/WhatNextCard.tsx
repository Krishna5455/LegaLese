"use client";

import Link from "next/link";
import { CircleDot, ArrowRight, CheckCircle2, AlertTriangle, FileText } from "lucide-react";
import type { LegalFlowStep } from "@/types/legalflow";

interface WhatNextCardProps {
  nextStep: LegalFlowStep | null;
  isComplete: boolean;
  onMarkStepComplete?: (stepId: string) => void;
  connectedDocUrl?: string | null;
}

export function WhatNextCard({
  nextStep,
  isComplete,
  onMarkStepComplete,
  connectedDocUrl,
}: WhatNextCardProps) {
  if (isComplete || !nextStep) {
    return (
      <div className="rounded-2xl border border-[#059669]/30 bg-[#059669]/5 p-6 sm:p-7 space-y-4 animate-fade-in shadow-xs">
        <div className="flex items-center gap-2 text-xs font-bold text-[#059669] uppercase tracking-wider">
          <CheckCircle2 className="w-4 h-4 text-[#059669]" />
          <span>LegalFlow Complete</span>
        </div>
        <div className="space-y-1">
          <h3 className="text-lg font-bold text-[#171717]">
            Your main journey steps are complete
          </h3>
          <p className="text-xs sm:text-sm text-[#5F6368] leading-relaxed">
            All primary requirements and review checkpoints for this legal matter have been fulfilled.
          </p>
        </div>
        <div className="flex flex-wrap gap-2.5 pt-2">
          {connectedDocUrl ? (
            <Link
              href={connectedDocUrl}
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
            >
              <FileText className="w-3.5 h-3.5 text-[#059669]" />
              <span>View Connected Document</span>
            </Link>
          ) : (
            <Link
              href="/dashboard#documents"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
            >
              <span>Open Document Vault</span>
            </Link>
          )}
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-4 py-2 text-xs font-semibold text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-2xs"
          >
            <span>Return to Workspace</span>
          </Link>
        </div>
      </div>
    );
  }

  const isRecommended = nextStep.status === "recommended";
  const isBlocked = nextStep.status === "blocked";
  const isOptional = nextStep.status === "optional";

  return (
    <div className="rounded-2xl border-2 border-[#059669]/40 bg-white p-6 sm:p-7 space-y-4 shadow-sm relative overflow-hidden">
      {/* Top Tag */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-bold text-[#059669] uppercase tracking-wider">
          <CircleDot className="w-4 h-4 animate-pulse text-[#059669]" />
          <span>What Do I Do Next?</span>
        </div>
        <span
          className={`rounded-full px-2.5 py-0.5 text-[10px] font-mono font-semibold uppercase ${
            isRecommended
              ? "bg-[#FEF3C7] text-[#B45309] border border-[#FDE68A]"
              : isBlocked
              ? "bg-[#FEF2F2] text-[#B91C1C] border border-[#FECACA]"
              : isOptional
              ? "bg-[#F7F7F5] text-[#8A8F98] border border-[#E7E5E2]"
              : "bg-[#059669]/10 text-[#059669] border border-[#059669]/20"
          }`}
        >
          {isRecommended
            ? "Recommended"
            : isBlocked
            ? "Requires Attention"
            : isOptional
            ? "Optional"
            : "Next Step"}
        </span>
      </div>

      {/* Main Title & Explanation */}
      <div className="space-y-1.5">
        <h3 className="text-lg sm:text-xl font-bold text-[#171717]">
          {nextStep.title}
        </h3>
        <p className="text-xs sm:text-[13px] text-[#5F6368] leading-relaxed">
          {nextStep.description}
        </p>
      </div>

      {/* Formal Guidance alert if applicable */}
      {nextStep.formalGuidance && (
        <div className="rounded-lg bg-[#FEF3C7]/40 border border-[#FDE68A] p-3 text-xs text-[#92400E] flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-[#B45309] shrink-0 mt-0.5" />
          <span>{nextStep.formalGuidance}</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        {nextStep.actionUrl ? (
          <Link
            href={nextStep.actionUrl}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-5 py-2.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs active:scale-98"
          >
            <span>{nextStep.actionLabel || "Proceed"}</span>
            <ArrowRight className="w-3.5 h-3.5 text-[#059669]" />
          </Link>
        ) : (
          <button
            type="button"
            onClick={() => onMarkStepComplete && onMarkStepComplete(nextStep.id)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#059669] px-5 py-2.5 text-xs font-semibold text-white hover:bg-[#047857] transition-all shadow-xs active:scale-98 cursor-pointer"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Mark Step Completed</span>
          </button>
        )}

        {nextStep.actionUrl && onMarkStepComplete && (
          <button
            type="button"
            onClick={() => onMarkStepComplete(nextStep.id)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-3.5 py-2 text-xs font-semibold text-[#5F6368] hover:text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-2xs cursor-pointer"
          >
            <CheckCircle2 className="w-3.5 h-3.5 text-[#059669]" />
            <span>Mark as Done</span>
          </button>
        )}
      </div>
    </div>
  );
}
