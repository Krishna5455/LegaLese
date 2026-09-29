"use client";

import { useState } from "react";
import {
  ShieldCheck,
  Scale,
  Info,
  ChevronDown,
  ChevronUp,
  FileCheck2,
  CheckCircle2,
  ArrowRight,
} from "lucide-react";
import type { TrustSummary } from "@/types/legalflow";

interface TrustSignalCardProps {
  trustSummary?: TrustSummary;
  onRequestReview?: () => void;
  scenarioKey?: string;
  hasHighRiskFindings?: boolean;
}

export function TrustSignalCard({
  trustSummary,
  onRequestReview,
  scenarioKey,
  hasHighRiskFindings = false,
}: TrustSignalCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (!trustSummary) return null;

  const isProfRecommended =
    trustSummary.level === "professional_review_recommended" || hasHighRiskFindings;
  const isFormalProcess = trustSummary.level === "formal_process";

  // Deterministic specific reasons based on scenario and analysis
  const specificReasons: string[] = [];
  if (scenarioKey === "hire_freelancer") {
    specificReasons.push(
      "Contractual liability limits and indemnity caps affect exposure.",
      "Work-for-hire and IP assignment clauses determine deliverable copyright.",
      "Milestone payment triggers and termination notice periods."
    );
  } else if (scenarioKey === "create_nda") {
    specificReasons.push(
      "Duration and scope of proprietary trade secret definitions.",
      "Non-compete or non-solicitation restrictions requiring commercial fairness."
    );
  } else {
    specificReasons.push(
      "Unilateral indemnification or uncapped liability risks.",
      "Enforceability of dispute jurisdiction and statutory governing law."
    );
  }

  return (
    <div className="rounded-2xl border border-[#E7E5E2] bg-white p-5 sm:p-6 space-y-4 shadow-xs">
      {/* Card Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className={`flex h-9 w-9 items-center justify-center rounded-xl shrink-0 ${
              isProfRecommended
                ? "bg-[#FEF3C7] text-[#B45309] border border-[#FDE68A]"
                : isFormalProcess
                ? "bg-[#EFF6FF] text-[#1D4ED8] border border-[#BFDBFE]"
                : "bg-[#059669]/10 text-[#059669] border border-[#059669]/20"
            }`}
          >
            {isProfRecommended ? (
              <Scale className="w-4 h-4" />
            ) : isFormalProcess ? (
              <Info className="w-4 h-4" />
            ) : (
              <ShieldCheck className="w-4 h-4" />
            )}
          </div>
          <div className="min-w-0">
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98]">
              Trust &amp; Review Layer
            </p>
            <h3 className="text-sm font-bold text-[#171717] truncate">
              {isProfRecommended
                ? "Professional Review Recommended"
                : trustSummary.label}
            </h3>
          </div>
        </div>

        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold uppercase ${
            isProfRecommended
              ? "bg-[#FEF3C7] text-[#B45309]"
              : "bg-[#059669]/10 text-[#059669]"
          }`}
        >
          {isProfRecommended ? "Recommended" : "AI Assisted"}
        </span>
      </div>

      {/* 3-Level Trust Status Matrix */}
      <div className="rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-3 space-y-2 text-xs">
        {/* Tier 1: AI Assistance */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#059669]" />
            <span className="font-semibold text-[#171717]">AI Assistance</span>
          </div>
          <span className="text-[11px] text-[#5F6368]">Drafting &amp; Risk Pre-Audit</span>
        </div>

        {/* Tier 2: Professional Review */}
        <div className="flex items-center justify-between pt-1 border-t border-[#E7E5E2]">
          <div className="flex items-center gap-2">
            <Scale
              className={`w-3.5 h-3.5 ${
                isProfRecommended ? "text-[#B45309]" : "text-[#8A8F98]"
              }`}
            />
            <span
              className={`font-semibold ${
                isProfRecommended ? "text-[#B45309]" : "text-[#5F6368]"
              }`}
            >
              Professional Review
            </span>
          </div>
          <span
            className={`text-[11px] font-medium ${
              isProfRecommended ? "text-[#B45309]" : "text-[#8A8F98]"
            }`}
          >
            {isProfRecommended ? "Recommended" : "Optional"}
          </span>
        </div>

        {/* Tier 3: Formal Process */}
        <div className="flex items-center justify-between pt-1 border-t border-[#E7E5E2]">
          <div className="flex items-center gap-2">
            <FileCheck2 className="w-3.5 h-3.5 text-[#8A8F98]" />
            <span className="font-semibold text-[#5F6368]">Formal Process</span>
          </div>
          <span className="text-[11px] text-[#8A8F98]">Bilateral Execution</span>
        </div>
      </div>

      {/* Explanation Text */}
      <p className="text-xs text-[#5F6368] leading-relaxed">
        {trustSummary.reason}
      </p>

      {/* Expandable "Why is this recommended?" Accordion */}
      {isProfRecommended && (
        <div className="border-t border-[#F0EFEA] pt-2 space-y-2">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="flex items-center justify-between w-full text-xs font-semibold text-[#171717] hover:text-[#059669] transition-colors cursor-pointer"
          >
            <span>Why is professional review recommended?</span>
            {expanded ? (
              <ChevronUp className="w-3.5 h-3.5 text-[#8A8F98]" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 text-[#8A8F98]" />
            )}
          </button>

          {expanded && (
            <div className="rounded-lg bg-white border border-[#E7E5E2] p-3 space-y-1.5 text-xs text-[#5F6368] animate-fade-in">
              <p className="font-semibold text-[#171717] text-[11px]">
                Identified areas benefiting from human legal judgment:
              </p>
              <ul className="space-y-1 text-[11px]">
                {specificReasons.map((r, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-[#B45309] font-bold">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Formal notes if available */}
      {trustSummary.formal_notes && trustSummary.formal_notes.length > 0 && (
        <div className="rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-3 space-y-1">
          <p className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98] font-bold">
            Formal Process Guidance:
          </p>
          <ul className="space-y-1 text-xs text-[#5F6368]">
            {trustSummary.formal_notes.map((note, idx) => (
              <li key={idx} className="flex items-start gap-1.5">
                <span className="text-[#059669] font-bold leading-none mt-1">•</span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* CTA Button: Request Professional Review (Handoff Package) */}
      {onRequestReview && (
        <button
          type="button"
          onClick={onRequestReview}
          className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-[#171717] px-4 py-2.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs cursor-pointer active:scale-98"
        >
          <Scale className="w-3.5 h-3.5 text-[#059669]" />
          <span>Request Professional Review [Demo]</span>
          <ArrowRight className="w-3.5 h-3.5 text-[#8A8F98]" />
        </button>
      )}

      {/* Disclaimer */}
      <p className="text-[10px] text-[#8A8F98] leading-normal pt-1 border-t border-[#F0EFEA]">
        LegaLese provides workflow assistance and structured guidance. This is a recommendation signal, not formal legal advice.
      </p>
    </div>
  );
}
