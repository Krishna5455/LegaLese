"use client";

import Link from "next/link";
import {
  CheckCircle2,
  CircleDot,
  Clock,
  AlertTriangle,
  ArrowRight,
  ShieldAlert,
} from "lucide-react";
import type { LegalFlowStep } from "@/types/legalflow";

interface JourneyTimelineProps {
  steps: LegalFlowStep[];
  onToggleStep?: (stepId: string) => void;
}

export function JourneyTimeline({ steps, onToggleStep }: JourneyTimelineProps) {
  return (
    <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-8 space-y-6 shadow-xs">
      <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-4">
        <div className="space-y-0.5">
          <h2 className="text-base sm:text-lg font-bold text-[#171717]">
            Journey Sequence
          </h2>
          <p className="text-xs text-[#5F6368]">
            Sequential roadmap of actions required to finalize and protect this matter.
          </p>
        </div>
      </div>

      {/* Vertical Steps Timeline */}
      <div className="relative space-y-4 before:absolute before:left-[17px] before:top-3 before:bottom-3 before:w-0.5 before:bg-[#E7E5E2]">
        {steps.map((step, idx) => {
          const isDone = step.status === "completed";
          const isNext = step.status === "next";
          const isRecommended = step.status === "recommended";
          const isBlocked = step.status === "blocked";

          return (
            <div
              key={step.id || idx}
              className={`relative flex items-start gap-4 p-4 rounded-xl border transition-all ${
                isNext
                  ? "border-[#059669] bg-[#059669]/5 shadow-2xs"
                  : isDone
                  ? "border-[#E7E5E2] bg-[#F7F7F5]/50 opacity-75 hover:opacity-100"
                  : isRecommended
                  ? "border-[#FDE68A] bg-[#FEF3C7]/20"
                  : "border-[#E7E5E2] bg-white hover:border-[#D4D2CD]"
              }`}
            >
              {/* Step Status Icon Indicator */}
              <button
                type="button"
                onClick={() => onToggleStep && onToggleStep(step.id)}
                className="pt-0.5 z-10 shrink-0 cursor-pointer focus:outline-none"
                title={isDone ? "Mark incomplete" : "Mark completed"}
                aria-label={`Step ${idx + 1}: ${step.title}`}
              >
                {isDone ? (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#059669] text-white shadow-2xs hover:scale-105 transition-transform">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                ) : isNext ? (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-white border-2 border-[#059669] text-[#059669] shadow-2xs animate-pulse">
                    <CircleDot className="w-3.5 h-3.5" />
                  </div>
                ) : isRecommended ? (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#FEF3C7] border border-[#FDE68A] text-[#B45309]">
                    <AlertTriangle className="w-3.5 h-3.5" />
                  </div>
                ) : isBlocked ? (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#FEF2F2] border border-[#FECACA] text-[#B91C1C]">
                    <ShieldAlert className="w-3.5 h-3.5" />
                  </div>
                ) : (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#F7F7F5] border border-[#E7E5E2] text-[#8A8F98]">
                    <Clock className="w-3.5 h-3.5" />
                  </div>
                )}
              </button>

              {/* Step Details */}
              <div className="flex-1 space-y-1.5 min-w-0">
                <div className="flex items-center gap-2 flex-wrap justify-between">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3
                      className={`text-sm font-bold ${
                        isDone ? "text-[#5F6368] line-through" : "text-[#171717]"
                      }`}
                    >
                      {step.title}
                    </h3>

                    {step.category && (
                      <span className="rounded bg-[#F0EFEA] text-[#5F6368] border border-[#E7E5E2] px-2 py-0.2 text-[10px] font-mono">
                        {step.category}
                      </span>
                    )}

                    {step.badge && (
                      <span className="rounded bg-[#059669]/10 text-[#059669] border border-[#059669]/20 px-2 py-0.2 text-[10px] font-mono">
                        {step.badge}
                      </span>
                    )}
                  </div>

                  <span
                    className={`rounded-full px-2 py-0.2 text-[10px] font-mono font-semibold uppercase ${
                      isDone
                        ? "bg-[#059669]/10 text-[#059669]"
                        : isNext
                        ? "bg-[#059669] text-white"
                        : isRecommended
                        ? "bg-[#FEF3C7] text-[#B45309]"
                        : "bg-[#F7F7F5] text-[#8A8F98]"
                    }`}
                  >
                    {step.status}
                  </span>
                </div>

                <p className="text-xs text-[#5F6368] leading-relaxed">
                  {step.description}
                </p>

                {step.formalGuidance && (
                  <p className="text-[11px] text-[#B45309] flex items-center gap-1.5 pt-0.5">
                    <AlertTriangle className="w-3 h-3 shrink-0" />
                    <span>{step.formalGuidance}</span>
                  </p>
                )}

                {/* Step Action Button */}
                {step.actionUrl && !isDone && (
                  <div className="pt-2">
                    <Link
                      href={step.actionUrl}
                      className="inline-flex items-center gap-1 text-xs font-semibold text-[#059669] hover:underline"
                    >
                      <span>{step.actionLabel || "Execute Step"}</span>
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
