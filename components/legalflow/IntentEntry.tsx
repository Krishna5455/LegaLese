"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sparkles, ArrowRight, Loader2, AlertCircle } from "lucide-react";
import { createLegalFlowFromIntent } from "@/lib/actions/legalflow";

const CANONICAL_SUGGESTIONS = [
  { label: "Hire a freelancer for my website", query: "I want to hire a freelancer for my website" },
  { label: "Review an incoming contract for risk", query: "I want to review a contract" },
  { label: "Create a bilateral NDA", query: "I want to create an NDA" },
];

export function IntentEntry() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("Structuring your legal journey...");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    router.prefetch("/dashboard/flow/demo-freelance-flow");
    router.prefetch("/dashboard/vault");
    router.prefetch("/dashboard/calendar");
  }, [router]);

  const handleSubmit = async (overrideQuery?: string) => {
    const targetQuery = (overrideQuery ?? query).trim();
    if (!targetQuery) {
      setErrorMessage("Please describe what you are trying to accomplish.");
      return;
    }

    setErrorMessage(null);
    setIsSubmitting(true);
    setLoadingMessage("Entering workspace...");

    try {
      const result = await createLegalFlowFromIntent(targetQuery);

      if (result.error || !result.flowId) {
        setErrorMessage(result.error || "Could not start LegalFlow. Please try again.");
        setIsSubmitting(false);
        return;
      }

      router.push(`/dashboard/flow/${result.flowId}`);
    } catch {
      setErrorMessage("An unexpected error occurred. Please try again.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="rounded-2xl border border-[#E5E5E3] bg-white p-6 sm:p-8 shadow-xs space-y-5">
      <div className="space-y-1.5">
        <div className="inline-flex items-center gap-1.5 rounded-full bg-[#ECFDF5] border border-[#A7F3D0] px-2.5 py-0.5 text-[11px] font-semibold text-[#065F46]">
          <Sparkles className="w-3 h-3 text-[#059669]" />
          <span>Intent Command Center</span>
        </div>
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#111827]">
          What are you trying to accomplish?
        </h1>
        <p className="text-xs sm:text-sm text-[#4B5563]">
          Type your goal in plain English. LegaLese will break down the journey into structured steps.
        </p>
      </div>

      {/* Input and Submit Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!isSubmitting) handleSubmit();
        }}
        className="space-y-3"
      >
        <div className="relative flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
          <div className="relative flex-1">
            <input
              id="intent-input"
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                if (errorMessage) setErrorMessage(null);
              }}
              disabled={isSubmitting}
              placeholder="e.g. I want to hire a freelancer for my website..."
              className="w-full rounded-xl border border-[#E5E5E3] bg-[#F9F9F8] px-4 py-3 text-sm text-[#111827] placeholder:text-[#9CA3AF] focus:bg-white focus:border-[#111827] focus:outline-none focus:ring-2 focus:ring-[#111827]/10 transition-all disabled:opacity-60"
              aria-label="What are you trying to accomplish?"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting || !query.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#111827] px-6 py-3 text-sm font-semibold text-white hover:bg-[#1F2937] transition-all shadow-xs disabled:cursor-not-allowed disabled:opacity-50 shrink-0 cursor-pointer active:scale-98"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-[#10B981]" />
                <span>{loadingMessage}</span>
              </>
            ) : (
              <>
                <span>Start LegalFlow</span>
                <ArrowRight className="w-4 h-4 text-[#10B981]" />
              </>
            )}
          </button>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <div className="flex items-center gap-2 text-xs text-[#DC2626] bg-[#FEF2F2] border border-[#FEE2E2] px-3.5 py-2 rounded-lg animate-fade-in">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}
      </form>

      {/* Suggestion Chips */}
      <div className="space-y-2 pt-2 border-t border-[#E5E5E3]">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-mono uppercase tracking-wider text-[#6B7280]">
            Suggested:
          </span>
          {CANONICAL_SUGGESTIONS.map((item) => (
            <button
              key={item.label}
              type="button"
              disabled={isSubmitting}
              onClick={() => {
                setQuery(item.query);
                handleSubmit(item.query);
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#E5E5E3] bg-[#F9F9F8] px-3 py-1 text-xs font-medium text-[#374151] hover:border-[#111827] hover:bg-white hover:text-[#111827] transition-all shadow-2xs cursor-pointer active:scale-98 disabled:opacity-50"
            >
              <span>{item.label}</span>
              <ArrowRight className="w-3 h-3 text-[#9CA3AF]" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
