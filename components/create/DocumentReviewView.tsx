"use client";

import { useState } from "react";
import type { DocumentReview, ReviewStatus } from "@/lib/ai/review-schema";

type DocumentReviewViewProps = {
  review: DocumentReview;
  documentTitle: string;
  onReturnToDocument?: () => void;
  onJumpToSection?: (sectionId: string) => void;
};

export function DocumentReviewView({
  review,
  documentTitle,
  onReturnToDocument,
  onJumpToSection,
}: DocumentReviewViewProps) {
  const [selectedStatusFilter, setSelectedStatusFilter] = useState<
    ReviewStatus | "all"
  >("all");

  const clearCount = review.findings.filter((f) => f.status === "clear").length;
  const attentionCount = review.findings.filter(
    (f) => f.status === "attention",
  ).length;
  const concernCount = review.findings.filter(
    (f) => f.status === "potential_concern",
  ).length;

  const filteredFindings =
    selectedStatusFilter === "all"
      ? review.findings
      : review.findings.filter((f) => f.status === selectedStatusFilter);

  const getStatusBadge = (status: ReviewStatus) => {
    switch (status) {
      case "clear":
        return (
          <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-bold text-emerald-700 dark:text-emerald-400">
            ✓ CLEAR
          </span>
        );
      case "attention":
        return (
          <span className="rounded-full bg-amber-500/10 border border-amber-500/30 px-3 py-1 text-xs font-bold text-amber-700 dark:text-amber-400">
            ⚠️ NEEDS ATTENTION
          </span>
        );
      case "potential_concern":
        return (
          <span className="rounded-full bg-rose-500/10 border border-rose-500/30 px-3 py-1 text-xs font-bold text-rose-700 dark:text-rose-400">
            ⚠️ POTENTIAL CONCERN
          </span>
        );
    }
  };

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-7 sm:p-8 shadow-xs">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
            Review your agreement
          </h2>
          <p className="mt-1.5 text-base text-muted">
            We found a few areas worth a closer look ({documentTitle}).
          </p>
        </div>

        {onReturnToDocument ? (
          <button
            onClick={onReturnToDocument}
            className="inline-flex items-center gap-1.5 text-sm font-bold text-accent hover:underline shrink-0"
          >
            ← View Agreement
          </button>
        ) : null}
      </div>

      {/* Prominent Legal Disclaimer Banner */}
      <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 text-sm text-amber-900 dark:text-amber-300 space-y-1.5">
        <div className="flex items-center gap-2 font-bold text-base">
          <span>⚠️ Important Legal Notice</span>
        </div>
        <p className="leading-relaxed">
          AI-generated review for informational purposes only. It is not a substitute for professional legal advice.
        </p>
      </div>

      {/* Overall Summary Card */}
      <div className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-3 card-hover shadow-xs">
        <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
          Review Overview
        </h3>
        <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
          {review.overall_summary}
        </p>
      </div>

      {/* Filter / Status Count Pills */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-5">
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => setSelectedStatusFilter("all")}
            className={`rounded-xl px-4 py-1.5 text-xs font-bold transition-all ${
              selectedStatusFilter === "all"
                ? "bg-accent text-white shadow-2xs"
                : "bg-surface border border-border text-foreground hover:border-accent/40"
            }`}
          >
            All Findings ({review.findings.length})
          </button>

          {clearCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("clear")}
              className={`rounded-xl px-4 py-1.5 text-xs font-bold transition-all ${
                selectedStatusFilter === "clear"
                  ? "bg-emerald-600 text-white shadow-2xs"
                  : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-500/20"
              }`}
            >
              ✓ Clear ({clearCount})
            </button>
          ) : null}

          {attentionCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("attention")}
              className={`rounded-xl px-4 py-1.5 text-xs font-bold transition-all ${
                selectedStatusFilter === "attention"
                  ? "bg-amber-600 text-white shadow-2xs"
                  : "bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-400 hover:bg-amber-500/20"
              }`}
            >
              ⚠️ Needs Attention ({attentionCount})
            </button>
          ) : null}

          {concernCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("potential_concern")}
              className={`rounded-xl px-4 py-1.5 text-xs font-bold transition-all ${
                selectedStatusFilter === "potential_concern"
                  ? "bg-rose-600 text-white shadow-2xs"
                  : "bg-rose-500/10 border border-rose-500/30 text-rose-700 dark:text-rose-400 hover:bg-rose-500/20"
              }`}
            >
              ⚠️ Potential Concern ({concernCount})
            </button>
          ) : null}
        </div>

        <span className="text-xs text-muted font-mono font-semibold">
          {filteredFindings.length} of {review.findings.length} items
        </span>
      </div>

      {/* Clause Finding Cards */}
      <div className="space-y-6">
        {filteredFindings.map((finding, idx) => (
          <article
            key={idx}
            className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-5 transition-all card-hover shadow-xs"
          >
            {/* Finding Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-4">
              <div className="flex items-center gap-3 flex-wrap">
                {getStatusBadge(finding.status)}
                <span className="rounded-md bg-accent/10 px-2.5 py-0.5 text-xs font-bold text-accent">
                  {finding.category}
                </span>
                <h4 className="text-lg font-bold text-foreground">
                  {finding.section_title}
                </h4>
              </div>

              {onJumpToSection ? (
                <button
                  onClick={() => onJumpToSection(finding.section_id)}
                  className="inline-flex items-center text-xs font-bold text-accent hover:underline shrink-0"
                >
                  View clause →
                </button>
              ) : null}
            </div>

            {/* Excerpt Box */}
            <div className="rounded-xl border border-border bg-surface-inset p-4 text-base italic text-foreground/80 border-l-4 border-l-accent/70 leading-relaxed">
              &quot;{finding.clause_excerpt}&quot;
            </div>

            {/* Why It Matters & What To Clarify */}
            <div className="grid gap-5 sm:grid-cols-2 text-base">
              <div className="space-y-1.5">
                <span className="font-bold text-foreground uppercase tracking-wider text-xs block">
                  Why It Matters
                </span>
                <p className="text-foreground/90 leading-relaxed">
                  {finding.why_it_matters}
                </p>
              </div>

              <div className="space-y-1.5">
                <span className="font-bold text-foreground uppercase tracking-wider text-xs block">
                  What To Clarify
                </span>
                <p className="text-foreground/90 leading-relaxed">
                  {finding.what_to_clarify}
                </p>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
