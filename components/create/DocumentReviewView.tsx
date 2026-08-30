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
          <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
            ✓ CLEAR
          </span>
        );
      case "attention":
        return (
          <span className="rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-xs font-semibold text-amber-700">
            ⚠️ NEEDS ATTENTION
          </span>
        );
      case "potential_concern":
        return (
          <span className="rounded-full bg-rose-50 border border-rose-200 px-2.5 py-0.5 text-xs font-semibold text-rose-700">
            ⚠️ POTENTIAL CONCERN
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-xl border border-indigo-200 bg-indigo-50/60 p-5 sm:p-6 shadow-xs">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
            Agreement Risk Review
          </h2>
          <p className="mt-1 text-xs sm:text-sm text-secondary">
            AI-driven clause evaluation for &quot;{documentTitle}&quot;.
          </p>
        </div>

        {onReturnToDocument ? (
          <button
            type="button"
            onClick={onReturnToDocument}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-accent hover:underline shrink-0"
          >
            ← View Full Text
          </button>
        ) : null}
      </div>

      {/* Prominent Legal Disclaimer Banner */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900 space-y-1">
        <div className="flex items-center gap-2 font-bold">
          <span>⚠️ Informational Review Notice</span>
        </div>
        <p className="leading-relaxed">
          AI-generated risk review for informational purposes only. Consult legal counsel for binding contract negotiation.
        </p>
      </div>

      {/* Overall Summary Card */}
      <div className="rounded-xl border border-border bg-surface p-6 space-y-2.5 card-hover shadow-xs">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
          Review Overview
        </h3>
        <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
          {review.overall_summary}
        </p>
      </div>

      {/* Filter / Status Count Pills */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setSelectedStatusFilter("all")}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
              selectedStatusFilter === "all"
                ? "bg-accent text-white shadow-xs"
                : "bg-surface border border-border text-foreground hover:bg-slate-50"
            }`}
          >
            All Findings ({review.findings.length})
          </button>

          {clearCount > 0 ? (
            <button
              type="button"
              onClick={() => setSelectedStatusFilter("clear")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                selectedStatusFilter === "clear"
                  ? "bg-emerald-600 text-white shadow-xs"
                  : "bg-emerald-50 border border-emerald-200 text-emerald-700 hover:bg-emerald-100"
              }`}
            >
              ✓ Clear ({clearCount})
            </button>
          ) : null}

          {attentionCount > 0 ? (
            <button
              type="button"
              onClick={() => setSelectedStatusFilter("attention")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                selectedStatusFilter === "attention"
                  ? "bg-amber-600 text-white shadow-xs"
                  : "bg-amber-50 border border-amber-200 text-amber-700 hover:bg-amber-100"
              }`}
            >
              ⚠️ Attention ({attentionCount})
            </button>
          ) : null}

          {concernCount > 0 ? (
            <button
              type="button"
              onClick={() => setSelectedStatusFilter("potential_concern")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                selectedStatusFilter === "potential_concern"
                  ? "bg-rose-600 text-white shadow-xs"
                  : "bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100"
              }`}
            >
              ⚠️ Concern ({concernCount})
            </button>
          ) : null}
        </div>

        <span className="text-xs text-muted font-mono font-medium">
          {filteredFindings.length} of {review.findings.length} items
        </span>
      </div>

      {/* Clause Finding Cards */}
      <div className="space-y-4">
        {filteredFindings.map((finding, idx) => (
          <article
            key={idx}
            className="rounded-xl border border-border bg-surface p-6 space-y-4 transition-all card-hover shadow-xs"
          >
            {/* Finding Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-2.5 flex-wrap">
                {getStatusBadge(finding.status)}
                <span className="rounded bg-accent-soft border border-accent/20 px-2 py-0.5 text-xs font-semibold text-accent">
                  {finding.category}
                </span>
                <h4 className="text-sm font-bold text-foreground">
                  {finding.section_title}
                </h4>
              </div>

              {onJumpToSection ? (
                <button
                  type="button"
                  onClick={() => onJumpToSection(finding.section_id)}
                  className="inline-flex items-center text-xs font-semibold text-accent hover:underline shrink-0"
                >
                  View clause →
                </button>
              ) : null}
            </div>

            {/* Excerpt Box */}
            <div className="rounded-lg border border-border bg-background p-3.5 text-xs italic text-secondary border-l-4 border-l-accent leading-relaxed">
              &quot;{finding.clause_excerpt}&quot;
            </div>

            {/* Why It Matters & What To Clarify */}
            <div className="grid gap-4 sm:grid-cols-2 text-xs">
              <div className="space-y-1">
                <span className="font-bold text-foreground uppercase tracking-wider text-[11px] block">
                  Why It Matters
                </span>
                <p className="text-secondary leading-relaxed">
                  {finding.why_it_matters}
                </p>
              </div>

              <div className="space-y-1">
                <span className="font-bold text-foreground uppercase tracking-wider text-[11px] block">
                  What To Clarify
                </span>
                <p className="text-secondary leading-relaxed">
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

