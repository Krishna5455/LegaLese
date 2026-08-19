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
          <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
            ✓ CLEAR
          </span>
        );
      case "attention":
        return (
          <span className="rounded-full bg-amber-500/10 border border-amber-500/30 px-2.5 py-0.5 text-xs font-semibold text-amber-700 dark:text-amber-400">
            ⚠️ ATTENTION
          </span>
        );
      case "potential_concern":
        return (
          <span className="rounded-full bg-rose-500/10 border border-rose-500/30 px-2.5 py-0.5 text-xs font-semibold text-rose-700 dark:text-rose-400">
            ⚠️ POTENTIAL CONCERN
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-0.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400">
              Clause-Level Review
            </span>
            <span className="text-xs text-muted font-mono">
              {review.findings.length} findings identified
            </span>
          </div>
          <h2 className="mt-2 text-xl font-bold tracking-tight text-foreground">
            Contract Review: {documentTitle}
          </h2>
          <p className="mt-1 text-xs text-muted">
            Actionable clause breakdown categorized by clarity, attention items, and potential concerns.
          </p>
        </div>

        {onReturnToDocument ? (
          <button
            onClick={onReturnToDocument}
            className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline shrink-0"
          >
            ← View Full Agreement Text
          </button>
        ) : null}
      </div>

      {/* Prominent Legal Disclaimer Banner */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-800 dark:text-amber-300 space-y-1">
        <div className="flex items-center gap-2 font-bold">
          <span>⚠️ Important Legal Notice</span>
        </div>
        <p className="leading-relaxed">
          AI-generated review for informational purposes only. It is not a substitute for professional legal advice.
        </p>
      </div>

      {/* Overall Summary Card */}
      <div className="rounded-xl border border-border bg-surface p-6 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-base">📊</span>
          <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
            Agreement Review Summary
          </h3>
        </div>
        <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-line">
          {review.overall_summary}
        </p>
      </div>

      {/* Filter / Status Count Pills */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setSelectedStatusFilter("all")}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
              selectedStatusFilter === "all"
                ? "bg-accent text-white"
                : "bg-surface border border-border text-foreground hover:border-accent/40"
            }`}
          >
            All Findings ({review.findings.length})
          </button>

          {clearCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("clear")}
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                selectedStatusFilter === "clear"
                  ? "bg-emerald-600 text-white"
                  : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-500/20"
              }`}
            >
              ✓ Clear ({clearCount})
            </button>
          ) : null}

          {attentionCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("attention")}
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                selectedStatusFilter === "attention"
                  ? "bg-amber-600 text-white"
                  : "bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-400 hover:bg-amber-500/20"
              }`}
            >
              ⚠️ Attention ({attentionCount})
            </button>
          ) : null}

          {concernCount > 0 ? (
            <button
              onClick={() => setSelectedStatusFilter("potential_concern")}
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                selectedStatusFilter === "potential_concern"
                  ? "bg-rose-600 text-white"
                  : "bg-rose-500/10 border border-rose-500/30 text-rose-700 dark:text-rose-400 hover:bg-rose-500/20"
              }`}
            >
              ⚠️ Potential Concern ({concernCount})
            </button>
          ) : null}
        </div>

        <span className="text-xs text-muted font-mono">
          Showing {filteredFindings.length} of {review.findings.length}
        </span>
      </div>

      {/* Clause Finding Cards */}
      <div className="space-y-4">
        {filteredFindings.map((finding, idx) => (
          <article
            key={idx}
            className="rounded-xl border border-border bg-surface p-6 space-y-4 transition-colors hover:border-accent/30"
          >
            {/* Finding Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-border/50 pb-3">
              <div className="flex items-center gap-2 flex-wrap">
                {getStatusBadge(finding.status)}
                <span className="rounded bg-accent/10 border border-accent/20 px-2 py-0.5 text-[11px] font-semibold text-accent">
                  {finding.category}
                </span>
                <h4 className="text-sm font-bold text-foreground">
                  {finding.section_title}
                </h4>
              </div>

              {onJumpToSection ? (
                <button
                  onClick={() => onJumpToSection(finding.section_id)}
                  className="inline-flex items-center text-xs font-medium text-accent hover:underline shrink-0"
                >
                  View Clause →
                </button>
              ) : null}
            </div>

            {/* Excerpt Box */}
            <div className="rounded-lg border border-border/80 bg-background/60 p-3 text-xs italic text-foreground/80 border-l-4 border-l-accent/60">
              &quot;{finding.clause_excerpt}&quot;
            </div>

            {/* Why It Matters & What To Clarify */}
            <div className="grid gap-4 sm:grid-cols-2 text-xs">
              <div className="space-y-1">
                <span className="font-bold text-foreground uppercase tracking-wider text-[10px] block">
                  Why It Matters
                </span>
                <p className="text-foreground/90 leading-relaxed">
                  {finding.why_it_matters}
                </p>
              </div>

              <div className="space-y-1">
                <span className="font-bold text-foreground uppercase tracking-wider text-[10px] block">
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
