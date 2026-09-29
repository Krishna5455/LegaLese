"use client";

import { useState, useMemo } from "react";
import type { DocumentReview, ReviewStatus } from "@/lib/ai/review-schema";
import { SpotlightCard } from "@/components/ui/SpotlightCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { AnimatedList } from "@/components/ui/AnimatedList";
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Search,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  ArrowRight,
  Info,
  FileText,
} from "lucide-react";

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
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [copiedFindingId, setCopiedFindingId] = useState<string | null>(null);

  // Accordion state: expand all by default for high & medium risk, or allow toggling
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>(() => {
    const map: Record<string, boolean> = {};
    review.findings.forEach((f, idx) => {
      map[`finding-${idx}`] = f.status !== "clear" || idx < 3;
    });
    return map;
  });

  const clearCount = review.findings.filter((f) => f.status === "clear").length;
  const attentionCount = review.findings.filter((f) => f.status === "attention").length;
  const concernCount = review.findings.filter(
    (f) => f.status === "potential_concern",
  ).length;
  const totalCount = review.findings.length;

  const clearPct = totalCount > 0 ? (clearCount / totalCount) * 100 : 0;
  const attentionPct = totalCount > 0 ? (attentionCount / totalCount) * 100 : 0;
  const concernPct = totalCount > 0 ? (concernCount / totalCount) * 100 : 0;

  // Extract unique categories
  const categories = useMemo(() => {
    const set = new Set<string>();
    review.findings.forEach((f) => {
      if (f.category) set.add(f.category);
    });
    return Array.from(set).sort();
  }, [review.findings]);

  // Filtered findings
  const filteredFindings = useMemo(() => {
    return review.findings.filter((f) => {
      // Status filter
      if (selectedStatusFilter !== "all" && f.status !== selectedStatusFilter) {
        return false;
      }
      // Category filter
      if (selectedCategory !== "all" && f.category !== selectedCategory) {
        return false;
      }
      // Text search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesTitle = f.section_title.toLowerCase().includes(q);
        const matchesCategory = f.category.toLowerCase().includes(q);
        const matchesExcerpt = f.clause_excerpt.toLowerCase().includes(q);
        const matchesWhy = f.why_it_matters.toLowerCase().includes(q);
        const matchesClarify = f.what_to_clarify.toLowerCase().includes(q);
        if (!matchesTitle && !matchesCategory && !matchesExcerpt && !matchesWhy && !matchesClarify) {
          return false;
        }
      }
      return true;
    });
  }, [review.findings, selectedStatusFilter, selectedCategory, searchQuery]);

  const toggleCard = (id: string) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleAllCards = (expand: boolean) => {
    const next: Record<string, boolean> = {};
    review.findings.forEach((_, idx) => {
      next[`finding-${idx}`] = expand;
    });
    setExpandedCards(next);
  };

  const handleCopyReviewSummary = async () => {
    try {
      const summaryText = [
        `Risk Review: ${documentTitle}`,
        `===========================================`,
        `Risk Breakdown:`,
        `- Potential Concerns: ${concernCount}`,
        `- Needs Attention: ${attentionCount}`,
        `- Clear: ${clearCount}`,
        ``,
        `OVERVIEW:`,
        review.overall_summary,
        ``,
        `KEY FINDINGS & RECOMMENDATIONS:`,
        ...review.findings.map(
          (f, i) =>
            `${i + 1}. [${f.status.toUpperCase()}] ${f.section_title} (${f.category})\n   Excerpt: "${f.clause_excerpt}"\n   Why: ${f.why_it_matters}\n   Action: ${f.what_to_clarify}\n`,
        ),
      ].join("\n");

      await navigator.clipboard.writeText(summaryText);
      setCopiedSummary(true);
      setTimeout(() => setCopiedSummary(false), 2000);
    } catch (e) {
      console.error("Failed to copy review", e);
    }
  };

  const handleCopyFinding = async (findingKey: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedFindingId(findingKey);
      setTimeout(() => setCopiedFindingId(null), 2000);
    } catch (e) {
      console.error("Failed to copy finding", e);
    }
  };

  const getStatusConfig = (status: ReviewStatus) => {
    switch (status) {
      case "clear":
        return {
          label: "Clear",
          badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
          borderLeft: "border-l-emerald-500",
          icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
        };
      case "attention":
        return {
          label: "Needs Attention",
          badge: "bg-amber-50 text-amber-700 border-amber-200",
          borderLeft: "border-l-amber-500",
          icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />,
        };
      case "potential_concern":
        return {
          label: "Potential Concern",
          badge: "bg-rose-50 text-rose-700 border-rose-200",
          borderLeft: "border-l-rose-500",
          icon: <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />,
        };
    }
  };

  const allCardsExpanded = review.findings.every(
    (_, idx) => expandedCards[`finding-${idx}`],
  );

  return (
    <div className="space-y-6">
      {/* Executive Risk Scorecard & Header */}
      <div className="rounded-2xl glass-card p-6 sm:p-7">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 text-[11px] font-semibold text-indigo-700">
                <ShieldAlert className="w-3 h-3 text-indigo-600" />
                Clause Risk Assessment
              </span>

              {/* Dynamic Health Assessment Verdict */}
              {concernCount > 0 ? (
                <span className="rounded-full bg-rose-50 border border-rose-200 px-2.5 py-0.5 text-[11px] font-semibold text-rose-700">
                  Action Recommended · Concerns Found
                </span>
              ) : attentionCount > 0 ? (
                <span className="rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-[11px] font-semibold text-amber-700">
                  Review Advised · Points to Clarify
                </span>
              ) : (
                <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-700">
                  Low Risk Profile · Standard Terms
                </span>
              )}
            </div>

            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground">
              Agreement Risk Review
            </h2>
            <p className="text-sm text-secondary max-w-2xl leading-relaxed">
              Objective clause-by-clause analysis of liability, rights, and potential exposure in{" "}
              <span className="font-semibold text-foreground">&ldquo;{documentTitle}&rdquo;</span>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-2 md:pt-0">
            <button
              type="button"
              onClick={handleCopyReviewSummary}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 transition-colors shadow-2xs cursor-pointer"
            >
              {copiedSummary ? (
                <>
                  <Check className="w-3.5 h-3.5 text-accent" />
                  <span className="text-accent">Copied Report</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-secondary" />
                  <span>Copy Report</span>
                </>
              )}
            </button>

            {onReturnToDocument ? (
              <button
                type="button"
                onClick={onReturnToDocument}
                className="inline-flex items-center gap-1.5 rounded-lg bg-surface border border-border px-3.5 py-2 text-xs font-semibold text-accent hover:border-accent/40 hover:bg-accent-soft/30 transition-all shadow-2xs cursor-pointer"
              >
                <span>View Full Text</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            ) : null}
          </div>
        </div>

        {/* Visual Segmented Health Gauge */}
        <div className="mt-6 pt-6 border-t border-border space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">Clause Health Meter</span>
            <span className="text-muted font-mono">{totalCount} total clauses evaluated</span>
          </div>

          <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden flex shadow-inner">
            {clearPct > 0 && (
              <div
                style={{ width: `${clearPct}%` }}
                className="bg-emerald-500 h-full transition-all duration-500"
                title={`${clearCount} Clear clauses (${Math.round(clearPct)}%)`}
              />
            )}
            {attentionPct > 0 && (
              <div
                style={{ width: `${attentionPct}%` }}
                className="bg-amber-500 h-full transition-all duration-500"
                title={`${attentionCount} Attention clauses (${Math.round(attentionPct)}%)`}
              />
            )}
            {concernPct > 0 && (
              <div
                style={{ width: `${concernPct}%` }}
                className="bg-rose-500 h-full transition-all duration-500"
                title={`${concernCount} Potential Concerns (${Math.round(concernPct)}%)`}
              />
            )}
          </div>

          {/* Metric Status Badges Row */}
          <div className="grid grid-cols-3 gap-3 pt-3">
            <div
              onClick={() => setSelectedStatusFilter("clear")}
              className={`rounded-xl border p-3 cursor-pointer transition-all ${
                selectedStatusFilter === "clear"
                  ? "border-emerald-500 bg-emerald-50/70 shadow-xs"
                  : "border-border bg-background/50 hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-800">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Clear Clauses</span>
              </div>
              <p className="mt-1 text-lg font-bold text-emerald-950 font-mono">
                {clearCount} <span className="text-xs font-normal text-emerald-700">({Math.round(clearPct)}%)</span>
              </p>
            </div>

            <div
              onClick={() => setSelectedStatusFilter("attention")}
              className={`rounded-xl border p-3 cursor-pointer transition-all ${
                selectedStatusFilter === "attention"
                  ? "border-amber-500 bg-amber-50/70 shadow-xs"
                  : "border-border bg-background/50 hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-800">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                <span>Needs Attention</span>
              </div>
              <p className="mt-1 text-lg font-bold text-amber-950 font-mono">
                {attentionCount} <span className="text-xs font-normal text-amber-700">({Math.round(attentionPct)}%)</span>
              </p>
            </div>

            <div
              onClick={() => setSelectedStatusFilter("potential_concern")}
              className={`rounded-xl border p-3 cursor-pointer transition-all ${
                selectedStatusFilter === "potential_concern"
                  ? "border-rose-500 bg-rose-50/70 shadow-xs"
                  : "border-border bg-background/50 hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-800">
                <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
                <span>Potential Concerns</span>
              </div>
              <p className="mt-1 text-lg font-bold text-rose-950 font-mono">
                {concernCount} <span className="text-xs font-normal text-rose-700">({Math.round(concernPct)}%)</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Subtle Legal Notice Alert */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3.5 text-xs text-amber-900 flex items-start gap-3">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <p className="font-semibold text-amber-950">Informational Review Notice</p>
          <p className="text-amber-900/90 leading-relaxed">
            AI risk evaluation highlights typical contractual pitfalls and points of ambiguity. It does not replace independent legal advice or represent an attorney-client relationship.
          </p>
        </div>
      </div>

      {/* Overall Review Summary Card */}
      <SpotlightCard className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-50 text-indigo-700">
              <FileText className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Executive Review Summary
            </h3>
          </div>
          <span className="text-[11px] font-medium text-secondary">Risk Analysis Overview</span>
        </div>
        <p className="text-sm sm:text-[15px] text-foreground leading-relaxed whitespace-pre-line font-normal">
          {review.overall_summary}
        </p>
      </SpotlightCard>

      {/* Interactive Filter & Search Bar */}
      <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 space-y-4 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          {/* Status Filter Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setSelectedStatusFilter("all")}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                selectedStatusFilter === "all"
                  ? "bg-foreground text-background shadow-xs"
                  : "bg-surface border border-border text-foreground hover:bg-slate-50"
              }`}
            >
              All ({totalCount})
            </button>

            {concernCount > 0 && (
              <button
                type="button"
                onClick={() => setSelectedStatusFilter("potential_concern")}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                  selectedStatusFilter === "potential_concern"
                    ? "bg-rose-600 text-white shadow-xs"
                    : "bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100"
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                <span>Concerns ({concernCount})</span>
              </button>
            )}

            {attentionCount > 0 && (
              <button
                type="button"
                onClick={() => setSelectedStatusFilter("attention")}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                  selectedStatusFilter === "attention"
                    ? "bg-amber-600 text-white shadow-xs"
                    : "bg-amber-50 border border-amber-200 text-amber-700 hover:bg-amber-100"
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                <span>Attention ({attentionCount})</span>
              </button>
            )}

            {clearCount > 0 && (
              <button
                type="button"
                onClick={() => setSelectedStatusFilter("clear")}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                  selectedStatusFilter === "clear"
                    ? "bg-emerald-600 text-white shadow-xs"
                    : "bg-emerald-50 border border-emerald-200 text-emerald-700 hover:bg-emerald-100"
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>Clear ({clearCount})</span>
              </button>
            )}
          </div>

          {/* Search and Category Filter Controls */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Category Select */}
            {categories.length > 0 && (
              <div className="relative">
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground focus:outline-none focus:border-accent appearance-none pr-8 cursor-pointer"
                >
                  <option value="all">All Categories</option>
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
                <ChevronDown className="w-3.5 h-3.5 text-muted absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>
            )}

            {/* Keyword Search */}
            <div className="relative min-w-[180px] sm:min-w-[220px]">
              <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search findings..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
              />
            </div>

            {/* Expand / Collapse All */}
            <button
              type="button"
              onClick={() => toggleAllCards(!allCardsExpanded)}
              className="px-2.5 py-1.5 rounded-lg border border-border bg-surface hover:bg-slate-50 text-xs font-semibold text-secondary transition-colors cursor-pointer"
            >
              {allCardsExpanded ? "Collapse All" : "Expand All"}
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-muted font-mono pt-1">
          <span>
            Showing {filteredFindings.length} of {totalCount} findings
          </span>
          {(selectedStatusFilter !== "all" ||
            selectedCategory !== "all" ||
            searchQuery) && (
            <button
              type="button"
              onClick={() => {
                setSelectedStatusFilter("all");
                setSelectedCategory("all");
                setSearchQuery("");
              }}
              className="text-accent hover:underline font-medium"
            >
              Reset Filters
            </button>
          )}
        </div>
      </div>

      {/* Findings List */}
      {filteredFindings.length === 0 ? (
        <EmptyState
          title="No findings match your filter"
          description="Try adjusting your status filter, selecting all categories, or clearing your search term."
          action={
            <button
              type="button"
              onClick={() => {
                setSelectedStatusFilter("all");
                setSelectedCategory("all");
                setSearchQuery("");
              }}
              className="rounded-lg bg-surface border border-border px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 transition-colors"
            >
              Reset All Filters
            </button>
          }
        />
      ) : (
        <AnimatedList className="space-y-4">
          {filteredFindings.map((finding, idx) => {
            const cardKey = `finding-${idx}`;
            const isExpanded = expandedCards[cardKey] ?? true;
            const config = getStatusConfig(finding.status);
            const findingClipboardText = `[${finding.status.toUpperCase()}] ${finding.section_title} (${finding.category})\nExcerpt: "${finding.clause_excerpt}"\nWhy It Matters: ${finding.why_it_matters}\nWhat To Clarify: ${finding.what_to_clarify}`;

            return (
              <article
                key={cardKey}
                className={`rounded-xl border border-border bg-surface transition-all shadow-xs border-l-4 ${config.borderLeft} card-hover`}
              >
                {/* Header Row */}
                <div
                  onClick={() => toggleCard(cardKey)}
                  className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 p-5 cursor-pointer select-none border-b border-border/60"
                >
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span
                      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${config.badge}`}
                    >
                      {config.icon}
                      <span>{config.label}</span>
                    </span>

                    <span className="rounded-md bg-slate-100 border border-slate-200 px-2 py-0.5 text-xs font-medium text-secondary">
                      {finding.category}
                    </span>

                    <h4 className="text-sm font-bold text-foreground">
                      {finding.section_title}
                    </h4>
                  </div>

                  <div className="flex items-center gap-2 self-end sm:self-auto">
                    {/* Copy finding action */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCopyFinding(cardKey, findingClipboardText);
                      }}
                      title="Copy finding details"
                      className="p-1.5 rounded-lg border border-border bg-surface hover:bg-slate-50 text-secondary transition-colors"
                    >
                      {copiedFindingId === cardKey ? (
                        <Check className="w-3.5 h-3.5 text-accent" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>

                    {/* Jump to section */}
                    {onJumpToSection && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onJumpToSection(finding.section_id);
                        }}
                        className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline px-2.5 py-1 rounded hover:bg-accent-soft/30 transition-colors"
                      >
                        <span>View Section</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    )}

                    <button
                      type="button"
                      aria-label="Toggle details"
                      className="text-secondary hover:text-foreground"
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Card Body */}
                {isExpanded && (
                  <div className="p-5 sm:p-6 space-y-4">
                    {/* Excerpt Blockquote */}
                    <div className="rounded-xl border border-border bg-background/50 p-4 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px] font-mono text-muted uppercase tracking-wider">
                        <span>Original Clause Excerpt</span>
                        <span>§ {finding.section_title}</span>
                      </div>
                      <p className="text-xs sm:text-[13px] font-mono text-secondary leading-relaxed italic border-l-2 border-slate-300 pl-3">
                        &ldquo;{finding.clause_excerpt}&rdquo;
                      </p>
                    </div>

                    {/* Two-Column Analysis Breakdown */}
                    <div className="grid gap-4 sm:grid-cols-2 text-xs">
                      <div className="rounded-xl border border-border/80 bg-background/40 p-4 space-y-2">
                        <span className="font-bold text-foreground uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                          <span>Why It Matters</span>
                        </span>
                        <p className="text-secondary leading-relaxed text-xs sm:text-[13px]">
                          {finding.why_it_matters}
                        </p>
                      </div>

                      <div className="rounded-xl border border-border/80 bg-background/40 p-4 space-y-2">
                        <span className="font-bold text-foreground uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                          <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
                          <span>Recommended Action / What to Clarify</span>
                        </span>
                        <p className="text-secondary leading-relaxed text-xs sm:text-[13px]">
                          {finding.what_to_clarify}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </AnimatedList>
      )}
    </div>
  );
}
