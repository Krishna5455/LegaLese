"use client";

import { useState, useMemo } from "react";
import type { DocumentExplanation } from "@/lib/ai/explanation-schema";
import { SpotlightCard } from "@/components/ui/SpotlightCard";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  Sparkles,
  Users,
  CreditCard,
  Clock,
  Shield,
  FileText,
  Search,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  CheckCircle2,
  ArrowRight,
  HelpCircle,
  Info,
  CheckSquare,
} from "lucide-react";

type DocumentExplanationViewProps = {
  explanation: DocumentExplanation;
  documentTitle: string;
  onReturnToDocument?: () => void;
  onJumpToSection?: (sectionId: string) => void;
};

export function DocumentExplanationView({
  explanation: exp,
  documentTitle,
  onReturnToDocument,
  onJumpToSection,
}: DocumentExplanationViewProps) {
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [copiedQuestions, setCopiedQuestions] = useState(false);
  const [clauseSearch, setClauseSearch] = useState("");
  const [expandedClauses, setExpandedClauses] = useState<Record<string, boolean>>(() => {
    // Default: expand the first 3 clauses
    const initial: Record<string, boolean> = {};
    exp.important_clauses.forEach((c, idx) => {
      initial[c.section_id] = idx < 3;
    });
    return initial;
  });
  const [checkedQuestions, setCheckedQuestions] = useState<Record<number, boolean>>({});

  const toggleClause = (id: string) => {
    setExpandedClauses((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleAllClauses = (expand: boolean) => {
    const next: Record<string, boolean> = {};
    exp.important_clauses.forEach((c) => {
      next[c.section_id] = expand;
    });
    setExpandedClauses(next);
  };

  const toggleQuestion = (idx: number) => {
    setCheckedQuestions((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handleCopySummary = async () => {
    try {
      const summaryText = [
        `Plain-Language Summary: ${documentTitle}`,
        `===========================================`,
        ``,
        `OVERVIEW:`,
        exp.agreement_summary,
        ``,
        `PARTIES:`,
        exp.parties.map((p) => `- ${p.name} (${p.role})`).join("\n"),
        ``,
        `PAYMENT TERMS:`,
        exp.payment_terms,
        ``,
        `KEY OBLIGATIONS:`,
        exp.key_obligations.map((o) => `- ${o}`).join("\n"),
        ``,
        `INTELLECTUAL PROPERTY:`,
        exp.intellectual_property,
        ``,
        `TERMINATION & DURATION:`,
        exp.duration_and_termination,
        ``,
        `CONFIDENTIALITY:`,
        exp.confidentiality,
      ].join("\n");

      await navigator.clipboard.writeText(summaryText);
      setCopiedSummary(true);
      setTimeout(() => setCopiedSummary(false), 2000);
    } catch (e) {
      console.error("Failed to copy summary", e);
    }
  };

  const handleCopyQuestions = async () => {
    try {
      const qText = [
        `Pre-Signing Questions for "${documentTitle}":`,
        ``,
        exp.clarification_questions.map((q, i) => `${i + 1}. ${q}`).join("\n"),
      ].join("\n");

      await navigator.clipboard.writeText(qText);
      setCopiedQuestions(true);
      setTimeout(() => setCopiedQuestions(false), 2000);
    } catch (e) {
      console.error("Failed to copy questions", e);
    }
  };

  // Filter important clauses
  const filteredClauses = useMemo(() => {
    if (!clauseSearch.trim()) return exp.important_clauses;
    const query = clauseSearch.toLowerCase();
    return exp.important_clauses.filter(
      (c) =>
        c.section_title.toLowerCase().includes(query) ||
        c.explanation.toLowerCase().includes(query),
    );
  }, [exp.important_clauses, clauseSearch]);

  const questionsAnsweredCount = Object.values(checkedQuestions).filter(Boolean).length;
  const totalQuestions = exp.clarification_questions.length;
  const allClausesExpanded = exp.important_clauses.every(
    (c) => expandedClauses[c.section_id],
  );

  return (
    <div className="space-y-6">
      {/* Executive Header Banner */}
      <div className="rounded-2xl glass-card p-6 sm:p-7">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 rounded-full bg-accent-soft border border-accent/20 px-2.5 py-0.5 text-[11px] font-semibold text-accent">
                <Sparkles className="w-3 h-3" />
                Plain-Language Breakdown
              </span>
              <span className="text-xs text-muted font-medium">
                AI Synthesized · Informational
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground">
              Understand Your Agreement
            </h2>
            <p className="text-sm text-secondary max-w-2xl leading-relaxed">
              Clear, non-technical explanation of terms, rights, and responsibilities in{" "}
              <span className="font-semibold text-foreground">&ldquo;{documentTitle}&rdquo;</span>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-2 md:pt-0">
            <button
              type="button"
              onClick={handleCopySummary}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 transition-colors shadow-2xs cursor-pointer"
            >
              {copiedSummary ? (
                <>
                  <Check className="w-3.5 h-3.5 text-accent" />
                  <span className="text-accent">Copied Summary</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-secondary" />
                  <span>Copy Summary</span>
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

        {/* Metric Overview Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-6 border-t border-border mt-6">
          <div className="rounded-xl border border-border/80 bg-background/60 p-3.5 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted font-medium">
              <Users className="w-3.5 h-3.5 text-secondary" />
              <span>Parties</span>
            </div>
            <p className="text-lg font-bold text-foreground">
              {exp.parties.length}{" "}
              <span className="text-xs font-normal text-secondary">
                {exp.parties.length === 1 ? "Party" : "Parties"}
              </span>
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-background/60 p-3.5 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
              <span>Obligations</span>
            </div>
            <p className="text-lg font-bold text-foreground">
              {exp.key_obligations.length}{" "}
              <span className="text-xs font-normal text-secondary">Identified</span>
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-background/60 p-3.5 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted font-medium">
              <FileText className="w-3.5 h-3.5 text-secondary" />
              <span>Key Clauses</span>
            </div>
            <p className="text-lg font-bold text-foreground">
              {exp.important_clauses.length}{" "}
              <span className="text-xs font-normal text-secondary">Explained</span>
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-background/60 p-3.5 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted font-medium">
              <HelpCircle className="w-3.5 h-3.5 text-amber-600" />
              <span>Checkpoints</span>
            </div>
            <p className="text-lg font-bold text-foreground">
              {exp.clarification_questions.length}{" "}
              <span className="text-xs font-normal text-secondary">To Verify</span>
            </p>
          </div>
        </div>
      </div>

      {/* Subtle Legal Notice Alert */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3.5 text-xs text-amber-900 flex items-start gap-3">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <p className="font-semibold text-amber-950">Informational Explanation Notice</p>
          <p className="text-amber-900/90 leading-relaxed">
            This plain-language summary is generated by AI to help you understand terms quickly. It does not constitute formal legal advice. For binding or complex negotiations, consult legal counsel.
          </p>
        </div>
      </div>

      {/* Core Summary Card with Spotlight */}
      <SpotlightCard className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent">
              <FileText className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Agreement Summary
            </h3>
          </div>
          <span className="text-[11px] font-medium text-secondary">Executive Overview</span>
        </div>
        <p className="text-sm sm:text-[15px] text-foreground leading-relaxed whitespace-pre-line font-normal">
          {exp.agreement_summary}
        </p>
      </SpotlightCard>

      {/* Two-Column Grid: Parties & Payment */}
      <div className="grid gap-5 md:grid-cols-2">
        {/* Parties Involved */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4 shadow-xs card-hover">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-secondary">
                <Users className="w-4 h-4" />
              </div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
                Parties Involved
              </h3>
            </div>
            <span className="text-xs text-muted font-medium font-mono">
              {exp.parties.length} signatories
            </span>
          </div>

          <div className="space-y-2.5">
            {exp.parties.map((party, idx) => {
              const initials = party.name
                .split(" ")
                .map((n) => n[0])
                .join("")
                .slice(0, 2)
                .toUpperCase() || "PA";

              return (
                <div
                  key={idx}
                  className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-3.5 transition-colors hover:border-slate-300"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-200 text-foreground font-semibold text-xs border border-border">
                      {initials}
                    </div>
                    <div className="min-w-0">
                      <p className="font-semibold text-foreground text-sm truncate">
                        {party.name}
                      </p>
                      <p className="text-[11px] text-secondary">Designated Signatory</p>
                    </div>
                  </div>
                  <span className="shrink-0 rounded-md bg-accent-soft border border-accent/20 px-2.5 py-1 text-xs font-semibold text-accent">
                    {party.role}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Payment Terms */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4 shadow-xs card-hover">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-50 text-accent">
                <CreditCard className="w-4 h-4" />
              </div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
                Payment Terms
              </h3>
            </div>
            <span className="text-xs text-accent font-semibold font-mono">
              Financial Terms
            </span>
          </div>

          <div className="rounded-lg border border-border/80 bg-background/50 p-4 space-y-2">
            <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
              {exp.payment_terms}
            </p>
          </div>
        </div>
      </div>

      {/* Key Obligations Checklist */}
      <div className="rounded-xl border border-border bg-surface p-6 sm:p-7 space-y-4 shadow-xs card-hover">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-50 text-accent">
              <CheckCircle2 className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Key Obligations & Deliverables
            </h3>
          </div>
          <span className="text-xs text-secondary font-medium font-mono">
            {exp.key_obligations.length} primary duties
          </span>
        </div>

        <div className="grid gap-2.5 sm:grid-cols-2">
          {exp.key_obligations.map((obligation, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 rounded-lg border border-border bg-background/50 p-3.5 text-xs text-foreground transition-all hover:bg-background hover:border-slate-300"
            >
              <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-bold mt-0.5">
                ✓
              </div>
              <span className="leading-relaxed font-medium">{obligation}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Three-Column Grid: Duration, Confidentiality, IP */}
      <div className="grid gap-5 md:grid-cols-3">
        {/* Duration & Termination */}
        <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 space-y-3 shadow-xs card-hover">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-secondary">
              <Clock className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Duration & Termination
            </h3>
          </div>
          <p className="text-xs sm:text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.duration_and_termination}
          </p>
        </div>

        {/* Confidentiality */}
        <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 space-y-3 shadow-xs card-hover">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-secondary">
              <Shield className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Confidentiality
            </h3>
          </div>
          <p className="text-xs sm:text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.confidentiality}
          </p>
        </div>

        {/* Intellectual Property */}
        <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 space-y-3 shadow-xs card-hover">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-secondary">
              <Sparkles className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Intellectual Property
            </h3>
          </div>
          <p className="text-xs sm:text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.intellectual_property}
          </p>
        </div>
      </div>

      {/* Important Clauses Detailed Section */}
      <div className="rounded-xl border border-border bg-surface p-6 sm:p-7 space-y-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-border pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-secondary" />
              <h3 className="text-base font-bold text-foreground">
                Key Section Explanations
              </h3>
            </div>
            <p className="text-xs text-secondary">
              Deep dive into specific sections with instant links to the original agreement text.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            {/* Search filter input */}
            <div className="relative min-w-[200px] sm:min-w-[240px]">
              <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Filter clauses..."
                value={clauseSearch}
                onChange={(e) => setClauseSearch(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
              />
            </div>

            {/* Expand / Collapse All Button */}
            <button
              type="button"
              onClick={() => toggleAllClauses(!allClausesExpanded)}
              className="px-2.5 py-1.5 rounded-lg border border-border bg-surface hover:bg-slate-50 text-xs font-semibold text-secondary transition-colors cursor-pointer"
            >
              {allClausesExpanded ? "Collapse All" : "Expand All"}
            </button>
          </div>
        </div>

        {/* Clause List */}
        {filteredClauses.length === 0 ? (
          <EmptyState
            title="No clauses match your search"
            description={`No clauses found matching "${clauseSearch}". Try searching for another keyword or clear the search.`}
            action={
              <button
                type="button"
                onClick={() => setClauseSearch("")}
                className="rounded-lg bg-surface border border-border px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-slate-50"
              >
                Clear Search
              </button>
            }
          />
        ) : (
          <div className="space-y-3">
            {filteredClauses.map((clause) => {
              const isExpanded = expandedClauses[clause.section_id] ?? true;

              return (
                <div
                  key={clause.section_id}
                  className="rounded-xl border border-border bg-background/40 transition-all duration-200 hover:border-slate-300"
                >
                  {/* Clickable Header */}
                  <div
                    onClick={() => toggleClause(clause.section_id)}
                    className="flex items-center justify-between p-4 cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-slate-200/80 text-foreground font-mono font-semibold text-[11px]">
                        §
                      </div>
                      <h4 className="text-sm font-semibold text-foreground">
                        {clause.section_title}
                      </h4>
                    </div>

                    <div className="flex items-center gap-3">
                      {onJumpToSection ? (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onJumpToSection(clause.section_id);
                          }}
                          className="hidden sm:inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline px-2 py-1 rounded hover:bg-accent-soft/40 transition-colors"
                        >
                          <span>View Section</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      ) : null}

                      <button
                        type="button"
                        aria-label="Toggle clause"
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

                  {/* Body */}
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-1 border-t border-border/60 text-xs sm:text-sm text-secondary leading-relaxed space-y-3">
                      <p className="whitespace-pre-line text-foreground/90">
                        {clause.explanation}
                      </p>

                      {onJumpToSection && (
                        <div className="pt-2 sm:hidden">
                          <button
                            type="button"
                            onClick={() => onJumpToSection(clause.section_id)}
                            className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline"
                          >
                            <span>View clause in original agreement</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Pre-Signing Clarification Checklist */}
      <div className="rounded-xl border border-indigo-200/80 bg-indigo-50/40 p-6 sm:p-7 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-indigo-200/60 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-indigo-700" />
              <h3 className="text-sm sm:text-base font-bold text-indigo-950">
                Pre-Signing Clarification Checklist
              </h3>
            </div>
            <p className="text-xs text-indigo-900/80 mt-1">
              Verify these key questions with the other party or your legal advisor prior to execution.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopyQuestions}
              className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-white px-3 py-1.5 text-xs font-semibold text-indigo-900 hover:bg-indigo-50 transition-colors shadow-2xs cursor-pointer"
            >
              {copiedQuestions ? (
                <>
                  <Check className="w-3.5 h-3.5 text-accent" />
                  <span className="text-accent">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Copy Questions</span>
                </>
              )}
            </button>

            <span className="rounded-full bg-white border border-indigo-200 px-2.5 py-1 text-xs font-mono font-semibold text-indigo-800">
              {questionsAnsweredCount}/{totalQuestions} Checked
            </span>
          </div>
        </div>

        <div className="space-y-2.5">
          {exp.clarification_questions.map((question, idx) => {
            const isChecked = !!checkedQuestions[idx];

            return (
              <div
                key={idx}
                onClick={() => toggleQuestion(idx)}
                className={`flex items-start gap-3 rounded-lg border p-3.5 text-xs transition-all cursor-pointer select-none ${
                  isChecked
                    ? "border-emerald-200 bg-emerald-50/50 text-emerald-950 opacity-80"
                    : "border-indigo-200/80 bg-white text-indigo-950 hover:border-indigo-300"
                }`}
              >
                <div className="mt-0.5">
                  {isChecked ? (
                    <CheckSquare className="w-4 h-4 text-emerald-600" />
                  ) : (
                    <div className="w-4 h-4 rounded border border-indigo-300 bg-white" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <span
                    className={`leading-relaxed font-medium ${
                      isChecked ? "line-through text-emerald-800" : ""
                    }`}
                  >
                    {question}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
