"use client";

import type { DocumentExplanation } from "@/lib/ai/explanation-schema";

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
  return (
    <div className="space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-xl border border-accent/30 bg-accent/5 p-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-accent/10 border border-accent/20 px-2.5 py-0.5 text-xs font-semibold text-accent">
              Plain-Language Breakdown
            </span>
            <span className="text-xs text-muted">AI-powered Summary</span>
          </div>
          <h2 className="mt-2 text-xl font-bold tracking-tight text-foreground">
            Understanding: {documentTitle}
          </h2>
          <p className="mt-1 text-xs text-muted">
            Key obligations, payment terms, important clauses, and clarification questions extracted from your agreement text.
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
          AI-generated explanation for informational purposes only. It is not a substitute for professional legal advice.
        </p>
      </div>

      {/* Grid of Main Explanation Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Card 1: AGREEMENT SUMMARY */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-base">📋</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Agreement Summary
            </h3>
          </div>
          <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.agreement_summary}
          </p>
        </div>

        {/* Card 2: PARTIES */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">👥</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Parties
            </h3>
          </div>
          <div className="space-y-2 text-xs">
            {exp.parties.map((p, idx) => (
              <div
                key={idx}
                className="flex items-start justify-between gap-2 rounded-lg border border-border/60 bg-background/50 p-3"
              >
                <div>
                  <span className="font-bold text-foreground text-sm">{p.name}</span>
                </div>
                <span className="rounded bg-accent/10 border border-accent/20 px-2 py-0.5 text-[10px] font-medium text-accent">
                  {p.role}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 3: PAYMENT TERMS */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">💳</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Payment Terms
            </h3>
          </div>
          <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.payment_terms}
          </p>
        </div>

        {/* Card 4: KEY OBLIGATIONS */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">⚡</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Key Obligations
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-foreground/90">
            {exp.key_obligations.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-emerald-500 font-bold">•</span>
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Card 5: DURATION & TERMINATION */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">⏳</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Duration & Termination
            </h3>
          </div>
          <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.duration_and_termination}
          </p>
        </div>

        {/* Card 6: CONFIDENTIALITY */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">🔒</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Confidentiality
            </h3>
          </div>
          <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.confidentiality}
          </p>
        </div>

        {/* Card 7: INTELLECTUAL PROPERTY */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-base">💡</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Intellectual Property
            </h3>
          </div>
          <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.intellectual_property}
          </p>
        </div>

        {/* Card 8: IMPORTANT CLAUSES */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-border/60 pb-3">
            <span className="text-base">🔍</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Important Clauses ({exp.important_clauses.length})
            </h3>
          </div>
          <div className="space-y-3">
            {exp.important_clauses.map((clause, idx) => (
              <div
                key={idx}
                className="rounded-lg border border-border/80 bg-background/60 p-4 space-y-2 transition-colors hover:border-accent/30"
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-foreground">
                    {clause.section_title}
                  </h4>
                  {onJumpToSection ? (
                    <button
                      onClick={() => onJumpToSection(clause.section_id)}
                      className="rounded bg-accent/10 border border-accent/20 px-2 py-0.5 text-[10px] font-mono text-accent hover:underline"
                    >
                      Section ID: {clause.section_id} →
                    </button>
                  ) : (
                    <span className="rounded bg-muted/10 border border-border px-2 py-0.5 text-[10px] font-mono text-muted">
                      ID: {clause.section_id}
                    </span>
                  )}
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed">
                  {clause.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Card 9: QUESTIONS TO CLARIFY */}
        <div className="md:col-span-2 rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-indigo-500/20 pb-3">
            <span className="text-base">❓</span>
            <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">
              Questions to Clarify Before Signing
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-indigo-900 dark:text-indigo-200">
            {exp.clarification_questions.map((q, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="font-bold text-indigo-500">•</span>
                <span className="leading-relaxed">{q}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
