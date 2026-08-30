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
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-xl border border-accent/20 bg-accent-soft p-5 sm:p-6 shadow-xs">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
            Plain-Language Breakdown
          </h2>
          <p className="mt-1 text-xs sm:text-sm text-secondary">
            Understand key terms and obligations for &quot;{documentTitle}&quot;.
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
          <span>⚠️ Informational Summary Notice</span>
        </div>
        <p className="leading-relaxed">
          AI-generated explanation for informational purposes only. Always verify binding decisions with qualified legal counsel.
        </p>
      </div>

      {/* Grid of Main Explanation Cards */}
      <div className="grid gap-5 md:grid-cols-2">
        {/* Card 1: WHAT IS THIS AGREEMENT ABOUT? */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-2.5 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Agreement Summary
          </h3>
          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.agreement_summary}
          </p>
        </div>

        {/* Card 2: PARTIES */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Parties Involved
          </h3>
          <div className="space-y-2.5">
            {exp.parties.map((p, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background p-3 text-xs"
              >
                <span className="font-semibold text-foreground">{p.name}</span>
                <span className="rounded bg-accent-soft border border-accent/20 px-2 py-0.5 font-semibold text-accent">
                  {p.role}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 3: PAYMENT */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Payment Terms
          </h3>
          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.payment_terms}
          </p>
        </div>

        {/* Card 4: KEY TERMS & OBLIGATIONS */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Key Obligations
          </h3>
          <ul className="space-y-2 text-sm text-foreground">
            {exp.key_obligations.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-emerald-600 font-bold">•</span>
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Card 5: DURATION & TERMINATION */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Duration & Termination
          </h3>
          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.duration_and_termination}
          </p>
        </div>

        {/* Card 6: CONFIDENTIALITY */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Confidentiality
          </h3>
          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.confidentiality}
          </p>
        </div>

        {/* Card 7: INTELLECTUAL PROPERTY */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Intellectual Property Rights
          </h3>
          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">
            {exp.intellectual_property}
          </p>
        </div>

        {/* Card 8: IMPORTANT CLAUSES */}
        <div className="md:col-span-2 rounded-xl border border-border bg-surface p-6 space-y-4 card-hover shadow-xs">
          <div className="border-b border-border pb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Key Section Explanations ({exp.important_clauses.length})
            </h3>
          </div>
          <div className="space-y-3">
            {exp.important_clauses.map((clause, idx) => (
              <div
                key={idx}
                className="rounded-lg border border-border bg-background p-4 space-y-2 transition-colors hover:border-slate-300"
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-foreground">
                    {clause.section_title}
                  </h4>
                  {onJumpToSection ? (
                    <button
                      type="button"
                      onClick={() => onJumpToSection(clause.section_id)}
                      className="text-xs font-semibold text-accent hover:underline"
                    >
                      View Clause →
                    </button>
                  ) : null}
                </div>
                <p className="text-xs text-secondary leading-relaxed">
                  {clause.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Card 9: QUESTIONS TO CLARIFY */}
        <div className="md:col-span-2 rounded-xl border border-indigo-200 bg-indigo-50/60 p-6 space-y-3 card-hover shadow-xs">
          <div className="border-b border-indigo-200/60 pb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-700">
              Clarification Questions Before Signing
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-indigo-950 font-medium">
            {exp.clarification_questions.map((q, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="font-bold text-indigo-600">•</span>
                <span className="leading-relaxed">{q}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

