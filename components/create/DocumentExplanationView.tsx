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
    <div className="space-y-8">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-2xl border border-accent/30 bg-accent/5 p-7 sm:p-8 shadow-xs">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
            Understand this agreement
          </h2>
          <p className="mt-1.5 text-base text-muted">
            A simple explanation of what you&apos;re signing ({documentTitle}).
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
          AI-generated explanation for informational purposes only. It is not a substitute for professional legal advice.
        </p>
      </div>

      {/* Grid of Main Explanation Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Card 1: WHAT IS THIS AGREEMENT ABOUT? */}
        <div className="md:col-span-2 rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-3 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            What is this agreement about?
          </h3>
          <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.agreement_summary}
          </p>
        </div>

        {/* Card 2: PARTIES */}
        <div className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Parties
          </h3>
          <div className="space-y-3 text-base">
            {exp.parties.map((p, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-3 rounded-xl border border-border bg-surface-inset p-4"
              >
                <span className="font-semibold text-foreground">{p.name}</span>
                <span className="rounded-md bg-accent/10 px-2.5 py-0.5 text-xs font-bold text-accent">
                  {p.role}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 3: PAYMENT */}
        <div className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Payment
          </h3>
          <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.payment_terms}
          </p>
        </div>

        {/* Card 4: KEY TERMS & OBLIGATIONS */}
        <div className="md:col-span-2 rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Key Terms & Obligations
          </h3>
          <ul className="space-y-3 text-base text-foreground/90">
            {exp.key_obligations.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
                <span className="text-emerald-500 font-bold">•</span>
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Card 5: DURATION & TERMINATION */}
        <div className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Duration & Termination
          </h3>
          <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.duration_and_termination}
          </p>
        </div>

        {/* Card 6: CONFIDENTIALITY */}
        <div className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Confidentiality
          </h3>
          <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.confidentiality}
          </p>
        </div>

        {/* Card 7: INTELLECTUAL PROPERTY */}
        <div className="md:col-span-2 rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 card-hover shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            Intellectual Property
          </h3>
          <p className="text-base text-foreground/90 leading-relaxed whitespace-pre-line">
            {exp.intellectual_property}
          </p>
        </div>

        {/* Card 8: IMPORTANT CLAUSES */}
        <div className="md:col-span-2 rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-5 card-hover shadow-xs">
          <div className="border-b border-border pb-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
              Important Clauses ({exp.important_clauses.length})
            </h3>
          </div>
          <div className="space-y-4">
            {exp.important_clauses.map((clause, idx) => (
              <div
                key={idx}
                className="rounded-xl border border-border bg-surface-inset p-5 space-y-2.5 transition-colors hover:border-accent/40"
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-base font-bold text-foreground">
                    {clause.section_title}
                  </h4>
                  {onJumpToSection ? (
                    <button
                      onClick={() => onJumpToSection(clause.section_id)}
                      className="text-xs font-bold text-accent hover:underline"
                    >
                      View Clause →
                    </button>
                  ) : null}
                </div>
                <p className="text-base text-foreground/90 leading-relaxed">
                  {clause.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Card 9: QUESTIONS TO CLARIFY */}
        <div className="md:col-span-2 rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-7 sm:p-8 space-y-5 card-hover shadow-xs">
          <div className="border-b border-indigo-500/20 pb-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">
              Questions to Clarify Before Signing
            </h3>
          </div>
          <ul className="space-y-3 text-base text-indigo-950 dark:text-indigo-200">
            {exp.clarification_questions.map((q, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
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
