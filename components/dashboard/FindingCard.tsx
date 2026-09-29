"use client";

import { useState } from "react";
import type { FindingWithClause, RiskLevel } from "@/types/analysis";
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Info,
  Copy,
  Check,
  ArrowRight,
  HelpCircle,
  FileText,
} from "lucide-react";

type RiskConfig = {
  label: string;
  badge: string;
  borderLeft: string;
  icon: React.ReactNode;
};

const RISK_CONFIG: Record<RiskLevel, RiskConfig> = {
  informational: {
    label: "Informational",
    badge: "bg-indigo-50 text-indigo-700 border-indigo-200",
    borderLeft: "border-l-indigo-500",
    icon: <Info className="w-3.5 h-3.5 text-indigo-600" />,
  },
  low: {
    label: "Low Risk",
    badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
    borderLeft: "border-l-emerald-500",
    icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
  },
  medium: {
    label: "Needs Attention",
    badge: "bg-amber-50 text-amber-700 border-amber-200",
    borderLeft: "border-l-amber-500",
    icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />,
  },
  high: {
    label: "Potential Concern",
    badge: "bg-rose-50 text-rose-700 border-rose-200",
    borderLeft: "border-l-rose-500",
    icon: <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />,
  },
};

type FindingCardProps = {
  finding: FindingWithClause;
  onViewInContract?: (clauseId: string) => void;
};

export function FindingCard({ finding, onViewInContract }: FindingCardProps) {
  const [copied, setCopied] = useState(false);
  const config = RISK_CONFIG[finding.risk_level] ?? RISK_CONFIG.informational;
  const clause = finding.clause;

  const handleCopyFinding = async () => {
    try {
      const parts = [
        `[${finding.risk_level.toUpperCase()}] ${finding.category}`,
        `===========================================`,
        `Explanation: ${finding.explanation}`,
      ];

      if (finding.why_it_matters) {
        parts.push(`Why It Matters: ${finding.why_it_matters}`);
      }

      if (finding.questions && finding.questions.length > 0) {
        parts.push(`Questions to Consider:`);
        finding.questions.forEach((q, i) => parts.push(`  ${i + 1}. ${q}`));
      }

      if (clause) {
        parts.push(`Clause: "${clause.text}" (${clause.section})`);
      }

      await navigator.clipboard.writeText(parts.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy finding", e);
    }
  };

  return (
    <article
      className={`rounded-xl border border-border bg-surface p-5 space-y-4 border-l-4 ${config.borderLeft} shadow-xs transition-all card-hover`}
    >
      {/* Header row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border/70 pb-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${config.badge}`}
          >
            {config.icon}
            <span>{config.label}</span>
          </span>

          <span className="rounded-md bg-slate-100 border border-slate-200/80 px-2 py-0.5 text-xs font-semibold text-secondary">
            {finding.category}
          </span>

          {finding.confidence != null && (
            <span className="text-[11px] font-mono text-muted">
              Confidence {Math.round(finding.confidence * 100)}%
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 self-end sm:self-auto">
          <button
            type="button"
            onClick={handleCopyFinding}
            title="Copy finding details"
            className="inline-flex items-center gap-1 rounded-lg border border-border bg-surface px-2.5 py-1 text-xs font-medium text-secondary hover:text-foreground hover:bg-slate-50 transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-accent" />
                <span className="text-accent text-[11px]">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-muted" />
                <span className="text-[11px]">Copy</span>
              </>
            )}
          </button>

          {clause && onViewInContract && (
            <button
              type="button"
              onClick={() => onViewInContract(clause.id)}
              className="inline-flex items-center gap-1 rounded-lg bg-accent/10 border border-accent/20 px-2.5 py-1 text-xs font-semibold text-accent hover:bg-accent/20 transition-colors cursor-pointer"
            >
              <span>View in contract</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Primary Explanation */}
      <div className="space-y-1">
        <p className="text-sm font-normal text-foreground leading-relaxed">
          {finding.explanation}
        </p>
      </div>

      {/* Linked Clause Excerpt Block */}
      {clause ? (
        <blockquote className="rounded-lg border border-border bg-background/60 p-3.5 space-y-1.5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-accent">
            <span className="font-semibold break-words min-w-0 max-w-full flex items-center gap-1.5">
              <FileText className="w-3 h-3 text-secondary" />
              <span>{clause.section}</span>
            </span>
            <div className="flex items-center gap-2 text-muted">
              {clause.clause_number && <span>Clause {clause.clause_number}</span>}
              {clause.page_number != null && <span>Page {clause.page_number}</span>}
            </div>
          </div>
          <p className="text-xs font-mono text-secondary leading-relaxed italic border-l-2 border-slate-300 pl-2.5">
            &ldquo;{clause.text}&rdquo;
          </p>
        </blockquote>
      ) : (
        <div className="flex items-center gap-2 text-[11px] font-mono text-muted italic">
          <Info className="w-3 h-3 text-muted/80" />
          <span>General contract finding (no specific clause excerpt pinned).</span>
        </div>
      )}

      {/* Two-Column Grid: Why it Matters & Questions to Clarify */}
      <div className="grid gap-3.5 sm:grid-cols-2 text-xs pt-1">
        {/* Why it matters */}
        {finding.why_it_matters ? (
          <div className="rounded-lg border border-amber-200/80 bg-amber-50/50 p-3 text-xs text-amber-950 space-y-1 leading-relaxed">
            <div className="flex items-center gap-1.5 font-bold text-amber-900 text-[11px] uppercase tracking-wider">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
              <span>Why It Matters</span>
            </div>
            <p className="text-xs text-amber-900/90 leading-relaxed font-normal">
              {finding.why_it_matters}
            </p>
          </div>
        ) : null}

        {/* Questions to consider */}
        {finding.questions && finding.questions.length > 0 ? (
          <div className="rounded-lg border border-accent/20 bg-accent-soft/30 p-3 space-y-1.5">
            <div className="flex items-center gap-1.5 text-[11px] font-bold text-accent uppercase tracking-wider">
              <HelpCircle className="w-3.5 h-3.5 text-accent" />
              <span>Questions to Clarify</span>
            </div>
            <ul className="space-y-1 text-xs text-foreground/90 leading-relaxed">
              {finding.questions.map((q, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <span className="text-accent font-bold mt-0.5">•</span>
                  <span>{q}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </article>
  );
}
