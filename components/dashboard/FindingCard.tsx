import type { FindingWithClause, RiskLevel } from "@/types/analysis";

type RiskConfig = {
  label: string;
  badge: string;
  borderLeft: string;
  icon: string;
};

const RISK_CONFIG: Record<RiskLevel, RiskConfig> = {
  informational: {
    label: "Informational",
    badge: "bg-sky-500/10 text-sky-400 border-sky-500/25",
    borderLeft: "border-l-sky-500",
    icon: "ℹ",
  },
  low: {
    label: "Low Risk",
    badge: "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
    borderLeft: "border-l-emerald-500",
    icon: "✓",
  },
  medium: {
    label: "Medium Risk",
    badge: "bg-amber-500/10 text-amber-400 border-amber-500/25",
    borderLeft: "border-l-amber-500",
    icon: "⚠",
  },
  high: {
    label: "High Risk",
    badge: "bg-red-500/10 text-red-400 border-red-500/25",
    borderLeft: "border-l-red-500",
    icon: "▲",
  },
};

type FindingCardProps = {
  finding: FindingWithClause;
};

export function FindingCard({ finding }: FindingCardProps) {
  const config = RISK_CONFIG[finding.risk_level] ?? RISK_CONFIG.informational;
  const clause = finding.clause;

  return (
    <div
      className={`rounded-lg border border-border bg-surface p-5 space-y-3.5 border-l-4 ${config.borderLeft} shadow-xs transition-colors hover:border-border-strong`}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/40 pb-3">
        <div className="flex items-center gap-2.5">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${config.badge}`}
          >
            <span aria-hidden="true" className="text-[10px]">{config.icon}</span>
            {config.label}
          </span>
          <h3 className="text-sm font-bold text-foreground tracking-tight">
            {finding.category}
          </h3>
        </div>

        {finding.confidence != null && (
          <span className="text-[11px] font-mono text-subtle">
            Confidence {Math.round(finding.confidence * 100)}%
          </span>
        )}
      </div>

      {/* Explanation */}
      <p className="text-xs sm:text-sm text-foreground leading-relaxed">
        {finding.explanation}
      </p>

      {/* Why it matters */}
      {finding.why_it_matters && (
        <div className="rounded border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/90 leading-relaxed">
          <span className="font-semibold text-amber-400">Why it matters: </span>
          {finding.why_it_matters}
        </div>
      )}

      {/* Questions to consider */}
      {finding.questions && finding.questions.length > 0 && (
        <div className="rounded-lg border border-accent/25 bg-accent/5 p-3.5 space-y-2">
          <p className="text-[11px] font-bold text-accent uppercase tracking-wider">
            Questions to Ask Counterparty / Attorney
          </p>
          <ul className="list-disc list-inside text-xs text-foreground space-y-1.5 leading-relaxed">
            {finding.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Linked Clause Evidence */}
      {clause ? (
        <blockquote className="rounded-lg border border-border bg-surface-inset p-3.5 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-accent">
            <span className="font-semibold break-words min-w-0 max-w-full">
              {clause.section}
            </span>
            <div className="flex items-center gap-2 text-subtle">
              {clause.clause_number && <span>Clause {clause.clause_number}</span>}
              {clause.page_number != null && <span>Page {clause.page_number}</span>}
            </div>
          </div>
          <p className="text-xs font-mono text-muted leading-relaxed italic">
            &ldquo;{clause.text}&rdquo;
          </p>
        </blockquote>
      ) : (
        <p className="text-[11px] font-mono text-subtle italic">
          No specific clause quote linked.
        </p>
      )}
    </div>
  );
}
