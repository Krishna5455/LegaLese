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
    badge: "bg-indigo-50 text-indigo-700 border-indigo-200",
    borderLeft: "border-l-indigo-500",
    icon: "ℹ",
  },
  low: {
    label: "Low Risk",
    badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
    borderLeft: "border-l-emerald-500",
    icon: "✓",
  },
  medium: {
    label: "Medium Risk",
    badge: "bg-amber-50 text-amber-700 border-amber-200",
    borderLeft: "border-l-amber-500",
    icon: "⚠",
  },
  high: {
    label: "High Risk",
    badge: "bg-rose-50 text-rose-700 border-rose-200",
    borderLeft: "border-l-rose-500",
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
      className={`rounded-xl border border-border bg-surface p-5 space-y-3 border-l-4 ${config.borderLeft} shadow-xs transition-all card-hover`}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
        <div className="flex items-center gap-2.5">
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${config.badge}`}
          >
            <span aria-hidden="true" className="text-[10px]">{config.icon}</span>
            {config.label}
          </span>
          <h3 className="text-sm font-bold text-foreground">
            {finding.category}
          </h3>
        </div>

        {finding.confidence != null && (
          <span className="text-[11px] font-mono text-muted">
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
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900 leading-relaxed font-medium">
          <span className="font-bold text-amber-800">Why it matters: </span>
          {finding.why_it_matters}
        </div>
      )}

      {/* Questions to consider */}
      {finding.questions && finding.questions.length > 0 && (
        <div className="rounded-lg border border-accent/20 bg-accent-soft p-3.5 space-y-1.5">
          <p className="text-[11px] font-bold text-accent uppercase tracking-wider">
            Questions to Ask Counterparty / Attorney
          </p>
          <ul className="list-disc list-inside text-xs text-foreground space-y-1 leading-relaxed">
            {finding.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Linked Clause Evidence */}
      {clause ? (
        <blockquote className="rounded-lg border border-border bg-background p-3.5 space-y-1.5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-accent">
            <span className="font-semibold break-words min-w-0 max-w-full">
              {clause.section}
            </span>
            <div className="flex items-center gap-2 text-muted">
              {clause.clause_number && <span>Clause {clause.clause_number}</span>}
              {clause.page_number != null && <span>Page {clause.page_number}</span>}
            </div>
          </div>
          <p className="text-xs font-mono text-secondary leading-relaxed italic">
            &ldquo;{clause.text}&rdquo;
          </p>
        </blockquote>
      ) : (
        <p className="text-[11px] font-mono text-muted italic">
          No specific clause quote linked.
        </p>
      )}
    </div>
  );
}

