import type { FindingWithClause, RiskLevel } from "@/types/analysis";

type RiskConfig = {
  label: string;
  badge: string;
  icon: string;
};

const RISK_CONFIG: Record<RiskLevel, RiskConfig> = {
  informational: {
    label: "Informational",
    badge: "bg-blue-50 text-blue-700 border-blue-200",
    icon: "ℹ",
  },
  low: {
    label: "Low",
    badge: "bg-green-50 text-green-700 border-green-200",
    icon: "✓",
  },
  medium: {
    label: "Medium",
    badge: "bg-yellow-50 text-yellow-800 border-yellow-200",
    icon: "⚠",
  },
  high: {
    label: "High",
    badge: "bg-orange-50 text-orange-700 border-orange-200",
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
    <div className="rounded-lg border border-border bg-background p-4 space-y-3">
      {/* Header row */}
      <div className="flex flex-wrap items-start gap-2">
        <span
          className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${config.badge}`}
        >
          <span aria-hidden="true">{config.icon}</span>
          {config.label}
        </span>
        <span className="text-sm font-semibold text-foreground">
          {finding.category}
        </span>
        {finding.confidence != null && (
          <span className="ml-auto text-xs text-muted">
            Confidence: {Math.round(finding.confidence * 100)}%
          </span>
        )}
      </div>

      {/* Explanation */}
      <p className="text-sm text-foreground">{finding.explanation}</p>

      {/* Why it matters */}
      {finding.why_it_matters && (
        <p className="text-sm text-muted italic">
          {finding.why_it_matters}
        </p>
      )}

      {/* Questions to consider */}
      {finding.questions && finding.questions.length > 0 && (
        <div className="rounded-md border border-accent/20 bg-accent/5 p-3 space-y-1.5">
          <p className="text-xs font-semibold text-accent uppercase tracking-wider">
            Questions to Ask / Consider
          </p>
          <ul className="list-disc list-inside text-xs text-foreground space-y-1">
            {finding.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Linked Clause Evidence */}
      {clause ? (
        <blockquote className="rounded border-l-4 border-accent/30 bg-surface pl-3 pr-2 py-2">
          <p className="text-xs text-muted leading-relaxed">
            &ldquo;{clause.text}&rdquo;
          </p>
          <footer className="mt-1 text-xs text-muted/70 flex flex-wrap gap-2">
            <span>{clause.section}</span>
            {clause.clause_number && <span>· Clause {clause.clause_number}</span>}
            {clause.page_number != null && <span>· Page {clause.page_number}</span>}
          </footer>
        </blockquote>
      ) : (
        <p className="text-xs text-muted/60 italic">
          No specific clause linked for this finding.
        </p>
      )}
    </div>
  );
}
