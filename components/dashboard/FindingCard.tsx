import type { Finding, RiskLevel } from "@/types/analysis";

type RiskConfig = {
  label: string;
  badge: string;
  icon: string;
};

const RISK_CONFIG: Record<RiskLevel, RiskConfig> = {
  info: {
    label: "Info",
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
  critical: {
    label: "Critical",
    badge: "bg-red-50 text-red-700 border-red-200",
    icon: "✕",
  },
};

type FindingCardProps = {
  finding: Finding;
};

export function FindingCard({ finding }: FindingCardProps) {
  const config = RISK_CONFIG[finding.risk_level] ?? RISK_CONFIG.info;

  return (
    <div className="rounded-lg border border-border bg-background p-4">
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
        {finding.confidence && (
          <span className="ml-auto text-xs text-muted">
            Confidence: {finding.confidence}
          </span>
        )}
      </div>

      {/* Explanation */}
      <p className="mt-2 text-sm text-foreground">{finding.explanation}</p>

      {/* Why it matters */}
      {finding.why_it_matters && (
        <p className="mt-1.5 text-sm text-muted italic">
          {finding.why_it_matters}
        </p>
      )}

      {/* Evidence block */}
      {finding.evidence_text && (
        <blockquote className="mt-3 rounded border-l-4 border-accent/30 bg-surface pl-3 pr-2 py-2">
          <p className="text-xs text-muted leading-relaxed">
            &ldquo;{finding.evidence_text}&rdquo;
          </p>
          {(finding.page_number != null || finding.source_section) && (
            <footer className="mt-1 text-xs text-muted/70">
              {finding.source_section && (
                <span>{finding.source_section}</span>
              )}
              {finding.source_section && finding.page_number != null && (
                <span> · </span>
              )}
              {finding.page_number != null && (
                <span>Page {finding.page_number}</span>
              )}
            </footer>
          )}
        </blockquote>
      )}

      {/* No evidence note */}
      {!finding.evidence_text && (
        <p className="mt-2 text-xs text-muted/60 italic">
          No specific passage cited for this finding.
        </p>
      )}
    </div>
  );
}
