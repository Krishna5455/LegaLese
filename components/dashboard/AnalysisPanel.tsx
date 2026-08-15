"use client";

import { useState } from "react";

import { FindingCard } from "@/components/dashboard/FindingCard";
import type { AnalysisWithDetails, OverallRisk } from "@/types/analysis";

// ─── Overall risk badge ───────────────────────────────────────────────────────

const RISK_BADGE: Record<
  OverallRisk,
  { label: string; classes: string }
> = {
  low: {
    label: "Low Risk",
    classes: "bg-green-50 text-green-700 border-green-200",
  },
  medium: {
    label: "Medium Risk",
    classes: "bg-yellow-50 text-yellow-800 border-yellow-200",
  },
  high: {
    label: "High Risk",
    classes: "bg-orange-50 text-orange-700 border-orange-200",
  },
  critical: {
    label: "Critical Risk",
    classes: "bg-red-50 text-red-700 border-red-200",
  },
};

function OverallRiskBadge({ risk }: { risk: OverallRisk | null | undefined }) {
  if (!risk) return null;
  const cfg = RISK_BADGE[risk];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${cfg.classes}`}
    >
      {cfg.label}
    </span>
  );
}

// ─── Tab navigation ───────────────────────────────────────────────────────────

type Tab = "summary" | "findings" | "keyTerms" | "obligations" | "questions";

const TABS: { id: Tab; label: string }[] = [
  { id: "summary", label: "Summary" },
  { id: "findings", label: "Findings" },
  { id: "keyTerms", label: "Key Terms" },
  { id: "obligations", label: "Obligations" },
  { id: "questions", label: "Questions" },
];

// ─── Main panel ───────────────────────────────────────────────────────────────

type AnalysisPanelProps = {
  analysis: AnalysisWithDetails;
};

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("summary");

  const tabCount: Record<Tab, number | null> = {
    summary: null,
    findings: analysis.findings.length,
    keyTerms: analysis.key_terms.length,
    obligations: analysis.obligations.length,
    questions: analysis.questions.length,
  };

  function formatDate(iso: string | null | undefined) {
    if (!iso) return null;
    try {
      return new Intl.DateTimeFormat("en-US", {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(iso));
    } catch {
      return null;
    }
  }

  return (
    <div className="mt-4 rounded-xl border border-border bg-background overflow-hidden">
      {/* Panel header */}
      <div className="border-b border-border bg-surface px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm font-semibold text-foreground">
            AI Analysis
          </span>
          <OverallRiskBadge risk={analysis.overall_risk} />
          {analysis.analyzed_at && (
            <span className="ml-auto text-xs text-muted">
              {formatDate(analysis.analyzed_at)}
            </span>
          )}
        </div>
        {analysis.was_truncated && (
          <p className="mt-1 text-xs text-yellow-700 bg-yellow-50 rounded px-2 py-1 mt-2">
            ⚠ This document was very long. Only the first portion was analyzed.
          </p>
        )}
      </div>

      {/* Tabs */}
      <div className="flex overflow-x-auto border-b border-border bg-surface">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex shrink-0 items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "border-b-2 border-accent text-accent"
                : "text-muted hover:text-foreground"
            }`}
          >
            {tab.label}
            {tabCount[tab.id] != null && (
              <span
                className={`rounded-full px-1.5 py-0.5 text-xs ${
                  activeTab === tab.id
                    ? "bg-accent/10 text-accent"
                    : "bg-border text-muted"
                }`}
              >
                {tabCount[tab.id]}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-4">
        {/* Summary */}
        {activeTab === "summary" && (
          <div className="space-y-3">
            <p className="text-sm text-foreground leading-relaxed">
              {analysis.summary ?? "No summary available."}
            </p>
            <div className="rounded-lg border border-border bg-surface p-3">
              <p className="text-xs text-muted">
                <strong>Important:</strong> LegaLese helps you understand
                contracts — it is not a lawyer and does not provide legal
                advice. For significant decisions, please consult a qualified
                legal professional.
              </p>
            </div>
          </div>
        )}

        {/* Findings */}
        {activeTab === "findings" && (
          <div className="space-y-3">
            {analysis.findings.length === 0 ? (
              <p className="text-sm text-muted">No findings identified.</p>
            ) : (
              analysis.findings.map((finding) => (
                <FindingCard key={finding.id} finding={finding} />
              ))
            )}
          </div>
        )}

        {/* Key Terms */}
        {activeTab === "keyTerms" && (
          <div className="space-y-3">
            {analysis.key_terms.length === 0 ? (
              <p className="text-sm text-muted">No key terms identified.</p>
            ) : (
              <dl className="space-y-3">
                {analysis.key_terms.map((kt) => (
                  <div
                    key={kt.id}
                    className="rounded-lg border border-border bg-background p-3"
                  >
                    <dt className="text-sm font-semibold text-foreground">
                      {kt.term}
                    </dt>
                    <dd className="mt-1 text-sm text-muted">{kt.definition}</dd>
                    {(kt.page_number != null || kt.source_section) && (
                      <dd className="mt-1 text-xs text-muted/60">
                        {kt.source_section}
                        {kt.source_section && kt.page_number != null && " · "}
                        {kt.page_number != null && `Page ${kt.page_number}`}
                      </dd>
                    )}
                  </div>
                ))}
              </dl>
            )}
          </div>
        )}

        {/* Obligations */}
        {activeTab === "obligations" && (
          <div className="space-y-3">
            {analysis.obligations.length === 0 ? (
              <p className="text-sm text-muted">No obligations identified.</p>
            ) : (
              <ul className="space-y-3">
                {analysis.obligations.map((obl) => (
                  <li
                    key={obl.id}
                    className="rounded-lg border border-border bg-background p-3"
                  >
                    {obl.party && (
                      <span className="mb-1.5 inline-block rounded bg-accent/10 px-2 py-0.5 text-xs font-semibold text-accent">
                        {obl.party}
                      </span>
                    )}
                    <p className="text-sm text-foreground">{obl.description}</p>
                    {(obl.page_number != null || obl.source_section) && (
                      <p className="mt-1 text-xs text-muted/60">
                        {obl.source_section}
                        {obl.source_section && obl.page_number != null && " · "}
                        {obl.page_number != null && `Page ${obl.page_number}`}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Questions */}
        {activeTab === "questions" && (
          <div className="space-y-3">
            {analysis.questions.length === 0 ? (
              <p className="text-sm text-muted">No questions suggested.</p>
            ) : (
              <ul className="space-y-3">
                {analysis.questions.map((q, idx) => (
                  <li
                    key={q.id}
                    className="rounded-lg border border-border bg-background p-3"
                  >
                    <div className="flex gap-2">
                      <span className="shrink-0 font-semibold text-accent text-sm">
                        {idx + 1}.
                      </span>
                      <div>
                        <p className="text-sm font-medium text-foreground">
                          {q.question_text}
                        </p>
                        {q.context && (
                          <p className="mt-1 text-xs text-muted">{q.context}</p>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
