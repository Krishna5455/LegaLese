"use client";

import { useState } from "react";

import { FindingCard } from "@/components/dashboard/FindingCard";
import { getRiskLabel } from "@/lib/ai/scorer";
import type { DetailedAnalysis } from "@/types/analysis";

function RiskScoreBadge({ riskScore }: { riskScore: number | null }) {
  const { label, level } = getRiskLabel(riskScore);
  const colorMap: Record<string, string> = {
    informational: "bg-blue-50 text-blue-700 border-blue-200",
    low: "bg-green-50 text-green-700 border-green-200",
    medium: "bg-yellow-50 text-yellow-800 border-yellow-200",
    high: "bg-orange-50 text-orange-700 border-orange-200",
  };


  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${
        colorMap[level] ?? colorMap.low
      }`}
    >
      {label}
    </span>
  );
}

type Tab = "summary" | "findings" | "clauses" | "keyTerms" | "obligations";

const TABS: { id: Tab; label: string }[] = [
  { id: "summary", label: "Summary" },
  { id: "findings", label: "Findings" },
  { id: "clauses", label: "Clauses" },
  { id: "keyTerms", label: "Key Terms" },
  { id: "obligations", label: "Obligations" },
];

type AnalysisPanelProps = {
  analysis: DetailedAnalysis;
};

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("summary");

  const tabCount: Record<Tab, number | null> = {
    summary: null,
    findings: analysis.findings.length,
    clauses: analysis.clauses.length,
    keyTerms: analysis.key_terms.length,
    obligations: analysis.obligations.length,
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
            AI Contract Analysis
          </span>
          <RiskScoreBadge riskScore={analysis.risk_score} />
          {analysis.model && (
            <span className="text-xs font-mono bg-border px-2 py-0.5 rounded text-muted">
              {analysis.model}
            </span>
          )}
          {analysis.created_at && (
            <span className="ml-auto text-xs text-muted">
              {formatDate(analysis.created_at)}
            </span>
          )}
        </div>
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
                <strong>Legal Disclaimer:</strong> LegaLese helps you understand
                contracts in plain language — it is not a law firm and does not
                provide legal advice. For important legal decisions, consult a
                qualified attorney.
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

        {/* Clauses */}
        {activeTab === "clauses" && (
          <div className="space-y-3">
            {analysis.clauses.length === 0 ? (
              <p className="text-sm text-muted">No clauses extracted.</p>
            ) : (
              analysis.clauses.map((clause) => (
                <div
                  key={clause.id}
                  className="rounded-lg border border-border bg-background p-3.5 space-y-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-accent break-words min-w-0 max-w-full">
                      {clause.section}
                    </span>

                    {clause.clause_number && (
                      <span className="text-xs text-muted">
                        Clause {clause.clause_number}
                      </span>
                    )}
                    {clause.page_number != null && (
                      <span className="text-xs text-muted ml-auto">
                        Page {clause.page_number}
                      </span>
                    )}
                  </div>
                  <p className="text-xs font-mono text-foreground bg-surface p-2.5 rounded border border-border/50 leading-relaxed">
                    {clause.text}
                  </p>
                </div>
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
                    <dd className="mt-1 text-sm text-muted">{kt.value}</dd>
                    {kt.clause && (
                      <dd className="mt-2 text-xs text-muted/70 italic border-l-2 border-accent/20 pl-2">
                        &ldquo;{kt.clause.text}&rdquo; ({kt.clause.section})
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
                    className="rounded-lg border border-border bg-background p-3 space-y-1.5"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      {obl.responsible_party && (
                        <span className="rounded bg-accent/10 px-2 py-0.5 text-xs font-semibold text-accent">
                          {obl.responsible_party}
                        </span>
                      )}
                      {obl.deadline && (
                        <span className="rounded bg-yellow-50 text-yellow-800 border border-yellow-200 px-2 py-0.5 text-xs">
                          Deadline: {obl.deadline}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-foreground">{obl.description}</p>
                    {obl.clause && (
                      <p className="text-xs text-muted/70 italic border-l-2 border-accent/20 pl-2">
                        &ldquo;{obl.clause.text}&rdquo; ({obl.clause.section})
                      </p>
                    )}
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
