"use client";

import { useState } from "react";
import { FindingCard } from "@/components/dashboard/FindingCard";
import { getRiskLabel } from "@/lib/ai/scorer";
import type { DetailedAnalysis } from "@/types/analysis";
import {
  FileText,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Info,
} from "lucide-react";

function RiskScoreBadge({ riskScore }: { riskScore: number | null }) {
  const { label, level } = getRiskLabel(riskScore);
  const colorMap: Record<string, { bg: string; icon: React.ReactNode }> = {
    informational: {
      bg: "bg-indigo-50 text-indigo-700 border-indigo-200",
      icon: <Info className="w-3.5 h-3.5 text-indigo-600" />,
    },
    low: {
      bg: "bg-emerald-50 text-emerald-700 border-emerald-200",
      icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
    },
    medium: {
      bg: "bg-amber-50 text-amber-700 border-amber-200",
      icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />,
    },
    high: {
      bg: "bg-rose-50 text-rose-700 border-rose-200",
      icon: <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />,
    },
  };

  const item = colorMap[level] ?? colorMap.low;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-0.5 text-xs font-semibold ${item.bg}`}
    >
      {item.icon}
      <span>{label}</span>
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
    <div className="mt-4 rounded-xl border border-border bg-background overflow-hidden shadow-xs">
      {/* Panel header */}
      <div className="border-b border-border bg-surface px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm font-bold text-foreground">
            Contract Audit Overview
          </span>
          <RiskScoreBadge riskScore={analysis.risk_score} />
          <span className="text-xs font-medium bg-accent-soft text-accent border border-accent/20 px-2 py-0.5 rounded">
            Commercial Analysis
          </span>
          {analysis.created_at && (
            <span className="ml-auto text-xs text-muted font-mono">
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
            className={`flex shrink-0 items-center gap-1.5 px-4 py-2.5 text-xs font-semibold transition-colors cursor-pointer ${
              activeTab === tab.id
                ? "border-b-2 border-accent text-accent bg-background"
                : "text-secondary hover:text-foreground hover:bg-slate-50"
            }`}
          >
            <span>{tab.label}</span>
            {tabCount[tab.id] != null && (
              <span
                className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                  activeTab === tab.id
                    ? "bg-accent/10 text-accent"
                    : "bg-slate-100 text-secondary"
                }`}
              >
                {tabCount[tab.id]}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-4 sm:p-5">
        {/* Summary */}
        {activeTab === "summary" && (
          <div className="space-y-4">
            <p className="text-sm text-foreground leading-relaxed">
              {analysis.summary ?? "No summary available."}
            </p>
            <div className="rounded-lg border border-border bg-surface p-3 text-xs text-secondary leading-relaxed">
              <strong className="text-foreground">Legal Notice:</strong> LegaLese provides automated contract analysis for informational guidance. It does not constitute legal representation.
            </div>
          </div>
        )}

        {/* Findings */}
        {activeTab === "findings" && (
          <div className="space-y-3">
            {analysis.findings.length === 0 ? (
              <p className="text-xs text-muted">No findings identified.</p>
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
              <p className="text-xs text-muted">No clauses extracted.</p>
            ) : (
              analysis.clauses.map((clause) => (
                <div
                  key={clause.id}
                  className="rounded-lg border border-border bg-background p-3.5 space-y-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-accent break-words min-w-0 max-w-full flex items-center gap-1">
                      <FileText className="w-3 h-3 text-secondary" />
                      <span>{clause.section}</span>
                    </span>

                    <div className="flex items-center gap-2 text-xs text-muted">
                      {clause.clause_number && (
                        <span>Clause {clause.clause_number}</span>
                      )}
                      {clause.page_number != null && (
                        <span>Page {clause.page_number}</span>
                      )}
                    </div>
                  </div>
                  <p className="text-xs font-mono text-secondary bg-surface p-2.5 rounded border border-border/50 leading-relaxed">
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
              <p className="text-xs text-muted">No key terms identified.</p>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                {analysis.key_terms.map((kt) => (
                  <div
                    key={kt.id}
                    className="rounded-lg border border-border bg-background p-3 space-y-1"
                  >
                    <p className="text-xs font-bold text-foreground">
                      {kt.term}
                    </p>
                    <p className="text-xs text-secondary leading-relaxed">{kt.value}</p>
                    {kt.clause && (
                      <p className="text-[11px] text-muted italic border-l-2 border-accent/30 pl-2 mt-1">
                        &ldquo;{kt.clause.text}&rdquo;
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Obligations */}
        {activeTab === "obligations" && (
          <div className="space-y-3">
            {analysis.obligations.length === 0 ? (
              <p className="text-xs text-muted">No obligations identified.</p>
            ) : (
              <div className="space-y-2.5">
                {analysis.obligations.map((obl) => (
                  <div
                    key={obl.id}
                    className="rounded-lg border border-border bg-background p-3 space-y-1"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      {obl.responsible_party && (
                        <span className="rounded bg-accent-soft text-accent border border-accent/20 px-2 py-0.5 text-[11px] font-semibold">
                          {obl.responsible_party}
                        </span>
                      )}
                      {obl.deadline && (
                        <span className="rounded bg-amber-50 text-amber-800 border border-amber-200 px-2 py-0.5 text-[11px] font-medium">
                          Deadline: {obl.deadline}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-foreground font-medium">{obl.description}</p>
                    {obl.clause && (
                      <p className="text-[11px] text-muted italic border-l-2 border-accent/30 pl-2">
                        &ldquo;{obl.clause.text}&rdquo;
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
