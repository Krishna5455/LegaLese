"use client";

import { useMemo, useState } from "react";

import { FindingCard } from "@/components/dashboard/FindingCard";
import { SearchFilterBar } from "@/components/dashboard/SearchFilterBar";
import type { DetailedAnalysis, RiskLevel } from "@/types/analysis";

type FilterOption = "all" | RiskLevel;
type Tab = "summary" | "findings" | "clauses" | "keyTerms" | "obligations";

type ContractDetailWorkspaceProps = {
  analysis: DetailedAnalysis;
};

export function ContractDetailWorkspace({
  analysis,
}: ContractDetailWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<Tab>("summary");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRisk, setSelectedRisk] = useState<FilterOption>("all");

  const query = searchQuery.trim().toLowerCase();

  // Compute risk counts across all findings
  const riskCounts = useMemo(() => {
    const counts: Record<FilterOption, number> = {
      all: analysis.findings.length,
      high: 0,
      medium: 0,
      low: 0,
      informational: 0,
    };

    analysis.findings.forEach((f) => {
      if (counts[f.risk_level] != null) {
        counts[f.risk_level]++;
      }
    });

    return counts;
  }, [analysis.findings]);

  // Filter findings based on search and selected risk level
  const filteredFindings = useMemo(() => {
    return analysis.findings.filter((f) => {
      // Risk filter match
      if (selectedRisk !== "all" && f.risk_level !== selectedRisk) {
        return false;
      }

      // Search query match
      if (query) {
        const inCategory = f.category.toLowerCase().includes(query);
        const inExplanation = f.explanation.toLowerCase().includes(query);
        const inWhyMatters = (f.why_it_matters || "").toLowerCase().includes(query);
        const inClauseText = (f.clause?.text || "").toLowerCase().includes(query);
        const inClauseSection = (f.clause?.section || "").toLowerCase().includes(query);
        const inQuestions = (f.questions || []).some((q) =>
          q.toLowerCase().includes(query),
        );

        return (
          inCategory ||
          inExplanation ||
          inWhyMatters ||
          inClauseText ||
          inClauseSection ||
          inQuestions
        );
      }

      return true;
    });
  }, [analysis.findings, selectedRisk, query]);

  // Filter clauses by search query
  const filteredClauses = useMemo(() => {
    if (!query) return analysis.clauses;
    return analysis.clauses.filter(
      (c) =>
        c.section.toLowerCase().includes(query) ||
        (c.clause_number || "").toLowerCase().includes(query) ||
        c.text.toLowerCase().includes(query),
    );
  }, [analysis.clauses, query]);

  // Filter key terms by search query
  const filteredKeyTerms = useMemo(() => {
    if (!query) return analysis.key_terms;
    return analysis.key_terms.filter(
      (kt) =>
        kt.term.toLowerCase().includes(query) ||
        kt.value.toLowerCase().includes(query) ||
        (kt.clause?.text || "").toLowerCase().includes(query),
    );
  }, [analysis.key_terms, query]);

  // Filter obligations by search query
  const filteredObligations = useMemo(() => {
    if (!query) return analysis.obligations;
    return analysis.obligations.filter(
      (o) =>
        o.description.toLowerCase().includes(query) ||
        (o.responsible_party || "").toLowerCase().includes(query) ||
        (o.deadline || "").toLowerCase().includes(query) ||
        (o.clause?.text || "").toLowerCase().includes(query),
    );
  }, [analysis.obligations, query]);

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "summary", label: "Executive Summary" },
    { id: "findings", label: "Findings", count: filteredFindings.length },
    { id: "clauses", label: "Extracted Clauses", count: filteredClauses.length },
    { id: "keyTerms", label: "Defined Terms", count: filteredKeyTerms.length },
    { id: "obligations", label: "Obligations", count: filteredObligations.length },
  ];

  return (
    <div className="space-y-6">
      {/* Search & Filter Controls */}
      <SearchFilterBar
        searchQuery={searchQuery}
        onSearchChange={(q) => {
          setSearchQuery(q);
          if (q.trim() && activeTab === "summary") {
            setActiveTab("findings");
          }
        }}
        selectedRisk={selectedRisk}
        onRiskChange={(r) => {
          setSelectedRisk(r);
          if (r !== "all" && activeTab === "summary") {
            setActiveTab("findings");
          }
        }}
        riskCounts={riskCounts}
      />

      {/* Main Workspace Container */}
      <div className="rounded-xl border border-border bg-background shadow-xs overflow-hidden print:border-none print:shadow-none">
        {/* Workspace Tab Bar */}
        <div className="flex overflow-x-auto border-b border-border bg-surface print:hidden">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveTab(t.id)}
              className={`flex shrink-0 items-center gap-2 px-5 py-3.5 text-sm font-medium transition-colors ${
                activeTab === t.id
                  ? "border-b-2 border-accent text-accent font-semibold bg-background"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {t.label}
              {t.count != null && (
                <span
                  className={`rounded-full px-2 py-0.5 text-xs ${
                    activeTab === t.id
                      ? "bg-accent/10 text-accent font-semibold"
                      : "bg-border text-muted"
                  }`}
                >
                  {t.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Workspace Content */}
        <div className="p-6 md:p-8">
          {/* Executive Summary */}
          {activeTab === "summary" && (
            <div className="space-y-6 max-w-3xl">
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  Executive Summary
                </h2>
                <p className="mt-3 text-sm text-foreground leading-relaxed">
                  {analysis.summary ?? "No executive summary available."}
                </p>
              </div>

              {/* Quick stats cards */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 pt-2">
                <div className="rounded-lg border border-border bg-surface p-3.5 text-center">
                  <p className="text-2xl font-bold text-foreground">
                    {analysis.findings.length}
                  </p>
                  <p className="text-xs text-muted">Findings</p>
                </div>
                <div className="rounded-lg border border-border bg-surface p-3.5 text-center">
                  <p className="text-2xl font-bold text-foreground">
                    {analysis.clauses.length}
                  </p>
                  <p className="text-xs text-muted">Clauses</p>
                </div>
                <div className="rounded-lg border border-border bg-surface p-3.5 text-center">
                  <p className="text-2xl font-bold text-foreground">
                    {analysis.key_terms.length}
                  </p>
                  <p className="text-xs text-muted">Defined Terms</p>
                </div>
                <div className="rounded-lg border border-border bg-surface p-3.5 text-center">
                  <p className="text-2xl font-bold text-foreground">
                    {analysis.obligations.length}
                  </p>
                  <p className="text-xs text-muted">Obligations</p>
                </div>
              </div>

              <div className="rounded-lg border border-border bg-surface p-4">
                <p className="text-xs text-muted leading-relaxed">
                  <strong>Legal Disclaimer:</strong> LegaLese is an automated
                  contract understanding application. It provides plain-language
                  summaries and risk highlights for informational purposes only.
                  LegaLese is not a law firm and does not provide legal advice.
                </p>
              </div>
            </div>
          )}

          {/* Findings */}
          {activeTab === "findings" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-foreground">
                  Risk Findings & Issues ({filteredFindings.length})
                </h2>
                {selectedRisk !== "all" && (
                  <span className="text-xs text-muted">
                    Showing {selectedRisk.toUpperCase()} risk findings
                  </span>
                )}
              </div>

              {filteredFindings.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-8 text-center">
                  <p className="text-sm font-medium text-foreground">
                    No matching findings found
                  </p>
                  <p className="mt-1 text-xs text-muted">
                    Try adjusting your search query or risk filter.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredFindings.map((finding) => (
                    <FindingCard key={finding.id} finding={finding} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Extracted Clauses */}
          {activeTab === "clauses" && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">
                Extracted Contract Clauses ({filteredClauses.length})
              </h2>

              {filteredClauses.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-8 text-center">
                  <p className="text-sm font-medium text-foreground">
                    No matching clauses found
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredClauses.map((clause) => (
                    <div
                      key={clause.id}
                      className="rounded-lg border border-border bg-background p-4 space-y-2"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/40 pb-2">
                        <span className="text-xs font-semibold text-accent">
                          {clause.section}
                        </span>
                        <div className="flex items-center gap-3 text-xs text-muted">
                          {clause.clause_number && (
                            <span>Clause {clause.clause_number}</span>
                          )}
                          {clause.page_number != null && (
                            <span>Page {clause.page_number}</span>
                          )}
                        </div>
                      </div>
                      <p className="text-xs font-mono text-foreground bg-surface p-3 rounded border border-border/50 leading-relaxed">
                        {clause.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Defined Terms */}
          {activeTab === "keyTerms" && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">
                Defined Key Terms ({filteredKeyTerms.length})
              </h2>

              {filteredKeyTerms.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-8 text-center">
                  <p className="text-sm font-medium text-foreground">
                    No matching key terms found
                  </p>
                </div>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  {filteredKeyTerms.map((kt) => (
                    <div
                      key={kt.id}
                      className="rounded-lg border border-border bg-background p-4 space-y-2"
                    >
                      <h3 className="text-sm font-semibold text-foreground">
                        {kt.term}
                      </h3>
                      <p className="text-sm text-muted">{kt.value}</p>
                      {kt.clause && (
                        <p className="text-xs text-muted/70 italic border-l-2 border-accent/30 pl-2 mt-2">
                          &ldquo;{kt.clause.text}&rdquo; ({kt.clause.section})
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
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">
                Duties & Obligations ({filteredObligations.length})
              </h2>

              {filteredObligations.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-8 text-center">
                  <p className="text-sm font-medium text-foreground">
                    No matching obligations found
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredObligations.map((o) => (
                    <div
                      key={o.id}
                      className="rounded-lg border border-border bg-background p-4 space-y-2"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        {o.responsible_party && (
                          <span className="rounded bg-accent/10 px-2.5 py-0.5 text-xs font-semibold text-accent">
                            {o.responsible_party}
                          </span>
                        )}
                        {o.deadline && (
                          <span className="rounded bg-yellow-50 text-yellow-800 border border-yellow-200 px-2 py-0.5 text-xs">
                            Deadline: {o.deadline}
                          </span>
                        )}
                      </div>
                      <p className="text-sm font-medium text-foreground">
                        {o.description}
                      </p>
                      {o.clause && (
                        <p className="text-xs text-muted/70 italic border-l-2 border-accent/30 pl-2 mt-1">
                          &ldquo;{o.clause.text}&rdquo; ({o.clause.section})
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
    </div>
  );
}
