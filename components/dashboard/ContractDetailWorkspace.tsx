"use client";

import { useMemo, useState, useRef } from "react";
import { FindingCard } from "@/components/dashboard/FindingCard";
import { SpotlightCard } from "@/components/ui/SpotlightCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { AnimatedList } from "@/components/ui/AnimatedList";
import type { DetailedAnalysis, RiskLevel } from "@/types/analysis";
import { getRiskLabel } from "@/lib/ai/scorer";
import {
  FileText,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Search,
  Copy,
  Check,
  ArrowRight,
  ChevronDown,
  Columns,
  Eye,
  HelpCircle,
  CheckSquare,
  Sparkles,
  Bookmark,
  Users,
} from "lucide-react";

type FilterOption = "all" | RiskLevel;
type Tab = "summary" | "findings" | "clauses" | "obligations" | "keyTerms";
type ViewLayout = "split" | "analysis" | "document";

type ContractDetailWorkspaceProps = {
  analysis: DetailedAnalysis;
};

export function ContractDetailWorkspace({
  analysis,
}: ContractDetailWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<Tab>("summary");
  const [layout, setLayout] = useState<ViewLayout>("split");
  const [mobilePane, setMobilePane] = useState<"analysis" | "document">("analysis");

  // Filter & Search states for Findings
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRisk, setSelectedRisk] = useState<FilterOption>("all");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  // Contract Document Viewer search state
  const [documentSearch, setDocumentSearch] = useState("");
  const [selectedDocumentSection, setSelectedDocumentSection] = useState<string>("all");

  // Highlighted clause state in Contract Document Viewer
  const [highlightedClauseId, setHighlightedClauseId] = useState<string | null>(null);
  const documentViewerRef = useRef<HTMLDivElement>(null);

  // Pre-signing checklist local state
  const [checkedQuestions, setCheckedQuestions] = useState<Record<string, boolean>>({});
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [copiedQuestions, setCopiedQuestions] = useState(false);

  // Accordion state for findings
  const [expandedFindings, setExpandedFindings] = useState<Record<string, boolean>>(() => {
    const map: Record<string, boolean> = {};
    analysis.findings.forEach((f, idx) => {
      map[f.id] = f.risk_level === "high" || f.risk_level === "medium" || idx < 3;
    });
    return map;
  });

  // Calculate Health & Risk Metrics
  const highCount = analysis.findings.filter((f) => f.risk_level === "high").length;
  const mediumCount = analysis.findings.filter((f) => f.risk_level === "medium").length;
  const lowCount = analysis.findings.filter((f) => f.risk_level === "low").length;
  const infoCount = analysis.findings.filter(
    (f) => f.risk_level === "informational",
  ).length;
  const clearCount = lowCount + infoCount;
  const totalFindings = analysis.findings.length;

  // Calculate human-readable health score out of 100
  const healthScore = useMemo(() => {
    if (totalFindings === 0) return 96;
    // Weighted risk penalty: high risk costs 22 pts, medium costs 10 pts, low costs 2 pts
    const deductions = highCount * 22 + mediumCount * 10 + lowCount * 2;
    return Math.max(32, Math.min(98, 100 - deductions));
  }, [totalFindings, highCount, mediumCount, lowCount]);

  const { label: riskLabel } = getRiskLabel(analysis.risk_score);

  // Extract unique categories for Findings filter
  const categories = useMemo(() => {
    const set = new Set<string>();
    analysis.findings.forEach((f) => {
      if (f.category) set.add(f.category);
    });
    return Array.from(set).sort();
  }, [analysis.findings]);

  // Extract unique section titles for Document Viewer quick jump
  const documentSections = useMemo(() => {
    const set = new Set<string>();
    analysis.clauses.forEach((c) => {
      if (c.section) set.add(c.section);
    });
    return Array.from(set);
  }, [analysis.clauses]);

  // Aggregate all clarification questions from findings
  const allClarificationQuestions = useMemo(() => {
    const list: { question: string; category: string }[] = [];
    analysis.findings.forEach((f) => {
      if (f.questions && f.questions.length > 0) {
        f.questions.forEach((q) => {
          if (!list.some((item) => item.question === q)) {
            list.push({ question: q, category: f.category });
          }
        });
      }
    });
    return list;
  }, [analysis.findings]);

  // Filtered findings for Findings tab
  const filteredFindings = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return analysis.findings.filter((f) => {
      // Risk filter
      if (selectedRisk !== "all") {
        if (selectedRisk === "low" && f.risk_level !== "low" && f.risk_level !== "informational") {
          return false;
        } else if (selectedRisk !== "low" && f.risk_level !== selectedRisk) {
          return false;
        }
      }
      // Category filter
      if (selectedCategory !== "all" && f.category !== selectedCategory) {
        return false;
      }
      // Search query
      if (q) {
        const inCat = f.category.toLowerCase().includes(q);
        const inExp = f.explanation.toLowerCase().includes(q);
        const inWhy = (f.why_it_matters || "").toLowerCase().includes(q);
        const inText = (f.clause?.text || "").toLowerCase().includes(q);
        const inSec = (f.clause?.section || "").toLowerCase().includes(q);
        const inQ = (f.questions || []).some((item) => item.toLowerCase().includes(q));
        if (!inCat && !inExp && !inWhy && !inText && !inSec && !inQ) {
          return false;
        }
      }
      return true;
    });
  }, [analysis.findings, selectedRisk, selectedCategory, searchQuery]);

  // Filtered clauses for Clauses tab
  const filteredClauses = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return analysis.clauses;
    return analysis.clauses.filter(
      (c) =>
        c.section.toLowerCase().includes(q) ||
        (c.clause_number || "").toLowerCase().includes(q) ||
        c.text.toLowerCase().includes(q),
    );
  }, [analysis.clauses, searchQuery]);

  // Filtered key terms for Key Terms tab
  const filteredKeyTerms = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return analysis.key_terms;
    return analysis.key_terms.filter(
      (kt) =>
        kt.term.toLowerCase().includes(q) ||
        kt.value.toLowerCase().includes(q) ||
        (kt.clause?.text || "").toLowerCase().includes(q),
    );
  }, [analysis.key_terms, searchQuery]);

  // Filtered obligations for Obligations tab
  const filteredObligations = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return analysis.obligations;
    return analysis.obligations.filter(
      (o) =>
        o.description.toLowerCase().includes(q) ||
        (o.responsible_party || "").toLowerCase().includes(q) ||
        (o.deadline || "").toLowerCase().includes(q) ||
        (o.clause?.text || "").toLowerCase().includes(q),
    );
  }, [analysis.obligations, searchQuery]);

  // Filtered clauses in the Left Pane (Contract Document Viewer)
  const viewerClauses = useMemo(() => {
    return analysis.clauses.filter((c) => {
      if (selectedDocumentSection !== "all" && c.section !== selectedDocumentSection) {
        return false;
      }
      if (documentSearch.trim()) {
        const q = documentSearch.toLowerCase();
        return (
          c.section.toLowerCase().includes(q) ||
          (c.clause_number || "").toLowerCase().includes(q) ||
          c.text.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [analysis.clauses, selectedDocumentSection, documentSearch]);

  // View In Contract action: scrolls and highlights the clause in Left Pane
  const handleViewInContract = (clauseId: string) => {
    // If mobile, switch to document pane
    setMobilePane("document");
    // If desktop is in analysis focus mode, switch to split view so user can see it
    if (layout === "analysis") {
      setLayout("split");
    }

    setHighlightedClauseId(clauseId);

    // Scroll to the targeted element smoothly
    setTimeout(() => {
      const el = document.getElementById(`contract-clause-${clauseId}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 120);

    // Fade out highlight after 4 seconds
    setTimeout(() => {
      setHighlightedClauseId((prev) => (prev === clauseId ? null : prev));
    }, 4000);
  };

  const handleCopySummary = async () => {
    try {
      const text = [
        `Executive Contract Summary`,
        `===========================`,
        analysis.summary ?? "No summary provided.",
        ``,
        `Contract Health: ${healthScore}/100 (${riskLabel})`,
        `Risk Findings: ${totalFindings} total (${highCount} concerns, ${mediumCount} attention, ${clearCount} clear)`,
      ].join("\n");
      await navigator.clipboard.writeText(text);
      setCopiedSummary(true);
      setTimeout(() => setCopiedSummary(false), 2000);
    } catch (e) {
      console.error("Failed to copy summary", e);
    }
  };

  const handleCopyQuestions = async () => {
    try {
      const text = [
        `Pre-Signing Questions to Clarify:`,
        `=================================`,
        ...allClarificationQuestions.map((item, i) => `${i + 1}. [${item.category}] ${item.question}`),
      ].join("\n");
      await navigator.clipboard.writeText(text);
      setCopiedQuestions(true);
      setTimeout(() => setCopiedQuestions(false), 2000);
    } catch (e) {
      console.error("Failed to copy questions", e);
    }
  };

  const toggleAllFindings = (expand: boolean) => {
    const next: Record<string, boolean> = {};
    analysis.findings.forEach((f) => {
      next[f.id] = expand;
    });
    setExpandedFindings(next);
  };

  const tabs: { id: Tab; label: string; count?: number; icon: React.ReactNode }[] = [
    { id: "summary", label: "Summary", icon: <Sparkles className="w-3.5 h-3.5" /> },
    {
      id: "findings",
      label: "Findings",
      count: filteredFindings.length,
      icon: <ShieldAlert className="w-3.5 h-3.5" />,
    },
    {
      id: "clauses",
      label: "Clauses",
      count: filteredClauses.length,
      icon: <FileText className="w-3.5 h-3.5" />,
    },
    {
      id: "obligations",
      label: "Obligations",
      count: filteredObligations.length,
      icon: <CheckSquare className="w-3.5 h-3.5" />,
    },
    {
      id: "keyTerms",
      label: "Key Terms",
      count: filteredKeyTerms.length,
      icon: <Bookmark className="w-3.5 h-3.5" />,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Workspace Control Bar: Layout Switcher & View Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-xl glass-card p-3 sm:px-4 sm:py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted hidden md:inline">
            Workspace View:
          </span>

          {/* Desktop Layout Buttons */}
          <div className="hidden md:flex items-center gap-1 bg-background p-1 rounded-lg border border-border">
            <button
              type="button"
              onClick={() => setLayout("split")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                layout === "split"
                  ? "bg-surface text-accent shadow-xs border border-border"
                  : "text-secondary hover:text-foreground hover:bg-slate-100"
              }`}
              title="View contract and analysis side-by-side"
            >
              <Columns className="w-3.5 h-3.5" />
              <span>Split View</span>
            </button>

            <button
              type="button"
              onClick={() => setLayout("analysis")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                layout === "analysis"
                  ? "bg-surface text-accent shadow-xs border border-border"
                  : "text-secondary hover:text-foreground hover:bg-slate-100"
              }`}
              title="Expand analysis to full width"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Analysis Focus</span>
            </button>

            <button
              type="button"
              onClick={() => setLayout("document")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                layout === "document"
                  ? "bg-surface text-accent shadow-xs border border-border"
                  : "text-secondary hover:text-foreground hover:bg-slate-100"
              }`}
              title="Read contract document full width"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Document Focus</span>
            </button>
          </div>

          {/* Mobile View Toggle */}
          <div className="flex md:hidden items-center gap-1 bg-background p-1 rounded-lg border border-border w-full">
            <button
              type="button"
              onClick={() => setMobilePane("analysis")}
              className={`flex-1 py-1.5 text-xs font-semibold rounded-md text-center transition-all cursor-pointer ${
                mobilePane === "analysis"
                  ? "bg-surface text-accent shadow-xs border border-border"
                  : "text-secondary"
              }`}
            >
              Analysis ({totalFindings})
            </button>
            <button
              type="button"
              onClick={() => setMobilePane("document")}
              className={`flex-1 py-1.5 text-xs font-semibold rounded-md text-center transition-all cursor-pointer ${
                mobilePane === "document"
                  ? "bg-surface text-accent shadow-xs border border-border"
                  : "text-secondary"
              }`}
            >
              Contract Text ({analysis.clauses.length})
            </button>
          </div>
        </div>

        {/* Health status summary pill */}
        <div className="flex items-center gap-2 self-end sm:self-auto">
          <span className="text-xs text-muted font-mono hidden lg:inline">
            Contract Health
          </span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-mono font-bold ${
              healthScore >= 75
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : healthScore >= 50
                  ? "bg-amber-50 text-amber-700 border-amber-200"
                  : "bg-rose-50 text-rose-700 border-rose-200"
            }`}
          >
            <span>{healthScore} / 100</span>
            <span className="font-sans font-semibold text-[11px]">· {riskLabel}</span>
          </span>
        </div>
      </div>

      {/* Main Two-Column Workspace Grid */}
      <div
        className={`grid gap-6 ${
          layout === "split"
            ? "lg:grid-cols-12 items-start"
            : layout === "document"
              ? "grid-cols-1"
              : "grid-cols-1"
        }`}
      >
        {/* ========================================================================= */}
        {/* LEFT PANE: CONTRACT DOCUMENT VIEWER (VISUAL ANCHOR)                      */}
        {/* ========================================================================= */}
        <aside
          ref={documentViewerRef}
          className={`space-y-4 ${
            layout === "split"
              ? "lg:col-span-5 xl:col-span-5"
              : layout === "document"
                ? "col-span-1"
                : "hidden"
          } ${mobilePane === "document" ? "block" : "hidden md:block"}`}
        >
          <div className="rounded-xl border border-border bg-surface shadow-xs overflow-hidden sticky top-6">
            {/* Document Pane Header */}
            <div className="border-b border-border p-4 space-y-3 bg-surface">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">
                      Original Contract Text
                    </h3>
                    <p className="text-[11px] text-muted">
                      {analysis.clauses.length} extracted clauses
                    </p>
                  </div>
                </div>

                {documentSections.length > 0 && (
                  <div className="relative">
                    <select
                      value={selectedDocumentSection}
                      onChange={(e) => setSelectedDocumentSection(e.target.value)}
                      className="rounded-lg border border-border bg-background px-2.5 py-1 text-[11px] font-medium text-foreground focus:outline-none focus:border-accent appearance-none pr-7 cursor-pointer max-w-[140px] truncate"
                    >
                      <option value="all">All Sections</option>
                      {documentSections.map((sec) => (
                        <option key={sec} value={sec}>
                          {sec}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="w-3 h-3 text-muted absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                )}
              </div>

              {/* In-Document Search input */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search contract text…"
                  value={documentSearch}
                  onChange={(e) => setDocumentSearch(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
                />
                {documentSearch && (
                  <button
                    type="button"
                    onClick={() => setDocumentSearch("")}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted hover:text-foreground"
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>

            {/* Rendered Contract Clauses in Scrollable Container */}
            <div className="max-h-[calc(100vh-280px)] overflow-y-auto p-4 space-y-3.5 divide-y divide-border/60">
              {viewerClauses.length === 0 ? (
                <EmptyState
                  title="No matching contract clauses"
                  description="Try clearing your search query or selecting 'All Sections'."
                  action={
                    <button
                      type="button"
                      onClick={() => {
                        setDocumentSearch("");
                        setSelectedDocumentSection("all");
                      }}
                      className="rounded-lg bg-surface border border-border px-3 py-1 text-xs font-semibold text-foreground hover:bg-slate-50"
                    >
                      Reset Filter
                    </button>
                  }
                />
              ) : (
                viewerClauses.map((clause) => {
                  const isHighlighted = highlightedClauseId === clause.id;

                  return (
                    <article
                      key={clause.id}
                      id={`contract-clause-${clause.id}`}
                      className={`pt-3.5 space-y-1.5 transition-all duration-300 rounded-lg p-2.5 ${
                        isHighlighted
                          ? "bg-accent-soft/50 ring-2 ring-accent border-accent/40 shadow-xs"
                          : "hover:bg-slate-50/70"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-muted">
                        <span className="font-semibold text-accent break-words min-w-0 max-w-full">
                          {clause.section}
                        </span>
                        <div className="flex items-center gap-2 shrink-0">
                          {clause.clause_number && (
                            <span>Clause {clause.clause_number}</span>
                          )}
                          {clause.page_number != null && (
                            <span>Page {clause.page_number}</span>
                          )}
                        </div>
                      </div>

                      <p className="text-xs font-mono text-foreground/90 leading-relaxed whitespace-pre-line">
                        {clause.text}
                      </p>
                    </article>
                  );
                })
              )}
            </div>
          </div>
        </aside>

        {/* ========================================================================= */}
        {/* RIGHT PANE: ANALYSIS WORKSPACE                                            */}
        {/* ========================================================================= */}
        <main
          className={`space-y-6 ${
            layout === "split"
              ? "lg:col-span-7 xl:col-span-7"
              : layout === "analysis"
                ? "col-span-1"
                : "hidden"
          } ${mobilePane === "analysis" ? "block" : "hidden md:block"}`}
        >
          {/* Main Analysis Tab Bar */}
          <div
            role="tablist"
            aria-label="Contract Analysis Tabs"
            className="flex items-center gap-1.5 p-1 glass-pill rounded-xl overflow-x-auto shadow-2xs"
          >
            {tabs.map((t) => {
              const isSelected = activeTab === t.id;

              return (
                <button
                  key={t.id}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => setActiveTab(t.id)}
                  className={`flex shrink-0 items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg transition-all cursor-pointer ${
                    isSelected
                      ? "bg-background text-accent shadow-xs border border-border"
                      : "text-secondary hover:text-foreground hover:bg-slate-50"
                  }`}
                >
                  {t.icon}
                  <span>{t.label}</span>
                  {t.count != null && (
                    <span
                      className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                        isSelected
                          ? "bg-accent/15 text-accent font-bold"
                          : "bg-slate-100 text-secondary"
                      }`}
                    >
                      {t.count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* TAB 1: EXECUTIVE SUMMARY */}
          {activeTab === "summary" && (
            <div className="space-y-6 animate-fadeIn">
              {/* Executive Health Scorecard */}
              <div className="rounded-2xl glass-card p-6 sm:p-7 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="rounded-full bg-accent-soft border border-accent/20 px-2.5 py-0.5 text-[11px] font-semibold text-accent flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        Executive Audit
                      </span>
                      <span className="text-xs text-muted font-medium">
                        Contract Health Assessment
                      </span>
                    </div>
                    <h2 className="text-xl sm:text-2xl font-bold text-foreground">
                      Contract Risk Health
                    </h2>
                  </div>

                  <button
                    type="button"
                    onClick={handleCopySummary}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-slate-50 transition-colors shadow-2xs self-start sm:self-auto cursor-pointer"
                  >
                    {copiedSummary ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-accent" />
                        <span className="text-accent">Copied Summary</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-secondary" />
                        <span>Copy Summary</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Score & Segmented Meter */}
                <div className="space-y-2.5 pt-2 border-t border-border">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-foreground">
                      Health Score: {healthScore} / 100
                    </span>
                    <span className="text-secondary font-medium font-mono">
                      {totalFindings} issues audited
                    </span>
                  </div>

                  {/* Visual Progress Meter */}
                  <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden flex shadow-inner">
                    {clearCount > 0 && (
                      <div
                        style={{ width: `${(clearCount / Math.max(1, totalFindings)) * 100}%` }}
                        className="bg-emerald-500 h-full transition-all duration-500"
                        title={`${clearCount} Clear clauses`}
                      />
                    )}
                    {mediumCount > 0 && (
                      <div
                        style={{ width: `${(mediumCount / Math.max(1, totalFindings)) * 100}%` }}
                        className="bg-amber-500 h-full transition-all duration-500"
                        title={`${mediumCount} Medium risk findings`}
                      />
                    )}
                    {highCount > 0 && (
                      <div
                        style={{ width: `${(highCount / Math.max(1, totalFindings)) * 100}%` }}
                        className="bg-rose-500 h-full transition-all duration-500"
                        title={`${highCount} Potential concerns`}
                      />
                    )}
                  </div>

                  {/* 3 Metric Cards */}
                  <div className="grid grid-cols-3 gap-3 pt-2">
                    <div
                      onClick={() => {
                        setSelectedRisk("low");
                        setActiveTab("findings");
                      }}
                      className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-3 cursor-pointer transition-all hover:bg-emerald-100/60"
                    >
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-800">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        <span>Clear</span>
                      </div>
                      <p className="mt-1 text-lg font-bold text-emerald-950 font-mono">
                        {clearCount}
                      </p>
                    </div>

                    <div
                      onClick={() => {
                        setSelectedRisk("medium");
                        setActiveTab("findings");
                      }}
                      className="rounded-xl border border-amber-200 bg-amber-50/50 p-3 cursor-pointer transition-all hover:bg-amber-100/60"
                    >
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-800">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                        <span>Attention</span>
                      </div>
                      <p className="mt-1 text-lg font-bold text-amber-950 font-mono">
                        {mediumCount}
                      </p>
                    </div>

                    <div
                      onClick={() => {
                        setSelectedRisk("high");
                        setActiveTab("findings");
                      }}
                      className="rounded-xl border border-rose-200 bg-rose-50/50 p-3 cursor-pointer transition-all hover:bg-rose-100/60"
                    >
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-800">
                        <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
                        <span>Concerns</span>
                      </div>
                      <p className="mt-1 text-lg font-bold text-rose-950 font-mono">
                        {highCount}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* What Should I Know Before Signing? (Executive Summary Card) */}
              <SpotlightCard className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent">
                      <FileText className="w-4 h-4" />
                    </div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
                      What Should I Know Before Signing?
                    </h3>
                  </div>
                  <span className="text-[11px] font-medium text-secondary">
                    Executive Summary
                  </span>
                </div>
                <p className="text-sm sm:text-[15px] text-foreground leading-relaxed whitespace-pre-line font-normal">
                  {analysis.summary ?? "No summary available."}
                </p>
              </SpotlightCard>

              {/* Top High-Priority Concerns Spotlight */}
              {highCount > 0 && (
                <div className="rounded-xl border border-rose-200 bg-rose-50/40 p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-sm text-rose-950">
                      <ShieldAlert className="w-4 h-4 text-rose-600" />
                      <span>Top Potential Concerns Requiring Attention</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedRisk("high");
                        setActiveTab("findings");
                      }}
                      className="text-xs font-semibold text-rose-700 hover:underline"
                    >
                      View All Concerns →
                    </button>
                  </div>

                  <div className="space-y-2">
                    {analysis.findings
                      .filter((f) => f.risk_level === "high")
                      .slice(0, 3)
                      .map((finding) => (
                        <div
                          key={finding.id}
                          onClick={() => {
                            setSelectedRisk("high");
                            setActiveTab("findings");
                            if (finding.clause) {
                              handleViewInContract(finding.clause.id);
                            }
                          }}
                          className="flex items-center justify-between gap-3 p-3 rounded-lg border border-rose-200/80 bg-white hover:border-rose-300 transition-colors cursor-pointer"
                        >
                          <div className="min-w-0">
                            <span className="rounded bg-rose-100 text-rose-800 px-2 py-0.5 text-[10px] font-bold mr-2">
                              {finding.category}
                            </span>
                            <span className="text-xs font-medium text-foreground truncate">
                              {finding.explanation}
                            </span>
                          </div>
                          <span className="text-xs font-semibold text-accent shrink-0 flex items-center gap-1">
                            <span>Review</span>
                            <ArrowRight className="w-3 h-3" />
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Pre-Signing Clarification Questions Checklist */}
              {allClarificationQuestions.length > 0 && (
                <div className="rounded-xl border border-indigo-200 bg-indigo-50/30 p-6 space-y-4 shadow-xs">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-indigo-200/60 pb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <HelpCircle className="w-4 h-4 text-indigo-700" />
                        <h3 className="text-sm sm:text-base font-bold text-indigo-950">
                          Pre-Signing Clarification Questions
                        </h3>
                      </div>
                      <p className="text-xs text-indigo-900/80 mt-0.5">
                        Ask counterparty or counsel these key points before signing.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={handleCopyQuestions}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-white px-3 py-1.5 text-xs font-semibold text-indigo-900 hover:bg-indigo-50 transition-colors shadow-2xs self-start sm:self-auto cursor-pointer"
                    >
                      {copiedQuestions ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-accent" />
                          <span className="text-accent">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-indigo-600" />
                          <span>Copy Questions</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="space-y-2.5">
                    {allClarificationQuestions.slice(0, 5).map((item, idx) => {
                      const isChecked = !!checkedQuestions[`q-${idx}`];

                      return (
                        <div
                          key={idx}
                          onClick={() =>
                            setCheckedQuestions((prev) => ({
                              ...prev,
                              [`q-${idx}`]: !prev[`q-${idx}`],
                            }))
                          }
                          className={`flex items-start gap-3 rounded-lg border p-3 text-xs transition-all cursor-pointer select-none ${
                            isChecked
                              ? "border-emerald-200 bg-emerald-50/50 text-emerald-950 opacity-80"
                              : "border-indigo-200/80 bg-white text-indigo-950 hover:border-indigo-300"
                          }`}
                        >
                          <div className="mt-0.5">
                            {isChecked ? (
                              <CheckSquare className="w-4 h-4 text-emerald-600" />
                            ) : (
                              <div className="w-4 h-4 rounded border border-indigo-300 bg-white" />
                            )}
                          </div>
                          <div className="min-w-0 flex-1">
                            <span
                              className={`leading-relaxed font-medium ${
                                isChecked ? "line-through text-emerald-800" : ""
                              }`}
                            >
                              {item.question}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: FINDINGS */}
          {activeTab === "findings" && (
            <div className="space-y-4 animate-fadeIn">
              {/* Findings Toolbar: Search, Risk Filters, Category Selector */}
              <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 space-y-4 shadow-xs">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  {/* Status pills */}
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setSelectedRisk("all")}
                      className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                        selectedRisk === "all"
                          ? "bg-foreground text-background shadow-xs"
                          : "bg-surface border border-border text-foreground hover:bg-slate-50"
                      }`}
                    >
                      All ({totalFindings})
                    </button>

                    {highCount > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedRisk("high")}
                        className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                          selectedRisk === "high"
                            ? "bg-rose-600 text-white shadow-xs"
                            : "bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100"
                        }`}
                      >
                        <ShieldAlert className="w-3.5 h-3.5" />
                        <span>Concerns ({highCount})</span>
                      </button>
                    )}

                    {mediumCount > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedRisk("medium")}
                        className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                          selectedRisk === "medium"
                            ? "bg-amber-600 text-white shadow-xs"
                            : "bg-amber-50 border border-amber-200 text-amber-700 hover:bg-amber-100"
                        }`}
                      >
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Attention ({mediumCount})</span>
                      </button>
                    )}

                    {clearCount > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedRisk("low")}
                        className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                          selectedRisk === "low"
                            ? "bg-emerald-600 text-white shadow-xs"
                            : "bg-emerald-50 border border-emerald-200 text-emerald-700 hover:bg-emerald-100"
                        }`}
                      >
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Clear ({clearCount})</span>
                      </button>
                    )}
                  </div>

                  {/* Controls: Search, Category, Expand */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Category Select */}
                    {categories.length > 0 && (
                      <div className="relative">
                        <select
                          value={selectedCategory}
                          onChange={(e) => setSelectedCategory(e.target.value)}
                          className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground focus:outline-none focus:border-accent appearance-none pr-8 cursor-pointer max-w-[150px] truncate"
                        >
                          <option value="all">All Categories</option>
                          {categories.map((cat) => (
                            <option key={cat} value={cat}>
                              {cat}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="w-3.5 h-3.5 text-muted absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                      </div>
                    )}

                    {/* Search Input */}
                    <div className="relative min-w-[160px] sm:min-w-[200px]">
                      <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        placeholder="Search findings…"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
                      />
                    </div>

                    <button
                      type="button"
                      onClick={() => toggleAllFindings(!Object.values(expandedFindings).every(Boolean))}
                      className="px-2.5 py-1.5 rounded-lg border border-border bg-surface hover:bg-slate-50 text-xs font-semibold text-secondary transition-colors cursor-pointer"
                    >
                      {Object.values(expandedFindings).every(Boolean) ? "Collapse All" : "Expand All"}
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-muted font-mono pt-1">
                  <span>
                    Showing {filteredFindings.length} of {totalFindings} findings
                  </span>
                  {(selectedRisk !== "all" || selectedCategory !== "all" || searchQuery) && (
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedRisk("all");
                        setSelectedCategory("all");
                        setSearchQuery("");
                      }}
                      className="text-accent hover:underline font-medium"
                    >
                      Reset Filters
                    </button>
                  )}
                </div>
              </div>

              {/* Findings Cards List */}
              {filteredFindings.length === 0 ? (
                <EmptyState
                  title="No findings match your filter"
                  description="Try adjusting your risk filter, clearing your search query, or selecting 'All Categories'."
                  action={
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedRisk("all");
                        setSelectedCategory("all");
                        setSearchQuery("");
                      }}
                      className="rounded-lg bg-surface border border-border px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-slate-50"
                    >
                      Reset All Filters
                    </button>
                  }
                />
              ) : (
                <AnimatedList className="space-y-4">
                  {filteredFindings.map((finding) => (
                    <FindingCard
                      key={finding.id}
                      finding={finding}
                      onViewInContract={handleViewInContract}
                    />
                  ))}
                </AnimatedList>
              )}
            </div>
          )}

          {/* TAB 3: EXTRACTED CLAUSES */}
          {activeTab === "clauses" && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
                <div>
                  <h2 className="text-base font-bold text-foreground">
                    Extracted Contract Clauses ({filteredClauses.length})
                  </h2>
                  <p className="text-xs text-secondary">
                    Structured legal sections extracted from the uploaded document.
                  </p>
                </div>

                <div className="relative min-w-[200px]">
                  <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Search clauses…"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-surface text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
                  />
                </div>
              </div>

              {filteredClauses.length === 0 ? (
                <EmptyState
                  title="No clauses match your search"
                  description="Try searching for a different clause term or clearing your search."
                  action={
                    <button
                      type="button"
                      onClick={() => setSearchQuery("")}
                      className="rounded-lg bg-surface border border-border px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-slate-50"
                    >
                      Clear Search
                    </button>
                  }
                />
              ) : (
                <div className="space-y-3.5">
                  {filteredClauses.map((clause) => {
                    const linkedFindings = analysis.findings.filter(
                      (f) => f.clause_id === clause.id,
                    );

                    return (
                      <article
                        key={clause.id}
                        className="rounded-xl border border-border bg-surface p-5 space-y-3 shadow-xs card-hover"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-3">
                          <div className="flex items-center gap-2">
                            <span className="rounded-md bg-accent-soft text-accent border border-accent/20 px-2 py-0.5 text-xs font-semibold">
                              {clause.section}
                            </span>
                            {linkedFindings.length > 0 && (
                              <span className="rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
                                {linkedFindings.length}{" "}
                                {linkedFindings.length === 1 ? "finding" : "findings"}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-3 text-xs text-muted">
                            {clause.clause_number && (
                              <span>Clause {clause.clause_number}</span>
                            )}
                            {clause.page_number != null && (
                              <span>Page {clause.page_number}</span>
                            )}
                            <button
                              type="button"
                              onClick={() => handleViewInContract(clause.id)}
                              className="inline-flex items-center gap-1 font-semibold text-accent hover:underline text-xs"
                            >
                              <span>View in contract</span>
                              <ArrowRight className="w-3 h-3" />
                            </button>
                          </div>
                        </div>

                        <p className="text-xs sm:text-[13px] font-mono text-secondary bg-background/60 p-3.5 rounded-lg border border-border leading-relaxed whitespace-pre-line">
                          {clause.text}
                        </p>
                      </article>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 4: OBLIGATIONS */}
          {activeTab === "obligations" && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
                <div>
                  <h2 className="text-base font-bold text-foreground">
                    Contractual Obligations ({filteredObligations.length})
                  </h2>
                  <p className="text-xs text-secondary">
                    Allocated responsibilities, milestones, and deliverables.
                  </p>
                </div>

                <div className="relative min-w-[200px]">
                  <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Search obligations…"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-surface text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
                  />
                </div>
              </div>

              {filteredObligations.length === 0 ? (
                <EmptyState
                  title="No obligations identified"
                  description="No specific obligations match your search or were extracted from this agreement."
                  action={
                    <button
                      type="button"
                      onClick={() => setSearchQuery("")}
                      className="rounded-lg bg-surface border border-border px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-slate-50"
                    >
                      Clear Search
                    </button>
                  }
                />
              ) : (
                <div className="space-y-3">
                  {filteredObligations.map((obl) => (
                    <article
                      key={obl.id}
                      className="rounded-xl border border-border bg-surface p-4 sm:p-5 space-y-2.5 shadow-xs card-hover"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          {obl.responsible_party ? (
                            <span className="inline-flex items-center gap-1 rounded-md bg-accent-soft text-accent border border-accent/20 px-2.5 py-0.5 text-xs font-bold">
                              <Users className="w-3 h-3" />
                              <span>{obl.responsible_party}</span>
                            </span>
                          ) : (
                            <span className="rounded-md bg-slate-100 text-secondary border border-slate-200 px-2 py-0.5 text-xs font-medium">
                              General Obligation
                            </span>
                          )}

                          {obl.deadline && (
                            <span className="rounded-md bg-amber-50 text-amber-800 border border-amber-200 px-2.5 py-0.5 text-xs font-medium">
                              Deadline: {obl.deadline}
                            </span>
                          )}
                        </div>

                        {obl.clause && (
                          <button
                            type="button"
                            onClick={() => handleViewInContract(obl.clause!.id)}
                            className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline"
                          >
                            <span>{obl.clause.section}</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        )}
                      </div>

                      <p className="text-sm font-medium text-foreground leading-relaxed">
                        {obl.description}
                      </p>

                      {obl.clause && (
                        <blockquote className="text-xs font-mono text-muted/90 italic border-l-2 border-slate-300 pl-2.5 mt-2 bg-background/50 p-2 rounded">
                          &ldquo;{obl.clause.text}&rdquo;
                        </blockquote>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 5: KEY TERMS */}
          {activeTab === "keyTerms" && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
                <div>
                  <h2 className="text-base font-bold text-foreground">
                    Defined Key Terms ({filteredKeyTerms.length})
                  </h2>
                  <p className="text-xs text-secondary">
                    Key contractual definitions, clauses, and economic metrics.
                  </p>
                </div>

                <div className="relative min-w-[200px]">
                  <Search className="w-3.5 h-3.5 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Search key terms…"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-border bg-surface text-xs text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
                  />
                </div>
              </div>

              {filteredKeyTerms.length === 0 ? (
                <EmptyState
                  title="No key terms identified"
                  description="No defined terms match your search or were extracted from this agreement."
                  action={
                    <button
                      type="button"
                      onClick={() => setSearchQuery("")}
                      className="rounded-lg bg-surface border border-border px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-slate-50"
                    >
                      Clear Search
                    </button>
                  }
                />
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  {filteredKeyTerms.map((kt) => (
                    <article
                      key={kt.id}
                      className="rounded-xl border border-border bg-surface p-4 sm:p-5 space-y-2.5 shadow-xs card-hover flex flex-col justify-between"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <h3 className="text-sm font-bold text-foreground">
                            {kt.term}
                          </h3>
                          {kt.clause && (
                            <button
                              type="button"
                              onClick={() => handleViewInContract(kt.clause!.id)}
                              className="text-[11px] font-semibold text-accent hover:underline inline-flex items-center gap-1 shrink-0"
                            >
                              <span>§ {kt.clause.section}</span>
                              <ArrowRight className="w-2.5 h-2.5" />
                            </button>
                          )}
                        </div>
                        <p className="text-xs sm:text-sm text-secondary leading-relaxed">
                          {kt.value}
                        </p>
                      </div>

                      {kt.clause && (
                        <blockquote className="text-[11px] font-mono text-muted/90 italic border-l-2 border-slate-300 pl-2 bg-background/50 p-2 rounded">
                          &ldquo;{kt.clause.text}&rdquo;
                        </blockquote>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
