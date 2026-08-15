"use client";

import type { RiskLevel } from "@/types/analysis";

type FilterOption = "all" | RiskLevel;

type SearchFilterBarProps = {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  selectedRisk: FilterOption;
  onRiskChange: (r: FilterOption) => void;
  riskCounts: Record<FilterOption, number>;
};

export function SearchFilterBar({
  searchQuery,
  onSearchChange,
  selectedRisk,
  onRiskChange,
  riskCounts,
}: SearchFilterBarProps) {
  const options: { id: FilterOption; label: string }[] = [
    { id: "all", label: "All Findings" },
    { id: "high", label: "High" },
    { id: "medium", label: "Medium" },
    { id: "low", label: "Low" },
    { id: "informational", label: "Informational" },
  ];

  const hasActiveFilter = searchQuery.trim() !== "" || selectedRisk !== "all";

  return (
    <div className="space-y-4 rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        {/* Search input */}
        <div className="relative flex-1">
          <svg
            className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search findings, clauses, terms, or obligations…"
            className="w-full rounded-lg border border-border bg-background pl-9 pr-8 py-2 text-xs text-foreground placeholder:text-muted focus:border-accent focus:outline-none"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => onSearchChange("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted hover:text-foreground"
            >
              ✕
            </button>
          )}
        </div>

        {/* Clear all filters */}
        {hasActiveFilter && (
          <button
            type="button"
            onClick={() => {
              onSearchChange("");
              onRiskChange("all");
            }}
            className="text-xs font-semibold text-accent hover:underline shrink-0"
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* Risk filter badges */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted mr-1">Risk Filter:</span>
        {options.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => onRiskChange(opt.id)}
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              selectedRisk === opt.id
                ? "bg-accent text-white font-semibold"
                : "bg-background border border-border text-muted hover:text-foreground"
            }`}
          >
            {opt.label}
            <span
              className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                selectedRisk === opt.id
                  ? "bg-white/20 text-white"
                  : "bg-surface text-muted"
              }`}
            >
              {riskCounts[opt.id] ?? 0}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
