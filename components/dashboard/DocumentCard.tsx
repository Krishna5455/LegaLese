"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useState, useTransition } from "react";

import { analyzeDocument, getAnalysis } from "@/lib/actions/analyses";
import { deleteDocument, processDocument } from "@/lib/actions/documents";
import { formatBytes } from "@/lib/documents/validation";
import type { DetailedAnalysis } from "@/types/analysis";
import type { Document } from "@/types/database";

const AnalysisPanel = dynamic(
  () =>
    import("@/components/dashboard/AnalysisPanel").then(
      (mod) => mod.AnalysisPanel,
    ),
  {
    loading: () => (
      <div className="mt-4 rounded-xl border border-border bg-surface p-6 text-center text-xs text-muted">
        <div className="flex items-center justify-center gap-2">
          <svg className="h-4 w-4 animate-spin text-accent" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span>Loading analysis details...</span>
        </div>
      </div>
    ),
    ssr: false,
  },
);

function formatDate(value: string) {
  try {
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function getFileType(doc: Document): string {
  if (doc.document_type) return doc.document_type.toUpperCase();
  const ext = doc.filename?.split(".").pop()?.toUpperCase();
  return ext || "FILE";
}

function getStatusBadge(status?: string | null) {
  const s = (status || "uploaded").toLowerCase();
  if (s === "complete") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
        Ready
      </span>
    );
  }
  if (s === "processing") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-xs font-semibold text-amber-700">
        <svg
          className="h-3 w-3 animate-spin text-amber-600"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
        Processing...
      </span>
    );
  }
  if (s === "failed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-50 border border-rose-200 px-2.5 py-0.5 text-xs font-semibold text-rose-700">
        <span className="h-1.5 w-1.5 rounded-full bg-rose-500"></span>
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 text-xs font-semibold text-indigo-700">
      <span className="h-1.5 w-1.5 rounded-full bg-indigo-500"></span>
      Uploaded
    </span>
  );
}

export function DocumentCard({
  document,
  existingAnalysis,
}: {
  document: Document;
  existingAnalysis?: DetailedAnalysis | null;
}) {
  const [isConfirming, setIsConfirming] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isDeleting, startDeleteTransition] = useTransition();
  const [isProcessing, startProcessTransition] = useTransition();
  const [isAnalyzing, startAnalyzeTransition] = useTransition();
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);

  const [analyzedResult, setAnalyzedResult] = useState<DetailedAnalysis | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);

  const localAnalysis = analyzedResult ?? existingAnalysis ?? null;

  const isPending = isDeleting || isProcessing || isAnalyzing || isLoadingDetails;
  const status = (document.status || "uploaded").toLowerCase();
  const hasAnalysis = localAnalysis != null;

  const handleDelete = () => {
    setActionError(null);
    startDeleteTransition(async () => {
      const result = await deleteDocument(document.id);
      if (result.error) {
        setActionError(result.error);
        setIsConfirming(false);
      }
    });
  };

  const handleProcess = () => {
    setActionError(null);
    startProcessTransition(async () => {
      const result = await processDocument(document.id);
      if (result.error) {
        setActionError(result.error);
      }
    });
  };

  const handleAnalyze = () => {
    setActionError(null);
    startAnalyzeTransition(async () => {
      const result = await analyzeDocument(document.id);
      if (result.error) {
        setActionError(result.error);
      } else if (result.analysis) {
        setAnalyzedResult(result.analysis);
        setShowAnalysis(true);
      }
    });
  };

  const handleToggleQuickView = async () => {
    if (!showAnalysis && localAnalysis && (!localAnalysis.clauses || localAnalysis.clauses.length === 0)) {
      setIsLoadingDetails(true);
      try {
        const res = await getAnalysis(document.id);
        if (res.analysis) {
          setAnalyzedResult(res.analysis);
        }
      } catch (err) {
        console.warn("Failed to load full analysis details for quick view:", err);
      } finally {
        setIsLoadingDetails(false);
      }
    }
    setShowAnalysis((prev) => !prev);
  };

  return (
    <li className="rounded-xl border border-border bg-surface p-4 transition-all hover:shadow-xs hover:border-slate-300">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3.5 min-w-0">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-accent-soft border border-accent/20 text-xs font-bold text-accent">
            {getFileType(document)}
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-bold text-foreground">
              {document.filename}
            </p>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-secondary">
              <span>{formatBytes(document.size_bytes)}</span>
              <span>•</span>
              <span>Added {formatDate(document.created_at)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 sm:justify-end">
          <div className="flex items-center gap-2">
            {getStatusBadge(document.status)}

            {/* Process / Retry button for documents not yet processed */}
            {(status === "uploaded" || status === "failed") && (
              <button
                type="button"
                onClick={handleProcess}
                disabled={isPending}
                className="rounded border border-border px-2 py-1 text-xs font-medium text-foreground hover:bg-background disabled:opacity-50 transition-colors"
                title="Extract and process text"
              >
                {isProcessing
                  ? "Processing..."
                  : status === "failed"
                    ? "Retry"
                    : "Process"}
              </button>
            )}

            {/* Analyze button — shown for fully processed documents without an analysis */}
            {status === "complete" && !hasAnalysis && (
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={isPending}
                className="inline-flex items-center gap-1 rounded bg-accent px-2.5 py-1 text-xs font-semibold text-white hover:bg-accent-hover disabled:opacity-50 transition-colors shadow-2xs"
                title="Analyze this contract with AI"
              >
                {isAnalyzing ? (
                  <>
                    <svg
                      className="h-3 w-3 animate-spin"
                      viewBox="0 0 24 24"
                      fill="none"
                      aria-hidden="true"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span>Analyzing…</span>
                  </>
                ) : (
                  <span>Analyze</span>
                )}
              </button>
            )}

            {/* View / Hide Analysis toggle & Open Full Analysis link */}
            {hasAnalysis && (
              <>
                <Link
                  href={`/dashboard/documents/${document.id}`}
                  className="inline-flex items-center gap-1 rounded bg-accent px-2.5 py-1 text-xs font-semibold text-white hover:bg-accent-hover transition-colors shadow-2xs"
                  title="Open dedicated analysis workspace"
                >
                  <span>Open Full Analysis</span>
                  <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </Link>
                <button
                  type="button"
                  onClick={handleToggleQuickView}
                  disabled={isPending}
                  className="rounded border border-accent/30 bg-accent/5 px-2.5 py-1 text-xs font-semibold text-accent hover:bg-accent/10 disabled:opacity-50 transition-colors"
                >
                  {isLoadingDetails
                    ? "Loading..."
                    : showAnalysis
                      ? "Hide Quick View"
                      : "Quick View"}
                </button>
              </>
            )}
          </div>

          {!isConfirming ? (
            <button
              type="button"
              onClick={() => setIsConfirming(true)}
              disabled={isPending}
              className="rounded p-1.5 text-xs font-medium text-muted hover:bg-red-50 hover:text-red-600 disabled:opacity-50 transition-colors"
              title="Delete contract"
              aria-label={`Delete ${document.filename}`}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleDelete}
                disabled={isPending}
                className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
              >
                {isDeleting ? "Deleting..." : "Confirm"}
              </button>
              <button
                type="button"
                onClick={() => setIsConfirming(false)}
                disabled={isPending}
                className="rounded border border-border px-2 py-1 text-xs font-medium text-muted hover:text-foreground disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {actionError ? (
        <p className="mt-2 rounded bg-red-50 p-2 text-xs text-red-700">
          {actionError}
        </p>
      ) : null}

      {/* Inline analysis panel (lazy loaded on demand) */}
      {showAnalysis && localAnalysis && (
        <AnalysisPanel analysis={localAnalysis} />
      )}
    </li>
  );
}
