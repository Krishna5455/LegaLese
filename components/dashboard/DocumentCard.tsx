"use client";

import { useState, useTransition } from "react";

import { AnalysisPanel } from "@/components/dashboard/AnalysisPanel";
import { analyzeDocument } from "@/lib/actions/analyses";
import { deleteDocument, processDocument } from "@/lib/actions/documents";
import { formatBytes } from "@/lib/documents/validation";
import type { AnalysisWithDetails } from "@/types/analysis";
import type { Document } from "@/types/database";

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
      <span className="inline-flex items-center gap-1.5 rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700">
        <span className="h-1.5 w-1.5 rounded-full bg-green-500"></span>
        Ready
      </span>
    );
  }
  if (s === "processing") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-yellow-50 px-2.5 py-0.5 text-xs font-medium text-yellow-700">
        <svg
          className="h-3 w-3 animate-spin text-yellow-600"
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
      <span className="inline-flex items-center gap-1.5 rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
        <span className="h-1.5 w-1.5 rounded-full bg-red-500"></span>
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
      <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
      Uploaded
    </span>
  );
}

export function DocumentCard({
  document,
  existingAnalysis,
}: {
  document: Document;
  existingAnalysis?: AnalysisWithDetails | null;
}) {
  const [isConfirming, setIsConfirming] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isDeleting, startDeleteTransition] = useTransition();
  const [isProcessing, startProcessTransition] = useTransition();
  const [isAnalyzing, startAnalyzeTransition] = useTransition();

  // Track the analysis result in local state so the panel can appear without
  // a full page reload (the server will also revalidate the dashboard path).
  const [localAnalysis, setLocalAnalysis] = useState<AnalysisWithDetails | null>(
    existingAnalysis ?? null,
  );
  const [showAnalysis, setShowAnalysis] = useState(false);

  const isPending = isDeleting || isProcessing || isAnalyzing;
  const status = (document.status || "uploaded").toLowerCase();
  const hasCompleteAnalysis = localAnalysis?.status === "complete";

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
        setLocalAnalysis(result.analysis);
        setShowAnalysis(true);
      }
    });
  };

  return (
    <li className="rounded-lg border border-border bg-surface p-4 transition-shadow hover:shadow-xs">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3 min-w-0">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md bg-accent/10 text-xs font-bold text-accent">
            {getFileType(document)}
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-foreground">
              {document.filename}
            </p>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
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
                className="rounded border border-border px-2 py-1 text-xs font-medium text-foreground hover:bg-background disabled:opacity-50"
                title="Extract and process text"
              >
                {isProcessing
                  ? "Processing..."
                  : status === "failed"
                    ? "Retry"
                    : "Process"}
              </button>
            )}

            {/* Analyze button — only shown for fully processed documents */}
            {status === "complete" && !hasCompleteAnalysis && (
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={isPending}
                className="inline-flex items-center gap-1 rounded bg-accent px-2.5 py-1 text-xs font-semibold text-white hover:bg-accent-hover disabled:opacity-50"
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
                    Analyzing…
                  </>
                ) : (
                  "Analyze"
                )}
              </button>
            )}

            {/* View / Hide Analysis toggle — shown when analysis is complete */}
            {hasCompleteAnalysis && (
              <button
                type="button"
                onClick={() => setShowAnalysis((v) => !v)}
                disabled={isPending}
                className="rounded border border-accent/30 bg-accent/5 px-2.5 py-1 text-xs font-semibold text-accent hover:bg-accent/10 disabled:opacity-50"
              >
                {showAnalysis ? "Hide Analysis" : "View Analysis"}
              </button>
            )}
          </div>

          {!isConfirming ? (
            <button
              type="button"
              onClick={() => setIsConfirming(true)}
              disabled={isPending}
              className="rounded p-1.5 text-xs font-medium text-muted hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
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
                className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                {isDeleting ? "Deleting..." : "Confirm"}
              </button>
              <button
                type="button"
                onClick={() => setIsConfirming(false)}
                disabled={isPending}
                className="rounded border border-border px-2 py-1 text-xs font-medium text-muted hover:text-foreground disabled:opacity-50"
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

      {/* Inline analysis panel */}
      {showAnalysis && localAnalysis && (
        <AnalysisPanel analysis={localAnalysis} />
      )}
    </li>
  );
}

