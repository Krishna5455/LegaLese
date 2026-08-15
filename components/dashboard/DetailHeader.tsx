"use client";

import Link from "next/link";
import { useState, useTransition } from "react";

import { downloadReport, generateReport } from "@/lib/actions/reports";
import { getRiskLabel } from "@/lib/ai/scorer";
import type { DetailedAnalysis, ReportRow } from "@/types/analysis";
import type { Document } from "@/types/database";

type DetailHeaderProps = {
  document: Document;
  analysis: DetailedAnalysis;
  initialReport?: ReportRow | null;
};

export function DetailHeader({
  document: doc,
  analysis,
  initialReport,
}: DetailHeaderProps) {

  const [report, setReport] = useState<ReportRow | null>(initialReport ?? null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);

  const [isGenerating, startGenerateTransition] = useTransition();
  const [isDownloading, startDownloadTransition] = useTransition();

  const { label: riskLabel, level: riskLevel } = getRiskLabel(analysis.risk_score);

  const riskBadgeStyles: Record<string, string> = {
    informational: "bg-blue-50 text-blue-700 border-blue-200",
    low: "bg-green-50 text-green-700 border-green-200",
    medium: "bg-yellow-50 text-yellow-800 border-yellow-200",
    high: "bg-orange-50 text-orange-700 border-orange-200",
  };

  const handleGenerateReport = () => {
    setReportError(null);
    startGenerateTransition(async () => {
      const result = await generateReport(doc.id);
      if (result.error) {
        setReportError(result.error);
      } else if (result.report) {
        setReport(result.report);
      }
    });
  };

  const handleDownloadReport = () => {
    setReportError(null);
    startDownloadTransition(async () => {
      const result = await downloadReport(doc.id);
      if (result.error) {
        setReportError(result.error);
      } else if (result.content && result.filename) {
        const blob = new Blob([result.content], { type: "text/markdown;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = window.document.createElement("a");
        a.href = url;
        a.download = result.filename;
        window.document.body.appendChild(a);
        a.click();
        window.document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    });
  };

  const handleCopyQuestions = () => {
    const allQuestions: string[] = [];
    analysis.findings.forEach((f) => {
      if (f.questions && f.questions.length > 0) {
        f.questions.forEach((q) => {
          if (!allQuestions.includes(q)) {
            allQuestions.push(q);
          }
        });
      }
    });

    if (allQuestions.length === 0) {
      alert("No review questions available in this analysis.");
      return;
    }

    const textToCopy = [
      `LegaLese Review Questions — ${doc.filename}`,
      `Generated on ${new Date().toLocaleDateString()}`,
      ``,
      ...allQuestions.map((q, i) => `${i + 1}. ${q}`),
    ].join("\n");


    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 3000);
    });
  };

  function formatDate(iso: string) {
    try {
      return new Intl.DateTimeFormat("en-US", {
        dateStyle: "medium",
      }).format(new Date(iso));
    } catch {
      return iso;
    }
  }

  return (
    <div className="space-y-6">
      {/* Back button */}
      <div>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 text-sm font-medium text-muted hover:text-foreground transition-colors"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Dashboard
        </Link>
      </div>

      {/* Title & metadata bar */}
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="rounded bg-accent/10 px-2 py-0.5 text-xs font-bold uppercase tracking-wider text-accent">
              {doc.document_type ? doc.document_type.toUpperCase() : "CONTRACT"}
            </span>
            <span className={`inline-flex items-center rounded-full border px-3 py-0.5 text-xs font-semibold ${riskBadgeStyles[riskLevel] ?? riskBadgeStyles.low}`}>
              {riskLabel}
            </span>
            {analysis.model && (
              <span className="rounded bg-border px-2 py-0.5 font-mono text-xs text-muted">
                {analysis.model}
              </span>
            )}
          </div>

          <h1 className="text-2xl font-bold text-foreground md:text-3xl">
            {doc.filename}
          </h1>

          <p className="text-xs text-muted">
            Uploaded {formatDate(doc.created_at)} · Analyzed {formatDate(analysis.created_at)}
          </p>

        </div>

        {/* Actions header */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleCopyQuestions}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-surface transition-colors"
            title="Copy all questions to clipboard"
          >
            <svg className="h-4 w-4 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            {copySuccess ? "Questions Copied!" : "Copy Questions"}
          </button>

          {!report ? (
            <button
              type="button"
              onClick={handleGenerateReport}
              disabled={isGenerating}
              className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white hover:bg-accent-hover disabled:opacity-50 transition-colors"
            >
              {isGenerating ? (
                <>
                  <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Generating Report…
                </>
              ) : (
                "Generate Report"
              )}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleDownloadReport}
              disabled={isDownloading}
              className="inline-flex items-center gap-2 rounded-lg border border-accent/40 bg-accent/10 px-4 py-2 text-xs font-semibold text-accent hover:bg-accent/20 disabled:opacity-50 transition-colors"
            >
              {isDownloading ? (
                "Downloading…"
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Report (.md)
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {reportError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
          {reportError}
        </div>
      )}
    </div>
  );
}
