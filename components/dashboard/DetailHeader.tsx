"use client";

import Link from "next/link";
import { useState, useTransition } from "react";
import { downloadReport, generateReport } from "@/lib/actions/reports";
import { getRiskLabel } from "@/lib/ai/scorer";
import type { DetailedAnalysis, ReportRow } from "@/types/analysis";
import type { Document } from "@/types/database";
import {
  ArrowLeft,
  Copy,
  Check,
  Download,
  Loader2,
  FileText,
  Calendar,
  AlertCircle,
  Sparkles,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Info,
} from "lucide-react";

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
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const [isGenerating, startGenerateTransition] = useTransition();
  const [isDownloading, startDownloadTransition] = useTransition();

  const { label: riskLabel, level: riskLevel } = getRiskLabel(analysis.risk_score);

  const riskConfig: Record<string, { badge: string; icon: React.ReactNode }> = {
    informational: {
      badge: "bg-indigo-50 text-indigo-700 border-indigo-200",
      icon: <Info className="w-3.5 h-3.5 text-indigo-600" />,
    },
    low: {
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
      icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
    },
    medium: {
      badge: "bg-amber-50 text-amber-700 border-amber-200",
      icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />,
    },
    high: {
      badge: "bg-rose-50 text-rose-700 border-rose-200",
      icon: <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />,
    },
  };

  const currentRisk = riskConfig[riskLevel] ?? riskConfig.low;

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
        const blob = new Blob([result.content], {
          type: "text/markdown;charset=utf-8",
        });
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
      setCopyStatus("No questions available");
      setTimeout(() => setCopyStatus(null), 3000);
      return;
    }

    const textToCopy = [
      `LegaLese Review Questions — ${doc.filename}`,
      `Generated on ${new Date().toLocaleDateString()}`,
      ``,
      ...allQuestions.map((q, i) => `${i + 1}. ${q}`),
    ].join("\n");

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopyStatus("Questions Copied!");
      setTimeout(() => setCopyStatus(null), 3000);
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
    <header className="space-y-4">
      {/* Back button */}
      <div>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-accent hover:underline transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Dashboard</span>
        </Link>
      </div>

      {/* Main Header Card */}
      <div className="rounded-2xl border border-border bg-surface p-6 sm:p-7 shadow-xs space-y-5">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2 max-w-3xl">
            {/* Badges row */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-md bg-accent-soft border border-accent/20 px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider text-accent">
                {doc.document_type ? doc.document_type.toUpperCase() : "CONTRACT"}
              </span>

              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${currentRisk.badge}`}
              >
                {currentRisk.icon}
                <span>{riskLabel}</span>
              </span>

              <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 border border-slate-200/80 px-2 py-0.5 text-xs font-medium text-secondary">
                <Sparkles className="w-3 h-3 text-accent" />
                <span>AI Commercial Audit</span>
              </span>
            </div>

            {/* Document Title */}
            <h1 className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight text-foreground">
              {doc.filename}
            </h1>

            {/* Dates / metadata */}
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
              <span className="inline-flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5" />
                <span>Uploaded {formatDate(doc.created_at)}</span>
              </span>
              {analysis.created_at && (
                <>
                  <span>•</span>
                  <span>Analyzed {formatDate(analysis.created_at)}</span>
                </>
              )}
              {analysis.model && (
                <>
                  <span>•</span>
                  <span className="font-mono">{analysis.model}</span>
                </>
              )}
            </div>
          </div>

          {/* Action Toolbar */}
          <div className="flex flex-wrap items-center gap-2 pt-1 md:pt-0">
            <button
              type="button"
              onClick={handleCopyQuestions}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 btn-interactive shadow-2xs cursor-pointer"
              title="Copy pre-signing review questions to clipboard"
            >
              {copyStatus ? (
                <>
                  <Check className="w-3.5 h-3.5 text-accent" />
                  <span className="text-accent">{copyStatus}</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-secondary" />
                  <span>Copy Questions</span>
                </>
              )}
            </button>

            {!report ? (
              <button
                type="button"
                onClick={handleGenerateReport}
                disabled={isGenerating}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-3.5 py-2 text-xs font-semibold text-white hover:bg-accent-hover disabled:opacity-50 btn-interactive shadow-xs cursor-pointer"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Generating Report…</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-3.5 h-3.5" />
                    <span>Generate Report</span>
                  </>
                )}
              </button>
            ) : (
              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={isDownloading}
                className="inline-flex items-center gap-2 rounded-lg border border-accent/30 bg-accent-soft px-3.5 py-2 text-xs font-semibold text-accent hover:bg-accent/20 disabled:opacity-50 btn-interactive shadow-xs cursor-pointer"
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Downloading…</span>
                  </>
                ) : (
                  <>
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Report (.md)</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Error Alert */}
        {reportError && (
          <div className="flex items-center justify-between rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800 font-medium animate-fadeIn">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{reportError}</span>
            </div>
            <button
              type="button"
              onClick={() => setReportError(null)}
              className="text-xs font-bold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
