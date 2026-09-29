"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { analyzeDocument } from "@/lib/actions/analyses";
import type { Document } from "@/types/database";
import {
  Sparkles,
  FileText,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Clock,
} from "lucide-react";

type UnanalyzedDocumentViewProps = {
  document: Document;
};

export function UnanalyzedDocumentView({ document: doc }: UnanalyzedDocumentViewProps) {
  const router = useRouter();
  const [isAnalyzing, startAnalyzeTransition] = useTransition();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const status = (doc.status || "uploaded").toLowerCase();
  const isReadyToAnalyze = status === "complete";

  const handleStartAnalysis = () => {
    setErrorMessage(null);
    startAnalyzeTransition(async () => {
      try {
        const result = await analyzeDocument(doc.id);
        if (result.error) {
          setErrorMessage(result.error);
        } else if (result.success) {
          router.refresh();
        }
      } catch (err) {
        console.error("Analysis trigger error:", err);
        setErrorMessage(
          "We could not complete the contract analysis right now. Please try again in a moment.",
        );
      }
    });
  };

  return (
    <div className="space-y-6">
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

      {/* Header card */}
      <div className="rounded-2xl border border-border bg-surface p-6 sm:p-7 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-accent-soft border border-accent/20 px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider text-accent">
            {doc.document_type ? doc.document_type.toUpperCase() : "CONTRACT"}
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 border border-slate-200 px-2.5 py-0.5 text-xs font-semibold text-secondary">
            {status === "complete" ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Text Extracted · Ready for AI</span>
              </>
            ) : status === "processing" ? (
              <>
                <Clock className="w-3.5 h-3.5 text-amber-600 animate-spin" />
                <span>Extracting Text…</span>
              </>
            ) : (
              <span>Extraction Pending</span>
            )}
          </span>
        </div>

        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground">
          {doc.filename}
        </h1>
        <p className="text-xs text-muted">
          Uploaded on {new Date(doc.created_at).toLocaleDateString()}
        </p>
      </div>

      {/* Analysis Call to Action or Loading Skeleton */}
      {isAnalyzing ? (
        <div className="rounded-2xl border border-accent/30 bg-surface p-8 sm:p-12 text-center space-y-6 shadow-xs animate-pulse">
          <div className="flex justify-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-soft border border-accent/20 text-accent shadow-xs">
              <Sparkles className="w-7 h-7 animate-pulse text-accent" />
            </div>
          </div>

          <div className="space-y-2 max-w-md mx-auto">
            <h3 className="text-lg font-bold text-foreground">
              Analyzing Contract with Gemini AI…
            </h3>
            <p className="text-xs sm:text-sm text-secondary leading-relaxed">
              Evaluating contract clauses, detecting potential liabilities, calculating risk scores, and generating plain-language takeaways.
            </p>
          </div>

          {/* Skeleton representation */}
          <div className="grid gap-3 max-w-lg mx-auto pt-4 text-left">
            <div className="h-4 w-40 rounded bg-slate-200" />
            <div className="h-3 w-full rounded bg-slate-100" />
            <div className="h-3 w-4/5 rounded bg-slate-100" />
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-border bg-surface p-8 sm:p-12 text-center space-y-5 shadow-xs">
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 border border-border text-secondary shadow-2xs">
              <FileText className="w-6 h-6 text-accent" />
            </div>
          </div>

          <div className="space-y-1.5 max-w-md mx-auto">
            <h3 className="text-base font-bold text-foreground">
              {isReadyToAnalyze
                ? "Contract Ready for Deep-Dive Analysis"
                : "Document Text Processing"}
            </h3>
            <p className="text-xs sm:text-sm text-secondary leading-relaxed">
              {isReadyToAnalyze
                ? "Text extraction is complete. Trigger AI audit to uncover hidden liability traps, extract obligations, and evaluate fairness."
                : status === "processing"
                  ? "We are currently extracting text from your uploaded document. Please check back in a few seconds."
                  : "Document text extraction has not completed. You can re-attempt processing from the dashboard."}
            </p>
          </div>

          {errorMessage && (
            <div className="max-w-md mx-auto flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800 font-medium text-left">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="pt-2 flex flex-wrap items-center justify-center gap-3">
            {isReadyToAnalyze ? (
              <button
                type="button"
                onClick={handleStartAnalysis}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-xs font-semibold text-white hover:bg-accent-hover transition-colors shadow-xs cursor-pointer"
              >
                <Sparkles className="w-4 h-4" />
                <span>Start AI Contract Analysis</span>
              </button>
            ) : (
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 rounded-lg bg-foreground px-4 py-2 text-xs font-semibold text-background hover:bg-slate-800 transition-colors shadow-xs"
              >
                <span>Return to Dashboard</span>
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
