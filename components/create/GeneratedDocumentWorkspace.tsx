"use client";

import Link from "next/link";
import { useState } from "react";
import dynamic from "next/dynamic";
import {
  FileText,
  Sparkles,
  ShieldAlert,
  SlidersHorizontal,
  ArrowLeft,
  Download,
  Copy,
  Check,
  Loader2,
  AlertCircle,
  Shield,
  Users,
} from "lucide-react";

const DocumentExplanationView = dynamic(
  () =>
    import("@/components/create/DocumentExplanationView").then(
      (m) => m.DocumentExplanationView,
    ),
  {
    loading: () => (
      <div className="rounded-2xl border border-border bg-surface p-8 space-y-6 animate-pulse">
        <div className="flex items-center justify-between border-b border-border pb-4">
          <div className="space-y-2">
            <div className="h-5 w-48 bg-slate-200 rounded-md" />
            <div className="h-3.5 w-72 bg-slate-100 rounded-md" />
          </div>
          <div className="h-8 w-28 bg-slate-200 rounded-md" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-16 bg-slate-100 rounded-xl" />
          ))}
        </div>
        <div className="h-28 bg-slate-100 rounded-xl" />
        <div className="grid gap-5 md:grid-cols-2">
          <div className="h-36 bg-slate-100 rounded-xl" />
          <div className="h-36 bg-slate-100 rounded-xl" />
        </div>
      </div>
    ),
    ssr: false,
  },
);

const DocumentReviewView = dynamic(
  () =>
    import("@/components/create/DocumentReviewView").then(
      (m) => m.DocumentReviewView,
    ),
  {
    loading: () => (
      <div className="rounded-2xl border border-border bg-surface p-8 space-y-6 animate-pulse">
        <div className="flex items-center justify-between border-b border-border pb-4">
          <div className="space-y-2">
            <div className="h-5 w-52 bg-slate-200 rounded-md" />
            <div className="h-3.5 w-80 bg-slate-100 rounded-md" />
          </div>
          <div className="h-8 w-28 bg-slate-200 rounded-md" />
        </div>
        <div className="h-3 w-full bg-slate-200 rounded-full" />
        <div className="grid grid-cols-3 gap-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-slate-100 rounded-xl" />
          ))}
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-32 bg-slate-100 rounded-xl" />
          ))}
        </div>
      </div>
    ),
    ssr: false,
  },
);

import { exportGeneratedDocument } from "@/lib/actions/generated-documents";
import { explainGeneratedDocumentAction } from "@/lib/actions/explanation";
import { reviewGeneratedDocumentAction } from "@/lib/actions/review";
import type { DocumentExplanation } from "@/lib/ai/explanation-schema";
import type { DocumentReview } from "@/lib/ai/review-schema";
import {
  generatedDocumentDownloadFilename,
  generatedDocumentToMarkdown,
} from "@/lib/generation/export";
import type { GeneratedDocumentRow } from "@/types/generation";

type GeneratedDocumentWorkspaceProps = {
  document: GeneratedDocumentRow;
};

type ExportFormat = "pdf" | "docx" | "md";
type ViewMode = "document" | "explanation" | "review";

export function GeneratedDocumentWorkspace({
  document: doc,
}: GeneratedDocumentWorkspaceProps) {
  const [copied, setCopied] = useState(false);
  const [activeExportFormat, setActiveExportFormat] = useState<ExportFormat | null>(
    null,
  );
  const [exportError, setExportError] = useState<string | null>(null);

  // Workspace View State (Default is "document" -> View Agreement)
  const [viewMode, setViewMode] = useState<ViewMode>("document");

  // Explanation state (cached in memory)
  const [explanation, setExplanation] = useState<DocumentExplanation | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  // Review state (cached in memory)
  const [review, setReview] = useState<DocumentReview | null>(null);
  const [isReviewing, setIsReviewing] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);

  const content = doc.generated_content;
  const markdownText = generatedDocumentToMarkdown(content, doc.created_at);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(markdownText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
      setExportError("Failed to copy document content to clipboard.");
    }
  };

  const handleDownload = async (format: ExportFormat) => {
    setActiveExportFormat(format);
    setExportError(null);

    try {
      const result = await exportGeneratedDocument(doc.id, format, content);

      if (!result.success || !result.data || !result.filename) {
        const response = await fetch(
          `/api/documents/generated/${doc.id}/export?format=${format}`,
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.error ||
              result.error ||
              `Failed to download ${format.toUpperCase()} document export.`,
          );
        }

        const blob = await response.blob();
        const filename = generatedDocumentDownloadFilename(content.title, format, doc.id);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return;
      }

      let blob: Blob;
      if (result.isBase64) {
        const binaryString = atob(result.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        blob = new Blob([bytes], {
          type: result.mimeType || "application/octet-stream",
        });
      } else {
        blob = new Blob([result.data], {
          type: result.mimeType || "text/markdown;charset=utf-8",
        });
      }

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = result.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(`Export download failed (${format}):`, err);
      setExportError(
        err instanceof Error
          ? err.message
          : "An error occurred while downloading the document.",
      );
    } finally {
      setActiveExportFormat(null);
    }
  };

  const handleUnderstandAgreement = async () => {
    if (explanation) {
      setViewMode("explanation");
      return;
    }

    setIsExplaining(true);
    setExplanationError(null);

    try {
      const res = await explainGeneratedDocumentAction(doc.id);
      if (!res.success || !res.explanation) {
        throw new Error(res.error || "Unable to generate explanation.");
      }
      setExplanation(res.explanation);
      setViewMode("explanation");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to analyze document.";
      console.error("[LegaLese/Workspace] Understand Agreement error:", msg);
      setExplanationError(msg);
    } finally {
      setIsExplaining(false);
    }
  };

  const handleReviewAgreement = async () => {
    if (review) {
      setViewMode("review");
      return;
    }

    setIsReviewing(true);
    setReviewError(null);

    try {
      const res = await reviewGeneratedDocumentAction(doc.id);
      if (!res.success || !res.review) {
        throw new Error(res.error || "Unable to generate contract review.");
      }
      setReview(res.review);
      setViewMode("review");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to review document.";
      console.error("[LegaLese/Workspace] Review Agreement error:", msg);
      setReviewError(msg);
    } finally {
      setIsReviewing(false);
    }
  };

  const handleJumpToSection = (secId: string) => {
    setViewMode("document");
    setTimeout(() => {
      const el = document.getElementById(`section-${secId}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 150);
  };

  const sortedSections = [...(content.sections ?? [])].sort(
    (a, b) => a.order - b.order,
  );

  return (
    <div className="space-y-8">
      {/* Top Back Navigation Link */}
      <div>
        <Link
          href="/dashboard"
          className="text-xs font-semibold text-accent hover:underline inline-flex items-center gap-1.5 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Dashboard</span>
        </Link>
      </div>

      {/* Header Banner & Title */}
      <div className="rounded-2xl border border-border bg-surface p-6 sm:p-7 space-y-6 shadow-xs">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
                Draft · Saved in Workspace
              </span>
            </div>
            <h1 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
              {content.title}
            </h1>
            <p className="mt-1 text-sm text-secondary">
              {content.parties.clientName} × {content.parties.freelancerName}
            </p>
          </div>

          {/* Export Action Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => handleDownload("pdf")}
              disabled={!!activeExportFormat}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white hover:bg-accent-hover disabled:opacity-50 transition-colors shadow-xs cursor-pointer"
            >
              {activeExportFormat === "pdf" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Preparing PDF...</span>
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5" />
                  <span>Download PDF</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={() => handleDownload("docx")}
              disabled={!!activeExportFormat}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {activeExportFormat === "docx" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>DOCX...</span>
                </>
              ) : (
                <span>DOCX</span>
              )}
            </button>

            <button
              type="button"
              onClick={() => handleDownload("md")}
              disabled={!!activeExportFormat}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {activeExportFormat === "md" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Markdown...</span>
                </>
              ) : (
                <span>Markdown</span>
              )}
            </button>

            <button
              type="button"
              onClick={handleCopy}
              disabled={!!activeExportFormat}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-slate-50 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-accent" />
                  <span className="text-accent">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-secondary" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Primary View Switcher Pipeline Navigation Bar */}
        <div
          role="tablist"
          aria-label="Agreement workspace views"
          className="flex items-center gap-1.5 p-1 glass-pill rounded-xl overflow-x-auto"
        >
          <button
            type="button"
            role="tab"
            aria-selected={viewMode === "document"}
            onClick={() => setViewMode("document")}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all shrink-0 cursor-pointer flex items-center gap-2 ${
              viewMode === "document"
                ? "bg-surface text-accent shadow-xs border border-border"
                : "text-secondary hover:text-foreground hover:bg-slate-100"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>View Agreement</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={viewMode === "explanation"}
            onClick={handleUnderstandAgreement}
            disabled={isExplaining}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all shrink-0 cursor-pointer flex items-center gap-2 ${
              viewMode === "explanation"
                ? "bg-surface text-accent shadow-xs border border-border"
                : "text-secondary hover:text-foreground hover:bg-slate-100"
            }`}
          >
            {isExplaining ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            <span>{isExplaining ? "Analyzing..." : "Understand Agreement"}</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={viewMode === "review"}
            onClick={handleReviewAgreement}
            disabled={isReviewing}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all shrink-0 cursor-pointer flex items-center gap-2 ${
              viewMode === "review"
                ? "bg-surface text-accent shadow-xs border border-border"
                : "text-secondary hover:text-foreground hover:bg-slate-100"
            }`}
          >
            {isReviewing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            ) : (
              <ShieldAlert className="w-3.5 h-3.5" />
            )}
            <span>{isReviewing ? "Reviewing..." : "Review Agreement"}</span>
          </button>

          <button
            type="button"
            disabled
            className="px-3.5 py-2 text-xs font-medium text-slate-400 select-none cursor-not-allowed flex items-center gap-2 shrink-0"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <span>Customize</span>
            <span className="rounded bg-accent-soft text-accent border border-accent/20 px-1.5 py-0.5 text-[9px] font-semibold">
              Coming Soon
            </span>
          </button>
        </div>

        {/* Error Alerts */}
        {exportError ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-xs sm:text-sm text-red-600 dark:text-red-400 flex items-center justify-between font-semibold">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span>{exportError}</span>
            </div>
            <button
              onClick={() => setExportError(null)}
              className="text-xs font-bold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {explanationError ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-xs sm:text-sm text-red-600 dark:text-red-400 flex items-center justify-between font-semibold">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span>{explanationError}</span>
            </div>
            <button
              onClick={() => setExplanationError(null)}
              className="text-xs font-bold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {reviewError ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-xs sm:text-sm text-red-600 dark:text-red-400 flex items-center justify-between font-semibold">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span>{reviewError}</span>
            </div>
            <button
              onClick={() => setReviewError(null)}
              className="text-xs font-bold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {/* Agreed Parties summary card */}
        <div className="rounded-xl border border-border bg-surface-inset p-5 text-sm space-y-2">
          <p className="font-bold text-foreground uppercase tracking-wider text-xs flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5 text-secondary" />
            <span>Agreed Parties</span>
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <span className="text-muted font-medium">Freelancer:</span>{" "}
              <span className="font-semibold text-foreground">
                {content.parties.freelancerName}
              </span>
            </div>
            <div>
              <span className="text-muted font-medium">Client:</span>{" "}
              <span className="font-semibold text-foreground">
                {content.parties.clientName}
              </span>
            </div>
          </div>
          {content.parties.clientAddress ? (
            <div>
              <span className="text-muted font-medium">Client Address:</span>{" "}
              <span className="text-foreground">
                {content.parties.clientAddress}
              </span>
            </div>
          ) : null}
        </div>
      </div>

      {/* Active AI Processing Loading Banners */}
      {isExplaining && (
        <div className="rounded-2xl border border-accent/25 bg-accent-soft/30 p-10 text-center space-y-3">
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface border border-accent/20 text-accent shadow-xs">
              <Sparkles className="w-6 h-6 animate-pulse text-accent" />
            </div>
          </div>
          <div className="space-y-1 max-w-md mx-auto">
            <h3 className="text-base font-bold text-foreground">
              Synthesizing Plain-Language Explanation...
            </h3>
            <p className="text-xs sm:text-sm text-secondary">
              Gemini is breaking down complex legal terms, obligations, and key checkpoints for your agreement.
            </p>
          </div>
        </div>
      )}

      {isReviewing && (
        <div className="rounded-2xl border border-indigo-200/80 bg-indigo-50/40 p-10 text-center space-y-3">
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface border border-indigo-200 text-indigo-700 shadow-xs">
              <ShieldAlert className="w-6 h-6 animate-pulse text-indigo-600" />
            </div>
          </div>
          <div className="space-y-1 max-w-md mx-auto">
            <h3 className="text-base font-bold text-foreground">
              Conducting Agreement Risk Review...
            </h3>
            <p className="text-xs sm:text-sm text-secondary">
              Gemini is evaluating contract clauses to detect potential liabilities, unbalanced terms, and points to clarify.
            </p>
          </div>
        </div>
      )}

      {/* Workspace View Mode Switcher */}
      {!isExplaining && !isReviewing && (
        <>
          {viewMode === "review" && review ? (
            <div className="animate-fadeIn">
              <DocumentReviewView
                review={review}
                documentTitle={content.title}
                onReturnToDocument={() => setViewMode("document")}
                onJumpToSection={handleJumpToSection}
              />
            </div>
          ) : viewMode === "explanation" && explanation ? (
            <div className="animate-fadeIn">
              <DocumentExplanationView
                explanation={explanation}
                documentTitle={content.title}
                onReturnToDocument={() => setViewMode("document")}
                onJumpToSection={handleJumpToSection}
              />
            </div>
          ) : (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <h2 className="text-xl font-bold text-foreground">
                  Agreement Text ({sortedSections.length} Sections)
                </h2>
              </div>

              {sortedSections.map((sec) => (
                <article
                  key={sec.id}
                  id={`section-${sec.id}`}
                  className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-4 transition-all card-hover shadow-xs"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-xl font-bold text-foreground">
                      {sec.order + 1}. {sec.title}
                    </h3>
                  </div>

                  <div className="prose prose-sm dark:prose-invert max-w-none text-base text-foreground/90 whitespace-pre-line leading-relaxed">
                    {sec.content}
                  </div>
                </article>
              ))}

              {/* Legal Disclaimer Card */}
              <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-6 text-sm text-amber-900 space-y-2">
                <div className="flex items-center gap-2 font-bold text-base">
                  <Shield className="w-4 h-4 text-amber-700" />
                  <span>Important Legal Notice</span>
                </div>
                <p className="leading-relaxed">{content.disclaimer}</p>
              </div>
            </div>
          )}
        </>
      )}

      {/* Roadmap Teasers Section: "More ways to protect yourself" */}
      <section className="rounded-2xl border border-border bg-surface p-7 sm:p-8 space-y-6 pt-6 mt-12 shadow-xs card-hover">
        <div className="border-b border-border pb-4">
          <h3 className="text-xs font-bold text-foreground uppercase tracking-wider">
            More ways to protect yourself
          </h3>
          <p className="text-base text-muted mt-1 font-medium">
            Future workflow enhancements coming to LegaLese.
          </p>
        </div>

        <div className="grid gap-5 md:grid-cols-3">
          {/* Card 1: Customize */}
          <div className="rounded-xl border border-border bg-surface-inset p-5 space-y-2.5 opacity-85">
            <div className="flex items-center justify-between">
              <span className="text-base font-bold text-foreground">1. Customize</span>
              <span className="rounded bg-muted/15 px-2.5 py-0.5 text-[10px] font-bold text-muted">
                COMING SOON
              </span>
            </div>
            <p className="text-sm text-muted leading-relaxed">
              Change clauses using simple natural language instructions.
            </p>
          </div>

          {/* Card 2: Community */}
          <div className="rounded-xl border border-border bg-surface-inset p-5 space-y-2.5 opacity-85">
            <div className="flex items-center justify-between">
              <span className="text-base font-bold text-foreground">2. Community</span>
              <span className="rounded bg-muted/15 px-2.5 py-0.5 text-[10px] font-bold text-muted">
                COMING SOON
              </span>
            </div>
            <p className="text-sm text-muted leading-relaxed">
              Learn from people who have faced similar legal-document situations.
            </p>
          </div>

          {/* Card 3: Legal Expert */}
          <div className="rounded-xl border border-border bg-surface-inset p-5 space-y-2.5 opacity-85">
            <div className="flex items-center justify-between">
              <span className="text-base font-bold text-foreground">3. Legal Expert</span>
              <span className="rounded bg-muted/15 px-2.5 py-0.5 text-[10px] font-bold text-muted">
                COMING SOON
              </span>
            </div>
            <p className="text-sm text-muted leading-relaxed">
              Get professional help when AI assistance isn&apos;t enough.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
