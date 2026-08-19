"use client";

import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/Button";
import { DocumentExplanationView } from "@/components/create/DocumentExplanationView";
import { DocumentReviewView } from "@/components/create/DocumentReviewView";
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

  // Workspace View State
  const [viewMode, setViewMode] = useState<ViewMode>("document");

  // Explanation state
  const [explanation, setExplanation] = useState<DocumentExplanation | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  // Review state
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
      // Primary export attempt passing doc.id and structured content
      const result = await exportGeneratedDocument(doc.id, format, content);

      if (!result.success || !result.data || !result.filename) {
        // Fallback attempt via Route Handler if server action returns error
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

  const formattedDate = new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(doc.created_at));

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                ✓ Draft Saved
              </span>
              <span className="text-xs text-muted font-mono">{formattedDate}</span>
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-foreground">
              {content.title}
            </h1>
            <p className="mt-1 text-xs text-muted">
              Document Type:{" "}
              <span className="font-semibold text-foreground">
                Freelance Service Agreement
              </span>
              {doc.model ? (
                <span className="ml-2 font-mono text-muted">
                  (Generated by {doc.model})
                </span>
              ) : null}
            </p>
          </div>

          {/* Export Action Buttons (PDF is Primary) */}
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="primary"
              onClick={() => handleDownload("pdf")}
              disabled={!!activeExportFormat}
              className="text-xs shadow-sm font-semibold"
            >
              {activeExportFormat === "pdf"
                ? "Preparing PDF..."
                : "Download PDF (.pdf)"}
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={() => handleDownload("docx")}
              disabled={!!activeExportFormat}
              className="text-xs font-medium"
            >
              {activeExportFormat === "docx"
                ? "Preparing DOCX..."
                : "Download DOCX (.docx)"}
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={() => handleDownload("md")}
              disabled={!!activeExportFormat}
              className="text-xs font-medium"
            >
              {activeExportFormat === "md"
                ? "Preparing MD..."
                : "Download Markdown (.md)"}
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={handleCopy}
              disabled={!!activeExportFormat}
              className="text-xs font-medium"
            >
              {copied ? "✓ Copied" : "Copy"}
            </Button>
          </div>
        </div>

        {/* Primary View Switcher Navigation Bar */}
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-border/50">
          <Button
            type="button"
            variant={viewMode === "document" ? "primary" : "outline"}
            onClick={() => setViewMode("document")}
            className="text-xs"
          >
            📄 View Agreement
          </Button>

          <Button
            type="button"
            variant={viewMode === "explanation" ? "primary" : "outline"}
            onClick={handleUnderstandAgreement}
            disabled={isExplaining}
            className="text-xs"
          >
            {isExplaining ? "Analyzing..." : "💡 Understand Agreement"}
          </Button>

          <Button
            type="button"
            variant={viewMode === "review" ? "primary" : "outline"}
            onClick={handleReviewAgreement}
            disabled={isReviewing}
            className="text-xs"
          >
            {isReviewing ? "Reviewing..." : "🔍 Review Agreement"}
          </Button>
        </div>

        {/* Error Alerts */}
        {exportError ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-600 dark:text-red-400 flex items-center justify-between">
            <span>⚠️ {exportError}</span>
            <button
              onClick={() => setExportError(null)}
              className="text-xs font-semibold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {explanationError ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-600 dark:text-red-400 flex items-center justify-between">
            <span>⚠️ {explanationError}</span>
            <button
              onClick={() => setExplanationError(null)}
              className="text-xs font-semibold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {reviewError ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-600 dark:text-red-400 flex items-center justify-between">
            <span>⚠️ {reviewError}</span>
            <button
              onClick={() => setReviewError(null)}
              className="text-xs font-semibold underline hover:no-underline ml-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {/* Parties summary card */}
        <div className="rounded-lg border border-border/80 bg-background/50 p-4 text-xs space-y-2">
          <p className="font-semibold text-foreground uppercase tracking-wider text-[10px]">
            Agreed Parties
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            <div>
              <span className="text-muted">Freelancer:</span>{" "}
              <span className="font-medium text-foreground">
                {content.parties.freelancerName}
              </span>
            </div>
            <div>
              <span className="text-muted">Client:</span>{" "}
              <span className="font-medium text-foreground">
                {content.parties.clientName}
              </span>
            </div>
          </div>
          {content.parties.clientAddress ? (
            <div>
              <span className="text-muted">Client Address:</span>{" "}
              <span className="text-foreground font-mono">
                {content.parties.clientAddress}
              </span>
            </div>
          ) : null}
        </div>
      </div>

      {/* Navigation Actions bar */}
      <div className="flex items-center justify-between text-xs">
        <Link
          href="/dashboard/create"
          className="text-accent hover:underline font-medium flex items-center gap-1"
        >
          ← Create another agreement
        </Link>
        <Link
          href="/dashboard"
          className="text-muted hover:text-foreground font-medium"
        >
          Return to Dashboard →
        </Link>
      </div>

      {/* Workspace View Mode Switcher */}
      {viewMode === "review" && review ? (
        <DocumentReviewView
          review={review}
          documentTitle={content.title}
          onReturnToDocument={() => setViewMode("document")}
          onJumpToSection={handleJumpToSection}
        />
      ) : viewMode === "explanation" && explanation ? (
        <DocumentExplanationView
          explanation={explanation}
          documentTitle={content.title}
          onReturnToDocument={() => setViewMode("document")}
          onJumpToSection={handleJumpToSection}
        />
      ) : (
        <div className="space-y-6">
          <div className="flex items-center justify-between border-b border-border/60 pb-3">
            <h2 className="text-base font-bold text-foreground">
              Agreement Sections ({sortedSections.length})
            </h2>
            <span className="text-xs text-muted">
              Section IDs preserved for explanation & review
            </span>
          </div>

          {sortedSections.map((sec) => (
            <article
              key={sec.id}
              id={`section-${sec.id}`}
              className="rounded-xl border border-border bg-surface p-6 space-y-3 transition-colors hover:border-accent/30"
            >
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-foreground">
                  {sec.order + 1}. {sec.title}
                </h3>
                <span className="rounded bg-accent/10 border border-accent/20 px-2 py-0.5 text-[10px] font-mono text-accent">
                  id: {sec.id}
                </span>
              </div>

              <div className="prose prose-sm dark:prose-invert max-w-none text-sm text-foreground/90 whitespace-pre-line leading-relaxed">
                {sec.content}
              </div>
            </article>
          ))}

          {/* Legal Disclaimer Card */}
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-5 text-xs text-amber-800 dark:text-amber-300 space-y-2">
            <div className="flex items-center gap-2 font-bold">
              <span>⚠️ Important Legal Notice</span>
            </div>
            <p className="leading-normal">{content.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  );
}
