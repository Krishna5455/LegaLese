"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { FileText, Plus, Upload } from "lucide-react";
import { DocumentCard } from "@/components/dashboard/DocumentCard";
import { AnimatedList } from "@/components/ui/AnimatedList";
import { EmptyState } from "@/components/ui/EmptyState";
import type { DetailedAnalysis } from "@/types/analysis";
import type { Document } from "@/types/database";

type DocumentListProps = {
  documents: Document[] | null;
  error?: string | null;
  analysesMap?: Record<string, DetailedAnalysis>;
};

export function DocumentList({
  documents,
  error,
  analysesMap = {},
}: DocumentListProps) {
  const router = useRouter();

  // Centralized controlled polling: one single timer across all documents
  useEffect(() => {
    const hasProcessing = documents?.some(
      (d) => (d.status || "").toLowerCase() === "processing",
    );
    if (!hasProcessing) return;

    const timer = setTimeout(() => {
      router.refresh();
    }, 4000);

    return () => clearTimeout(timer);
  }, [documents, router]);

  if (error) {
    return (
      <div className="rounded-xl border border-[#FECACA] bg-[#FEF2F2] p-6 text-center">
        <p className="text-sm font-semibold text-[#991B1B]">
          Unable to load stored documents
        </p>
        <p className="mt-1 text-xs text-[#B91C1C]">{error}</p>
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="w-5 h-5 text-[#059669]" />}
        title="No stored documents yet"
        description="Create a custom agreement or upload an existing PDF/DOCX contract to start your review workspace."
        action={
          <Link
            href="/dashboard/create"
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-medium text-white hover:bg-[#262626] transition-all shadow-xs active:scale-98"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Create Agreement</span>
          </Link>
        }
        secondaryAction={
          <Link
            href="#upload"
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-4 py-2 text-xs font-medium text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-2xs active:scale-98"
          >
            <Upload className="w-3.5 h-3.5 text-[#5F6368]" />
            <span>Upload Contract</span>
          </Link>
        }
      />
    );
  }

  return (
    <AnimatedList className="space-y-3">
      {documents.map((doc) => (
        <DocumentCard
          key={doc.id}
          document={doc}
          existingAnalysis={analysesMap[doc.id] ?? null}
        />
      ))}
    </AnimatedList>
  );
}
