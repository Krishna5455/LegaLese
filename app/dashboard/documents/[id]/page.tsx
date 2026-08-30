import Link from "next/link";
import { redirect } from "next/navigation";

import { ContractDetailWorkspace } from "@/components/dashboard/ContractDetailWorkspace";
import { DetailHeader } from "@/components/dashboard/DetailHeader";

import { getAnalysis } from "@/lib/actions/analyses";
import { getReport } from "@/lib/actions/reports";
import { createClient } from "@/lib/supabase/server";
import type { Document } from "@/types/database";

export const dynamic = "force-dynamic";

type DocumentDetailPageProps = {
  params: Promise<{ id: string }>;
};

export default async function DocumentDetailPage({
  params,
}: DocumentDetailPageProps) {
  const { id: documentId } = await params;

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  // 1. Fetch document and verify user ownership
  const { data: document, error: docError } = await supabase
    .from("documents")
    .select("*")
    .eq("id", documentId)
    .eq("user_id", user.id)
    .single();

  // Document Not Found or Unauthorized Access
  if (docError || !document) {
    return (
      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-6 py-16 text-center">
        <div className="rounded-full bg-red-50 p-4 text-red-600 mb-4">
          <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h1 className="text-2xl font-bold text-foreground">Document Not Found</h1>
        <p className="mt-2 text-sm text-muted">
          The document you requested does not exist or you do not have permission to view it.
        </p>
        <Link
          href="/dashboard"
          className="mt-6 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white hover:bg-accent-hover transition-colors"
        >
          Return to Dashboard
        </Link>
      </main>
    );
  }

  // Fetch analysis and report data in parallel
  const [{ analysis }, { report }] = await Promise.all([
    getAnalysis(documentId),
    getReport(documentId),
  ]);

  const docTyped = document as Document;
  const status = (docTyped.status || "").toLowerCase();

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8">
      {!analysis ? (
        <div className="space-y-6">
          <DetailHeader
            document={docTyped}
            analysis={{
              id: "",
              document_id: documentId,
              user_id: user.id,
              risk_score: null,
              summary: "No analysis generated yet for this document.",
              result: {},
              model: "Verified Commercial Analysis",
              created_at: new Date().toISOString(),
              clauses: [],
              findings: [],
              key_terms: [],
              obligations: [],
            }}
            initialReport={report}
          />

          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-surface px-6 py-16 text-center shadow-xs">
            <h2 className="text-lg font-bold text-foreground">
              No Analysis Available Yet
            </h2>
            <p className="mt-1 max-w-md text-xs text-secondary">
              {status === "complete"
                ? "This document has completed text extraction. Trigger AI analysis from the dashboard."
                : status === "processing"
                  ? "Document text extraction is currently processing. Please refresh in a moment."
                  : "Document text extraction has not completed."}
            </p>
            <Link
              href="/dashboard"
              className="mt-5 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white hover:bg-accent-hover transition-colors shadow-xs"
            >
              Go to Dashboard to Analyze
            </Link>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <DetailHeader
            document={docTyped}
            analysis={analysis}
            initialReport={report}
          />

          <ContractDetailWorkspace analysis={analysis} />
        </div>
      )}
    </main>
  );
}