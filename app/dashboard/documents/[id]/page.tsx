import Link from "next/link";
import { redirect } from "next/navigation";
import { ContractDetailWorkspace } from "@/components/dashboard/ContractDetailWorkspace";
import { DetailHeader } from "@/components/dashboard/DetailHeader";
import { UnanalyzedDocumentView } from "@/components/dashboard/UnanalyzedDocumentView";
import { getAnalysis } from "@/lib/actions/analyses";
import { getReport } from "@/lib/actions/reports";
import { createClient } from "@/lib/supabase/server";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";
import type { Document } from "@/types/database";
import { FileQuestion, ArrowLeft } from "lucide-react";

import { isDemoMode, getDemoDocumentAndAnalysis } from "@/lib/demo/demo-state";
import type { DetailedAnalysis, ReportRow } from "@/types/analysis";

export const dynamic = "force-dynamic";

type DocumentDetailPageProps = {
  params: Promise<{ id: string }>;
};

export default async function DocumentDetailPage({
  params,
}: DocumentDetailPageProps) {
  const { id: documentId } = await params;

  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  const isDemo = isDemoMode() && (documentId.startsWith("demo-") || user.id === "00000000-0000-0000-0000-000000000000");

  let docTyped: Document | null = null;
  let analysis: DetailedAnalysis | null = null;
  let report: ReportRow | null = null;

  if (isDemo) {
    const demoData = getDemoDocumentAndAnalysis(documentId);
    docTyped = demoData.document;
    analysis = demoData.analysis;
  } else {
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
          <div className="rounded-2xl bg-rose-50 border border-rose-200 p-4 text-rose-600 mb-4 shadow-2xs">
            <FileQuestion className="h-8 w-8 text-rose-600" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">Document Not Found</h1>
          <p className="mt-2 text-sm text-secondary max-w-md">
            The contract you requested does not exist or you do not have permission to access it.
          </p>
          <Link
            href="/dashboard"
            className="mt-6 inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white hover:bg-accent-hover transition-colors shadow-xs"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Return to Dashboard</span>
          </Link>
        </main>
      );
    }

    docTyped = document as Document;

    // Fetch analysis and report data in parallel
    const [analysisRes, reportRes] = await Promise.all([
      getAnalysis(documentId),
      getReport(documentId),
    ]);

    analysis = analysisRes.analysis ?? null;
    report = reportRes.report ?? null;
  }

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-4 sm:px-6 py-8">
      {!analysis ? (
        <UnanalyzedDocumentView document={docTyped} />
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