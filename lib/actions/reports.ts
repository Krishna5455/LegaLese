"use server";

import { revalidatePath } from "next/cache";

import { getAnalysis } from "@/lib/actions/analyses";
import { getRiskLabel } from "@/lib/ai/scorer";
import { createClient } from "@/lib/supabase/server";
import type { ReportRow } from "@/types/analysis";

export type GenerateReportResult = {
  success?: boolean;
  error?: string;
  report?: ReportRow;
};

export type DownloadReportResult = {
  success?: boolean;
  error?: string;
  content?: string;
  filename?: string;
};

export type GetReportResult = {
  report?: ReportRow | null;
  error?: string;
};

/**
 * Builds a structured Markdown summary report for an analyzed document,
 * uploads it to the private Storage bucket ('contracts'), and records an entry
 * in the 'reports' table.
 */
export async function generateReport(
  documentId: string,
): Promise<GenerateReportResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to generate reports." };
  }

  // 1. Verify document ownership
  const { data: doc, error: docError } = await supabase
    .from("documents")
    .select("*")
    .eq("id", documentId)
    .eq("user_id", user.id)
    .single();

  if (docError || !doc) {
    return { error: "Document not found or access denied." };
  }

  // 2. Fetch full analysis details
  const { analysis, error: analysisError } = await getAnalysis(documentId);
  if (analysisError || !analysis) {
    return { error: analysisError || "No complete analysis found for this document." };
  }

  // 3. Format Markdown report
  const risk = getRiskLabel(analysis.risk_score);
  const createdDate = new Date(analysis.created_at).toLocaleDateString("en-US", {
    dateStyle: "full",
  });

  const markdownLines: string[] = [
    `# LegaLese Contract Review Report`,
    ``,
    `**Document Name:** ${doc.filename}`,
    `**Document Type:** ${doc.document_type ? doc.document_type.toUpperCase() : "N/A"}`,
    `**Analysis Date:** ${createdDate}`,
    `**AI Model:** ${analysis.model}`,
    `**Overall Risk Assessment:** ${risk.label.toUpperCase()}`,
    ``,
    `---`,
    ``,
    `## Executive Summary`,
    ``,
    analysis.summary ?? "No summary provided.",
    ``,
    `---`,
    ``,
    `## Identified Findings & Risk Factors (${analysis.findings.length})`,
    ``,
  ];

  if (analysis.findings.length === 0) {
    markdownLines.push(`*No specific risk findings identified.*`, ``);
  } else {
    analysis.findings.forEach((f, idx) => {
      markdownLines.push(
        `### ${idx + 1}. [${f.risk_level.toUpperCase()}] ${f.category}`,
        ``,
        `**Explanation:** ${f.explanation}`,
      );
      if (f.why_it_matters) {
        markdownLines.push(`**Why It Matters:** ${f.why_it_matters}`);
      }
      if (f.questions && f.questions.length > 0) {
        markdownLines.push(
          `**Questions to Consider:**`,
          ...f.questions.map((q) => `  - ${q}`),
        );
      }
      if (f.clause) {
        markdownLines.push(
          `**Linked Source Clause (${f.clause.section}${f.clause.clause_number ? ` - Clause ${f.clause.clause_number}` : ""}${f.clause.page_number != null ? `, Page ${f.clause.page_number}` : ""}):**`,
          `> "${f.clause.text}"`,
        );
      }
      markdownLines.push(``);
    });
  }

  markdownLines.push(`---`, ``, `## Extracted Key Clauses (${analysis.clauses.length})`, ``);
  if (analysis.clauses.length === 0) {
    markdownLines.push(`*No key clauses extracted.*`, ``);
  } else {
    analysis.clauses.forEach((c) => {
      markdownLines.push(
        `- **${c.section}**${c.clause_number ? ` (Clause ${c.clause_number})` : ""}${c.page_number != null ? ` [Page ${c.page_number}]` : ""}:`,
        `  > "${c.text}"`,
        ``,
      );
    });
  }

  markdownLines.push(`---`, ``, `## Defined Terms (${analysis.key_terms.length})`, ``);
  if (analysis.key_terms.length === 0) {
    markdownLines.push(`*No defined terms extracted.*`, ``);
  } else {
    analysis.key_terms.forEach((kt) => {
      markdownLines.push(`- **${kt.term}**: ${kt.value}`);
      if (kt.clause) {
        markdownLines.push(`  *Source:* "${kt.clause.text}"`);
      }
      markdownLines.push(``);
    });
  }

  markdownLines.push(`---`, ``, `## Obligations & Duties (${analysis.obligations.length})`, ``);
  if (analysis.obligations.length === 0) {
    markdownLines.push(`*No specific obligations extracted.*`, ``);
  } else {
    analysis.obligations.forEach((o) => {
      const partyStr = o.responsible_party ? `[${o.responsible_party}] ` : "";
      const deadlineStr = o.deadline ? ` (Deadline: ${o.deadline})` : "";
      markdownLines.push(`- ${partyStr}${o.description}${deadlineStr}`);
      if (o.clause) {
        markdownLines.push(`  *Source:* "${o.clause.text}"`);
      }
      markdownLines.push(``);
    });
  }

  markdownLines.push(
    `---`,
    ``,
    `*Disclaimer: LegaLese provides AI-assisted contract review for informational purposes only. LegaLese is not a law firm and does not provide legal advice. Always consult a qualified attorney for legal decisions.*`,
  );

  const reportMarkdown = markdownLines.join("\n");
  const reportPath = `${user.id}/reports/${documentId}_report.md`;
  const reportBuffer = Buffer.from(reportMarkdown, "utf-8");

  // 4. Save to private Storage bucket 'contracts' using allowed MIME type 'text/plain'
  const { error: uploadError } = await supabase.storage
    .from("contracts")
    .upload(reportPath, reportBuffer, {
      contentType: "text/plain",
      upsert: true,
    });

  if (uploadError) {
    return { error: `Failed to save report to storage: ${uploadError.message}` };
  }

  // 5. Check if report row exists to prevent duplicate report entries
  const { data: existingReports } = await supabase
    .from("reports")
    .select("*")
    .eq("document_id", documentId)
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(1);

  let reportRow: ReportRow;

  if (existingReports && existingReports.length > 0) {
    const { data: updated, error: updateErr } = await supabase
      .from("reports")
      .update({ file_path: reportPath })
      .eq("id", existingReports[0].id)
      .select("*")
      .single();

    if (updateErr) {
      return { error: `Failed to update report record: ${updateErr.message}` };
    }
    reportRow = updated as ReportRow;
  } else {
    const { data: inserted, error: insertErr } = await supabase
      .from("reports")
      .insert({
        document_id: documentId,
        user_id: user.id,
        file_path: reportPath,
      })
      .select("*")
      .single();

    if (insertErr) {
      return { error: `Failed to record report in database: ${insertErr.message}` };
    }
    reportRow = inserted as ReportRow;
  }

  revalidatePath(`/dashboard/documents/${documentId}`);
  revalidatePath("/dashboard");

  return {
    success: true,
    report: reportRow,
  };
}


/**
 * Retrieves the latest report record for a document.
 */
export async function getReport(
  documentId: string,
): Promise<GetReportResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to view reports." };
  }

  const { data: reports, error: fetchErr } = await supabase
    .from("reports")
    .select("*")
    .eq("document_id", documentId)
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(1);

  if (fetchErr || !reports || reports.length === 0) {
    return { report: null };
  }

  return { report: reports[0] as ReportRow };
}

/**
 * Securely downloads the generated report content for an authenticated document owner.
 * Never exposes public URLs.
 */
export async function downloadReport(
  documentId: string,
): Promise<DownloadReportResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to download reports." };
  }

  // 1. Verify document ownership
  const { data: doc } = await supabase
    .from("documents")
    .select("filename")
    .eq("id", documentId)
    .eq("user_id", user.id)
    .single();

  if (!doc) {
    return { error: "Document not found or access denied." };
  }

  // 2. Fetch report record
  const { report, error: reportErr } = await getReport(documentId);
  if (reportErr || !report || !report.file_path) {
    return { error: reportErr || "No report generated yet. Click 'Generate Report' first." };
  }

  // 3. Download from private Storage bucket
  const { data: fileBlob, error: downloadErr } = await supabase.storage
    .from("contracts")
    .download(report.file_path);

  if (downloadErr || !fileBlob) {
    return { error: `Could not retrieve report file: ${downloadErr?.message || "File missing"}` };
  }

  const textContent = await fileBlob.text();
  const safeFilename = `${doc.filename.replace(/[^a-zA-Z0-9._-]/g, "_")}_Review_Report.md`;

  return {
    success: true,
    content: textContent,
    filename: safeFilename,
  };
}
