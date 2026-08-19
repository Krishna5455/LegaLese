"use server";

import { revalidatePath } from "next/cache";

import { generateFreelanceAgreementWithGemini } from "@/lib/ai/generation";
import {
  generatedDocumentDownloadFilename,
  generatedDocumentToMarkdown,
} from "@/lib/generation/export";
import { exportPdf } from "@/lib/generation/export-pdf";
import { exportDocx } from "@/lib/generation/export-docx";
import { parseFreelanceAgreementInput } from "@/lib/generation/freelance-agreement-schema";
import { createClient } from "@/lib/supabase/server";
import type {
  GeneratedDocumentContent,
  GeneratedDocumentRow,
} from "@/types/generation";

export type GenerateDocumentResult = {
  success?: boolean;
  error?: string;
  documentId?: string;
};

export type GetGeneratedDocumentResult = {
  document?: GeneratedDocumentRow | null;
  error?: string;
};

export type DownloadGeneratedDocumentResult = {
  success?: boolean;
  error?: string;
  content?: string;
  filename?: string;
};

export type ExportDocumentFormat = "pdf" | "docx" | "md";

export type ExportDocumentResult = {
  success?: boolean;
  error?: string;
  data?: string;
  filename?: string;
  mimeType?: string;
  isBase64?: boolean;
};

export async function generateFreelanceAgreement(
  input: unknown,
): Promise<GenerateDocumentResult> {
  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to create documents." };
  }

  const parsed = parseFreelanceAgreementInput(input);
  if (!parsed.success) {
    return { error: parsed.error };
  }

  try {
    const { output, modelUsed } = await generateFreelanceAgreementWithGemini(
      parsed.data,
    );

    const { data: row, error: dbError } = await supabase
      .from("generated_documents")
      .insert({
        user_id: user.id,
        document_type: output.documentType,
        title: output.title,
        input_data: parsed.data,
        generated_content: output,
        model: modelUsed,
        status: "draft",
      })
      .select("*")
      .single();

    if (dbError) {
      console.error("[LegaLese/generateFreelanceAgreement] DB error:", dbError);
      return {
        error:
          "Your agreement was generated but could not be saved. Please try again.",
      };
    }

    revalidatePath("/dashboard/create");

    return {
      success: true,
      documentId: (row as GeneratedDocumentRow).id,
    };
  } catch (err) {
    const message =
      err instanceof Error
        ? err.message
        : "An unexpected error occurred while generating your agreement.";
    console.error("[LegaLese/generateFreelanceAgreement] Error:", message);
    return { error: message };
  }
}

export async function getGeneratedDocument(
  documentId: string,
): Promise<GetGeneratedDocumentResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to view documents." };
  }

  const { data, error } = await supabase
    .from("generated_documents")
    .select("*")
    .eq("id", documentId)
    .eq("user_id", user.id)
    .single();

  if (error || !data) {
    return {
      error: "Generated document not found or you do not have permission to view it.",
    };
  }

  return { document: data as GeneratedDocumentRow };
}

export async function downloadGeneratedDocument(
  documentId: string,
): Promise<DownloadGeneratedDocumentResult> {
  const { document, error } = await getGeneratedDocument(documentId);

  if (error || !document) {
    return { error: error ?? "Document not found." };
  }

  const content = document.generated_content as GeneratedDocumentContent;
  const markdown = generatedDocumentToMarkdown(content, document.created_at);

  return {
    success: true,
    content: markdown,
    filename: generatedDocumentDownloadFilename(content.title, "md"),
  };
}

export async function exportGeneratedDocument(
  documentId: string,
  format: ExportDocumentFormat = "pdf",
): Promise<ExportDocumentResult> {
  console.log(
    `[LegaLese/ExportAction] Export requested | docId: ${documentId} | format: ${format}`,
  );

  const { document, error } = await getGeneratedDocument(documentId);

  if (error || !document) {
    console.warn(
      `[LegaLese/ExportAction] Document lookup failed | docId: ${documentId} | error: ${error}`,
    );
    return { error: error ?? "Document not found or access denied." };
  }

  const content = document.generated_content as GeneratedDocumentContent;
  const createdAt = document.created_at;

  if (!content || !content.title || !Array.isArray(content.sections)) {
    console.error("[LegaLese/ExportAction] Invalid document payload structure");
    return { error: "Invalid generated document payload structure." };
  }

  try {
    if (format === "pdf") {
      console.log("[LegaLese/ExportAction] Generating PDF buffer...");
      const pdfBuffer = await exportPdf(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "pdf");

      console.log(
        `[LegaLese/ExportAction] PDF export success | size: ${pdfBuffer.length} bytes`,
      );

      return {
        success: true,
        data: pdfBuffer.toString("base64"),
        filename,
        mimeType: "application/pdf",
        isBase64: true,
      };
    }

    if (format === "docx") {
      console.log("[LegaLese/ExportAction] Generating DOCX buffer...");
      const docxBuffer = await exportDocx(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "docx");

      console.log(
        `[LegaLese/ExportAction] DOCX export success | size: ${docxBuffer.length} bytes`,
      );

      return {
        success: true,
        data: docxBuffer.toString("base64"),
        filename,
        mimeType:
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        isBase64: true,
      };
    }

    // Markdown format
    console.log("[LegaLese/ExportAction] Generating Markdown export...");
    const markdown = generatedDocumentToMarkdown(content, createdAt);
    const filename = generatedDocumentDownloadFilename(content.title, "md");

    return {
      success: true,
      data: markdown,
      filename,
      mimeType: "text/markdown; charset=utf-8",
      isBase64: false,
    };
  } catch (err) {
    const errorMessage =
      err instanceof Error ? err.message : "An unexpected export error occurred.";
    console.error(
      `[LegaLese/ExportAction] Export exception | format: ${format} | error: ${errorMessage}`,
    );
    return { error: `Failed to generate ${format.toUpperCase()} export: ${errorMessage}` };
  }
}

export async function listGeneratedDocuments(): Promise<{
  documents?: GeneratedDocumentRow[];
  error?: string;
}> {
  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to view documents." };
  }

  const { data, error } = await supabase
    .from("generated_documents")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return { error: "Unable to load your generated documents." };
  }

  return { documents: (data as GeneratedDocumentRow[]) ?? [] };
}
