"use server";

import { revalidatePath } from "next/cache";

import { generateFreelanceAgreementWithGemini } from "@/lib/ai/generation";
import {
  generatedDocumentDownloadFilename,
  generatedDocumentToMarkdown,
} from "@/lib/generation/export";
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
    filename: generatedDocumentDownloadFilename(content.title),
  };
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
