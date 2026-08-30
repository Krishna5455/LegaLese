"use server";

import { explainGeneratedDocumentWithGemini } from "@/lib/ai/explanation";
import type { DocumentExplanation } from "@/lib/ai/explanation-schema";
import { getGeneratedDocument } from "@/lib/actions/generated-documents";
import type { GeneratedDocumentContent } from "@/types/generation";

export type ExplainDocumentResult = {
  success?: boolean;
  error?: string;
  explanation?: DocumentExplanation;
  modelUsed?: string;
};

export async function explainGeneratedDocumentAction(
  documentId: string,
): Promise<ExplainDocumentResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const { document, error: fetchError } = await getGeneratedDocument(documentId);

  if (fetchError || !document) {
    return {
      error: fetchError ?? "Generated document not found or access denied.",
    };
  }

  const content = document.generated_content as GeneratedDocumentContent;

  if (!content || !content.title || !Array.isArray(content.sections)) {
    return { error: "Invalid generated document structure." };
  }

  try {
    const { output, modelUsed } = await explainGeneratedDocumentWithGemini(content);

    return {
      success: true,
      explanation: output,
      modelUsed,
    };
  } catch (err) {
    console.error("[LegaLese/ExplainAction] Technical error:", err);
    return {
      error:
        "We could not generate the plain-language explanation right now. Please try again in a moment.",
    };
  }
}
