"use server";

import { reviewGeneratedDocumentWithGemini } from "@/lib/ai/review";
import type { DocumentReview } from "@/lib/ai/review-schema";
import { getGeneratedDocument } from "@/lib/actions/generated-documents";
import type { GeneratedDocumentContent } from "@/types/generation";

export type ReviewDocumentResult = {
  success?: boolean;
  error?: string;
  review?: DocumentReview;
  modelUsed?: string;
};

export async function reviewGeneratedDocumentAction(
  documentId: string,
): Promise<ReviewDocumentResult> {
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
    const { output, modelUsed } = await reviewGeneratedDocumentWithGemini(content);

    return {
      success: true,
      review: output,
      modelUsed,
    };
  } catch (err) {
    console.error("[LegaLese/ReviewAction] Technical error:", err);
    return {
      error:
        "We could not generate the contract review right now. Please try again in a moment.",
    };
  }
}
