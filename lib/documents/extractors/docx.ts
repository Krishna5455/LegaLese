import mammoth from "mammoth";

import { cleanDocumentText } from "@/lib/documents/cleaner";
import type { DocumentSection, ExtractionResult } from "@/types/processing";

export async function extractTextFromDocx(
  buffer: Buffer,
): Promise<ExtractionResult> {
  if (!buffer || buffer.length === 0) {
    throw new Error("DOCX buffer is empty.");
  }

  try {
    const rawResult = await mammoth.extractRawText({ buffer });
    const rawText = rawResult.value || "";

    const cleanedText = cleanDocumentText(rawText);
    const wordCount = cleanedText.trim()
      ? cleanedText.trim().split(/\s+/).filter(Boolean).length
      : 0;

    if (wordCount < 10) {
      throw new Error(
        "This DOCX document contains insufficient text for contract analysis (fewer than 10 words). Please upload a valid readable contract.",
      );
    }

    const paragraphs = cleanedText.split(/\n\n+/).filter(Boolean);
    const sections: DocumentSection[] = paragraphs.map((para, index) => ({
      sectionIndex: index,
      text: para.trim(),
    }));

    return {
      sections,
      fullText: cleanedText,
    };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);

    if (msg.includes("insufficient text")) {
      throw new Error(msg);
    }

    console.error("[LegaLese/extractTextFromDocx] Parser error:", error);
    throw new Error(
      "Unable to read this document. The DOCX file appears to be corrupted or invalid.",
    );
  }
}
