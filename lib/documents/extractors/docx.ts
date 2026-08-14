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
    if (!cleanedText) {
      return {
        sections: [],
        fullText: "",
      };
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
    throw new Error(`Failed to parse DOCX document: ${msg}`);
  }
}
