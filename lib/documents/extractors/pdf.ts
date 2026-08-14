import { extractText, getDocumentProxy } from "unpdf";

import { cleanDocumentText } from "@/lib/documents/cleaner";
import type { DocumentSection, ExtractionResult } from "@/types/processing";

export async function extractTextFromPdf(
  buffer: Buffer,
): Promise<ExtractionResult> {
  if (!buffer || buffer.length === 0) {
    throw new Error("PDF buffer is empty.");
  }

  try {
    const uint8Array = new Uint8Array(buffer);
    const pdf = await getDocumentProxy(uint8Array);
    const { totalPages, text } = await extractText(pdf, { mergePages: false });

    const pagesArray = Array.isArray(text) ? text : [text];
    const sections: DocumentSection[] = [];
    const fullTextParts: string[] = [];

    pagesArray.forEach((pageText, index) => {
      const pageNumber = index + 1;
      const cleaned = cleanDocumentText(pageText || "");
      if (cleaned) {
        sections.push({
          sectionIndex: sections.length,
          pageNumber,
          text: cleaned,
        });
        fullTextParts.push(cleaned);
      }
    });

    const fullText = fullTextParts.join("\n\n");
    const pageCount = totalPages || pagesArray.length || 1;

    return {
      pageCount,
      sections,
      fullText,
    };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to parse PDF document: ${msg}`);
  }
}
