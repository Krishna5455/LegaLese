import { PDFParse } from "pdf-parse";

import { cleanDocumentText } from "@/lib/documents/cleaner";
import type { DocumentSection, ExtractionResult } from "@/types/processing";

export async function extractTextFromPdf(
  buffer: Buffer,
): Promise<ExtractionResult> {
  if (!buffer || buffer.length === 0) {
    throw new Error("PDF buffer is empty.");
  }

  const parser = new PDFParse({ data: buffer });

  try {
    const textResult = await parser.getText();
    const pages = textResult.pages || [];
    const sections: DocumentSection[] = [];
    const fullTextParts: string[] = [];

    if (pages.length > 0) {
      pages.forEach((page, index) => {
        const pageNumber = page.num ?? index + 1;
        const cleanedPageText = cleanDocumentText(page.text || "");
        if (cleanedPageText) {
          sections.push({
            sectionIndex: sections.length,
            pageNumber,
            text: cleanedPageText,
          });
          fullTextParts.push(cleanedPageText);
        }
      });
    } else if (textResult.text) {
      const cleaned = cleanDocumentText(textResult.text);
      if (cleaned) {
        sections.push({
          sectionIndex: 0,
          pageNumber: 1,
          text: cleaned,
        });
        fullTextParts.push(cleaned);
      }
    }

    const fullText = fullTextParts.join("\n\n");
    const pageCount = textResult.total ?? (pages.length > 0 ? pages.length : 1);

    return {
      pageCount,
      sections,
      fullText,
    };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to parse PDF document: ${msg}`);
  } finally {
    await parser.destroy().catch(() => {});
  }
}
