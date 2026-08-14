import { cleanDocumentText } from "@/lib/documents/cleaner";
import type { DocumentSection, ExtractionResult } from "@/types/processing";

export function extractTextFromTxt(buffer: Buffer): ExtractionResult {
  if (!buffer || buffer.length === 0) {
    throw new Error("Text file buffer is empty.");
  }

  const rawText = buffer.toString("utf-8");
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
}
