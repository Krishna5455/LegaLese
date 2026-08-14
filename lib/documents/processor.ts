import { countWords } from "@/lib/documents/cleaner";
import { extractTextFromDocx } from "@/lib/documents/extractors/docx";
import { extractTextFromPdf } from "@/lib/documents/extractors/pdf";
import { extractTextFromTxt } from "@/lib/documents/extractors/txt";
import { getFileExtension } from "@/lib/documents/validation";
import type { ProcessedDocument } from "@/types/processing";

export type ProcessDocumentParams = {
  documentId: string;
  filename: string;
  documentType?: string | null;
  buffer: Buffer;
};

export async function processDocumentBuffer({
  documentId,
  filename,
  documentType,
  buffer,
}: ProcessDocumentParams): Promise<ProcessedDocument> {
  if (!buffer || buffer.length === 0) {
    throw new Error("Cannot process an empty document buffer.");
  }

  // Determine type from documentType or file extension
  let resolvedType: "PDF" | "DOCX" | "TXT";
  const typeUpper = (documentType || "").toUpperCase();

  if (typeUpper === "PDF") {
    resolvedType = "PDF";
  } else if (typeUpper === "DOCX" || typeUpper === "DOC") {
    resolvedType = "DOCX";
  } else if (typeUpper === "TXT") {
    resolvedType = "TXT";
  } else {
    const ext = getFileExtension(filename);
    if (ext === ".pdf") resolvedType = "PDF";
    else if (ext === ".docx") resolvedType = "DOCX";
    else if (ext === ".txt") resolvedType = "TXT";
    else {
      throw new Error(
        `Unsupported document format for text extraction: ${filename}`,
      );
    }
  }

  let extraction;

  switch (resolvedType) {
    case "PDF":
      extraction = await extractTextFromPdf(buffer);
      break;
    case "DOCX":
      extraction = await extractTextFromDocx(buffer);
      break;
    case "TXT":
      extraction = extractTextFromTxt(buffer);
      break;
    default:
      throw new Error(`Unhandled document type: ${resolvedType}`);
  }

  if (!extraction || !extraction.fullText || !extraction.fullText.trim()) {
    throw new Error(
      `No readable text could be extracted from "${filename}". The document may be empty or contain non-extractable scanned images.`,
    );
  }

  const processed: ProcessedDocument = {
    documentId,
    filename,
    documentType: resolvedType,
    extractedAt: new Date().toISOString(),
    pageCount: extraction.pageCount,
    characterCount: extraction.fullText.length,
    wordCount: countWords(extraction.fullText),
    sections: extraction.sections,
    fullText: extraction.fullText,
  };

  return processed;
}
