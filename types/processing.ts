export type DocumentSection = {
  sectionIndex: number;
  pageNumber?: number; // Present for PDFs (1-indexed)
  heading?: string;
  text: string;
};

export type ProcessedDocument = {
  documentId: string;
  filename: string;
  documentType: "PDF" | "DOCX" | "TXT";
  extractedAt: string; // ISO 8601 string
  pageCount?: number;
  characterCount: number;
  wordCount: number;
  sections: DocumentSection[];
  fullText: string;
};

export type ExtractionResult = {
  pageCount?: number;
  sections: DocumentSection[];
  fullText: string;
};
