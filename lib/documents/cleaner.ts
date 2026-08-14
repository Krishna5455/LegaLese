/**
 * Text normalization utility for legal documents.
 * Normalizes extraction artifacts without altering legal phrasing or clause structure.
 */

export function cleanDocumentText(text: string): string {
  if (!text) return "";

  return (
    text
      // 1. Normalize carriage returns to standard line feeds
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      // 2. Normalize horizontal whitespace (tabs and multiple spaces) on individual lines
      .replace(/[^\S\n]+/g, " ")
      // 3. Remove trailing whitespace from lines
      .replace(/[^\S\n]+$/gm, "")
      // 4. Remove leading whitespace from lines unless it's an indentation
      .replace(/^[ \t]+/gm, (match) => (match.length > 4 ? "    " : match))
      // 5. Normalize multiple consecutive blank lines to at most two newlines
      .replace(/\n{3,}/g, "\n\n")
      // 6. Trim start and end
      .trim()
  );
}

export function countWords(text: string): number {
  if (!text || !text.trim()) return 0;
  return text.trim().split(/\s+/).filter(Boolean).length;
}
