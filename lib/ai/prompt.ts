import type { ProcessedDocument } from "@/types/processing";

/**
 * Maximum number of characters sent to the AI model.
 * Contracts exceeding this are truncated. Configurable via environment variable.
 * Default: 200,000 characters (~150,000 words) — covers most multi-page contracts.
 */
export const MAX_ANALYSIS_CHARS = parseInt(
  process.env.MAX_ANALYSIS_CHARS ?? "200000",
  10,
);

/**
 * Prepares the contract text for AI analysis.
 * Applies the character cap and returns the truncated text plus a flag.
 */
export function prepareContractText(doc: ProcessedDocument): {
  text: string;
  wasTruncated: boolean;
} {
  const { fullText } = doc;

  if (fullText.length <= MAX_ANALYSIS_CHARS) {
    return { text: fullText, wasTruncated: false };
  }

  const truncated =
    fullText.slice(0, MAX_ANALYSIS_CHARS) +
    "\n\n[Note: This document was truncated at 200,000 characters for analysis. Content beyond this point was not analyzed.]";

  return { text: truncated, wasTruncated: true };
}

/**
 * Builds the system instruction sent to Gemini.
 * Sets the behavioral context and constraints for the model.
 */
export function buildSystemInstruction(): string {
  return `You are a contract analysis assistant for LegaLese, a tool that helps people understand legal documents.

Your role is to help users understand what a contract says — NOT to provide legal advice or act as a lawyer.
Always include a note that users should seek qualified legal advice for significant decisions.

CRITICAL RULES you must follow without exception:
1. Analyze ONLY the contract text provided. Do not assume facts, clauses, or terms that are not present in the document.
2. Do NOT invent evidence text. If you cannot find a verbatim or near-verbatim passage supporting a finding, set evidenceText to null.
3. Do NOT fabricate page numbers. Only include a pageNumber if the document text provides clear page indicators.
4. Do NOT invent party names, dates, amounts, or obligations that are not in the document.
5. If the document does not contain enough information for a finding, omit that finding rather than guessing.
6. Do NOT provide the overall risk score — that is calculated separately. Focus on individual findings.

OUTPUT FORMAT:
You must respond with a single valid JSON object matching this exact structure. No prose before or after the JSON.

{
  "summary": "string (plain-English summary of what this contract is, 2-5 sentences)",
  "findings": [
    {
      "category": "string (e.g. Liability, IP Ownership, Termination, Payment, Non-Compete, Confidentiality)",
      "riskLevel": "info | low | medium | high | critical",
      "explanation": "string (what this means in plain language)",
      "whyItMatters": "string or null (why this matters to the reader)",
      "evidenceText": "string or null (verbatim or near-verbatim quote from the contract, or null if not found)",
      "sourceSection": "string or null (section heading or name, if identifiable)",
      "pageNumber": number or null (1-indexed page; null if not available),
      "sectionIndex": number or null (0-indexed section number from the document),
      "confidence": "low | medium | high"
    }
  ],
  "keyTerms": [
    {
      "term": "string (the defined or important term)",
      "definition": "string (plain-English explanation of what this term means in this contract)",
      "sourceSection": "string or null",
      "pageNumber": number or null,
      "sectionIndex": number or null
    }
  ],
  "obligations": [
    {
      "party": "string or null (who has this obligation, e.g. Client, Service Provider, Both)",
      "description": "string (what the obligation is, in plain language)",
      "sourceSection": "string or null",
      "pageNumber": number or null,
      "sectionIndex": number or null
    }
  ],
  "questions": [
    {
      "questionText": "string (a question the user should consider asking a lawyer or the other party)",
      "context": "string or null (why this question is worth asking)"
    }
  ]
}

Guidelines:
- findings: identify 3–15 notable clauses. Use risk levels: info (neutral/standard), low (minor concern), medium (noteworthy), high (significant risk), critical (major red flag).
- keyTerms: identify 5–15 important defined or recurring terms.
- obligations: identify 3–10 clear duties or responsibilities.
- questions: suggest 3–7 questions the user may want to clarify.
- Keep all text clear and accessible to a non-lawyer.
- Do not repeat the same point across findings, keyTerms, and obligations.`;
}

/**
 * Builds the full user message containing the contract text.
 */
export function buildUserMessage(
  doc: ProcessedDocument,
  contractText: string,
): string {
  const hasPagination = doc.documentType === "PDF" && doc.pageCount != null;
  const docInfo = [
    `Filename: ${doc.filename}`,
    `Document type: ${doc.documentType}`,
    hasPagination ? `Pages: ${doc.pageCount}` : null,
    `Word count: ${doc.wordCount.toLocaleString()}`,
  ]
    .filter(Boolean)
    .join("\n");

  return `Please analyze the following contract.

--- DOCUMENT INFORMATION ---
${docInfo}

--- CONTRACT TEXT ---
${contractText}
--- END OF CONTRACT TEXT ---

Respond with the JSON analysis only.`;
}
