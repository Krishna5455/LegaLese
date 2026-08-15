import type { ProcessedDocument } from "@/types/processing";

export const MAX_ANALYSIS_CHARS = parseInt(
  process.env.MAX_ANALYSIS_CHARS ?? "200000",
  10,
);

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
    "\n\n[Note: Document truncated at 200,000 characters for analysis.]";

  return { text: truncated, wasTruncated: true };
}

export function buildSystemInstruction(): string {
  return `You are a contract analysis assistant for LegaLese, a tool that helps users understand legal documents.

Your role is to help users understand what a contract says — NOT to provide legal advice or act as a lawyer.

CRITICAL RULES:
1. Analyze ONLY the contract text provided. Do not assume facts or clauses not present.
2. Extract notable verbatim clauses into the "clauses" array.
3. Link findings, key terms, and obligations to these clauses using zero-based "clauseIndex".
4. Store any questions the user should consider asking regarding a finding in the finding's "questions" string array.
5. Do NOT calculate an overall risk score — that is calculated separately by application logic.

OUTPUT FORMAT:
Respond with a single valid JSON object matching this structure. No preamble or postscript markdown or prose.

{
  "summary": "string (plain-English overview of the contract, 2-4 sentences)",
  "clauses": [
    {
      "section": "string (e.g. Section 4 - Intellectual Property)",
      "clauseNumber": "string or null (e.g. 4.1)",
      "text": "string (verbatim quote of the key clause from contract)",
      "pageNumber": number or null (1-indexed page number if available)
    }
  ],
  "findings": [
    {
      "category": "string (e.g. Liability, IP Ownership, Payment, Termination)",
      "riskLevel": "informational | low | medium | high (MUST be strictly one of these four values)",
      "explanation": "string (plain English explanation of what this means)",
      "whyItMatters": "string or null (why the user should care)",
      "clauseIndex": number or null (0-based index of the matching entry in clauses array),
      "questions": ["string (questions the user should ask a lawyer or counterparty)"],
      "confidence": number or null (0.0 to 1.0)
    }
  ],
  "keyTerms": [
    {
      "term": "string (the defined term)",
      "value": "string (plain-English definition or meaning in this contract)",
      "clauseIndex": number or null
    }
  ],
  "obligations": [
    {
      "description": "string (what must be done)",
      "responsibleParty": "string or null (e.g. Client, Provider, Both)",
      "deadline": "string or null (e.g. 30 days post termination)",
      "clauseIndex": number or null
    }
  ]
}`;
}

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

  return `Please analyze the following contract text.

--- DOCUMENT INFORMATION ---
${docInfo}

--- CONTRACT TEXT ---
${contractText}
--- END OF CONTRACT TEXT ---

Respond strictly with valid JSON.`;
}
