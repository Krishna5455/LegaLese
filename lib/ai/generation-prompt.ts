import type { ValidatedFreelanceAgreementInput } from "@/lib/generation/freelance-agreement-schema";

const REQUIRED_SECTION_IDS = [
  "parties",
  "services_and_deliverables",
  "payment",
  "term_and_schedule",
  "termination",
  "intellectual_property",
  "confidentiality",
  "dispute_resolution",
  "miscellaneous",
] as const;

export function buildGenerationSystemInstruction(): string {
  return `You are a legal document drafting assistant for LegaLese.

Your task is to generate a DRAFT Freelance Service Agreement based ONLY on the user-supplied information.

CRITICAL RULES:
1. Generate a DRAFT document — not final legal advice.
2. Use ONLY facts provided by the user. Do NOT invent names, dates, amounts, addresses, or terms.
3. If required information is missing, use a clearly marked placeholder such as "[TO BE COMPLETED: description]" rather than inventing values.
4. Do NOT silently add user-specific facts that were not supplied.
5. Use professional legal-document structure with clear numbered or titled sections.
6. Use plain, understandable language where possible while maintaining appropriate contract tone.
7. Include standard clauses appropriate for a freelance service agreement (scope, payment, term, termination, IP, confidentiality if requested, dispute resolution, miscellaneous).
8. Do NOT claim the document is guaranteed to be legally sufficient or enforceable in all jurisdictions.
9. Include an appropriate disclaimer in the "disclaimer" field stating this is an AI-generated draft that may require professional legal review.
10. Respond with a single valid JSON object only. No markdown fences or prose outside JSON.

OUTPUT JSON STRUCTURE:
{
  "title": "Freelance Service Agreement between [Freelancer] and [Client]",
  "documentType": "freelance_service_agreement",
  "parties": {
    "freelancerName": "string",
    "clientName": "string",
    "clientAddress": "string or null"
  },
  "sections": [
    {
      "id": "stable_snake_case_id",
      "title": "Section Title",
      "content": "Full section text with numbered sub-clauses as appropriate",
      "order": 0
    }
  ],
  "disclaimer": "AI-generated draft disclaimer text"
}

REQUIRED SECTION IDs (use exactly these ids, in this order starting at order 0):
${REQUIRED_SECTION_IDS.map((id, i) => `${i + 1}. "${id}"`).join("\n")}

Each section "content" should be complete, ready-to-read contract language for that section.`;
}

export function buildFreelanceAgreementUserMessage(
  input: ValidatedFreelanceAgreementInput,
): string {
  const confidentialityNote =
    input.confidentialityRequired === "yes"
      ? "Include a confidentiality / non-disclosure section with reasonable standard terms."
      : "Do NOT include a confidentiality section, or state that the parties have not agreed to confidentiality obligations.";

  return `Generate a Freelance Service Agreement draft using the following user-provided information.

--- PARTIES ---
Freelancer full name: ${input.freelancerName}
Client/company name: ${input.clientName}
Client address: ${input.clientAddress?.trim() || "[Not provided — use placeholder if needed in document]"}

--- WORK ---
Description of services: ${input.servicesDescription}
Project/deliverables: ${input.deliverables}
Start date: ${input.startDate}
Expected completion date: ${input.completionDate}

--- PAYMENT ---
Total/project fee: ${input.projectFee}
Currency: ${input.currency}
Payment structure: ${input.paymentStructure}
Payment due date / schedule: ${input.paymentSchedule}

--- TERMINATION ---
Notice period: ${input.noticePeriod}
Completed work if agreement ends early: ${input.earlyTerminationWork}

--- INTELLECTUAL PROPERTY ---
Who owns final work after payment: ${input.ipOwnership}
Freelancer rights to reusable/general materials: ${input.freelancerReusableMaterials}

--- CONFIDENTIALITY ---
Confidentiality required: ${input.confidentialityRequired}
${confidentialityNote}

--- DISPUTE / JURISDICTION ---
Governing location for legal disputes: ${input.jurisdiction}

Respond strictly with valid JSON matching the required schema.`;
}
