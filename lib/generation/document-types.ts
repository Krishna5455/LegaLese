import type { DocumentTypeDefinition, DocumentTypeId } from "@/types/generation";

export const DOCUMENT_TYPES: Record<DocumentTypeId, DocumentTypeDefinition> = {
  freelance_service_agreement: {
    id: "freelance_service_agreement",
    label: "Freelance Service Agreement",
    description:
      "A contract between a freelancer and a client for project-based services.",
  },
};

export const DOCUMENT_TYPE_LIST = Object.values(DOCUMENT_TYPES);

export function isDocumentTypeId(value: string): value is DocumentTypeId {
  return value in DOCUMENT_TYPES;
}
