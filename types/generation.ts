export type DocumentTypeId = "freelance_service_agreement";

export type GeneratedSection = {
  id: string;
  title: string;
  content: string;
  order: number;
};

export type GeneratedDocumentParties = {
  freelancerName: string;
  clientName: string;
  clientAddress?: string | null;
};

export type GeneratedDocumentContent = {
  title: string;
  documentType: DocumentTypeId;
  parties: GeneratedDocumentParties;
  sections: GeneratedSection[];
  disclaimer: string;
};

export type GeneratedDocumentRow = {
  id: string;
  user_id: string;
  document_type: DocumentTypeId;
  title: string;
  input_data: Record<string, unknown>;
  generated_content: GeneratedDocumentContent;
  model: string | null;
  status: string;
  created_at: string;
  updated_at: string;
};

export type FreelanceAgreementInput = {
  freelancerName: string;
  clientName: string;
  clientAddress?: string;
  servicesDescription: string;
  deliverables: string;
  startDate: string;
  completionDate: string;
  projectFee: string;
  paymentStructure: string;
  paymentSchedule: string;
  currency: string;
  noticePeriod: string;
  earlyTerminationWork: string;
  ipOwnership: string;
  freelancerReusableMaterials: string;
  confidentialityRequired: "yes" | "no";
  jurisdiction: string;
};

export type DocumentTypeDefinition = {
  id: DocumentTypeId;
  label: string;
  description: string;
};
