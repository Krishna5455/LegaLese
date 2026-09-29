import type {
  LegalCalendarEvent,
  LegalFlowRow,
  LegalFlowStep,
} from "@/types/legalflow";
import type { VaultDocument } from "@/components/vault/DocumentVaultView";
import type { Document } from "@/types/database";
import type { DetailedAnalysis } from "@/types/analysis";
import type { GeneratedDocumentRow } from "@/types/generation";

export function isDemoMode(): boolean {
  return process.env.NEXT_PUBLIC_DEMO_MODE === "true";
}

export const DEMO_USER = {
  id: "00000000-0000-0000-0000-000000000000",
  email: "demo@legalese.app",
  user_metadata: { full_name: "Demo Guest" },
  app_metadata: { provider: "email" },
  aud: "authenticated",
  created_at: "2026-09-28T00:00:00.000Z",
};

// In-memory demo store for ephemeral demo flows
const demoFlowStore = new Map<string, LegalFlowRow>();

export const DEFAULT_DEMO_FLOW: LegalFlowRow = {
  id: "demo-freelance-flow",
  user_id: DEMO_USER.id,
  title: "Freelance Website Agreement",
  intent_query: "I want to hire a freelancer for my website.",
  scenario_key: "hire_freelancer",
  current_stage: "review",
  status: "active",
  steps: [
    {
      id: "step-understand",
      title: "Understand Scope & Deliverables",
      description: "Specify service requirements, payment milestones, and intellectual property ownership boundaries.",
      stage: "understand",
      status: "completed",
      trustLevel: "ai_assistance",
      actionLabel: "View Scope Guide",
      category: "Requirements",
      badge: "AI Assisted",
    },
    {
      id: "step-prepare",
      title: "Prepare Freelance Agreement",
      description: "Generate a customized agreement with milestones, payment terms, and plain-English summary.",
      stage: "prepare",
      status: "completed",
      trustLevel: "ai_assistance",
      actionUrl: "/dashboard/create",
      actionLabel: "Generate Agreement",
      category: "Drafting",
      badge: "AI Generator",
    },
    {
      id: "step-review",
      title: "Review the liability clause",
      description: "Audit liability limits, IP indemnification, and late payment interest clauses for mutual fairness.",
      stage: "review",
      status: "next",
      trustLevel: "ai_assistance",
      actionUrl: "/dashboard/documents/demo-doc-1",
      actionLabel: "Review Clause",
      category: "Analysis",
      badge: "AI Pre-Sign Audit",
    },
    {
      id: "step-trust-check",
      title: "Trust Check: Professional Review",
      description: "Verify if liability limits or cross-border IP transfer warrant independent attorney verification.",
      stage: "trust_check",
      status: "recommended",
      trustLevel: "professional_review_recommended",
      actionLabel: "Request Attorney Review",
      category: "Trust Layer",
      badge: "Verified Expert",
      formalGuidance: "Professional review recommended for high-exposure liability and copyright buyout terms.",
    },
    {
      id: "step-formal-signing",
      title: "Obtain Signatures & Formal Process",
      description: "Execute bilateral signatures with the freelancer and record effective start date.",
      stage: "whats_next",
      status: "optional",
      trustLevel: "formal_process",
      actionLabel: "Execute Signatures",
      category: "Execution",
      badge: "Formal Process",
      formalGuidance: "Agreements require digital or written execution to establish enforceable contractual binding.",
    },
    {
      id: "step-track-milestones",
      title: "Track Payment & Delivery Deadlines",
      description: "Monitor milestones, payment schedules, and notice dates automatically in the Legal Calendar.",
      stage: "track",
      status: "optional",
      trustLevel: "ai_assistance",
      actionUrl: "/dashboard/calendar",
      actionLabel: "View Obligations",
      category: "Tracking",
      badge: "Calendar Sync",
    },
    {
      id: "step-vault-store",
      title: "Store in Documents Repository",
      description: "Archive signed agreements and AI audit reports in your secure workspace repository.",
      stage: "complete",
      status: "optional",
      trustLevel: "ai_assistance",
      actionUrl: "/dashboard/vault",
      actionLabel: "View in Repository",
      category: "Documents",
      badge: "Secure Repo",
    },
  ],
  trust_summary: {
    level: "professional_review_recommended",
    label: "Professional Review Recommended for High-Liability Clauses",
    reason: "IP assignment, indemnity, and liability caps in freelance retainers significantly impact ownership of deliverables.",
    formal_notes: [
      "Signatures from both parties are required before work commences.",
      "Ensure clear jurisdiction and dispute resolution clauses match local commercial law.",
    ],
  },
  created_at: new Date(Date.now() - 3600 * 1000 * 2).toISOString(),
  updated_at: new Date().toISOString(),
};

demoFlowStore.set(DEFAULT_DEMO_FLOW.id, DEFAULT_DEMO_FLOW);

export function getDemoFlow(id: string): LegalFlowRow | null {
  if (demoFlowStore.has(id)) {
    return demoFlowStore.get(id)!;
  }
  // If requesting any ID in demo mode, return or generate on the fly
  if (id.startsWith("demo-")) {
    return DEFAULT_DEMO_FLOW;
  }
  return demoFlowStore.get(id) || null;
}

export function saveDemoFlow(flow: LegalFlowRow): void {
  demoFlowStore.set(flow.id, flow);
}

export function listDemoFlows(): LegalFlowRow[] {
  return Array.from(demoFlowStore.values()).sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  );
}

export function updateDemoFlowStep(
  flowId: string,
  stepId: string,
  newStatus: LegalFlowStep["status"],
): LegalFlowRow | null {
  const flow = getDemoFlow(flowId) || DEFAULT_DEMO_FLOW;
  const updatedSteps = (flow.steps || []).map((s) => {
    if (s.id === stepId) {
      return { ...s, status: newStatus };
    }
    return s;
  });

  const allDone = updatedSteps.every((s) => s.status === "completed");
  const updatedFlow: LegalFlowRow = {
    ...flow,
    steps: updatedSteps,
    status: allDone ? "completed" : flow.status,
    updated_at: new Date().toISOString(),
  };

  demoFlowStore.set(updatedFlow.id, updatedFlow);
  return updatedFlow;
}

export const DEMO_CALENDAR_EVENTS: LegalCalendarEvent[] = [
  {
    id: "demo-evt-1",
    flowId: "demo-freelance-flow",
    title: "50% Advance Retainer Payment Due",
    description: "Source: Freelance Website Development Agreement",
    date: new Date(Date.now() + 86400000 * 3).toISOString().split("T")[0],
    responsibleParty: "Client (You)",
    type: "payment",
    status: "pending",
    amount: "$2,500 USD",
  },
  {
    id: "demo-evt-2",
    flowId: "demo-freelance-flow",
    title: "UI/UX Prototype Delivery Milestone",
    description: "Source: Freelance Website Development Agreement",
    date: new Date(Date.now() + 86400000 * 14).toISOString().split("T")[0],
    responsibleParty: "Freelancer",
    type: "delivery",
    status: "pending",
  },
  {
    id: "demo-evt-3",
    flowId: "demo-freelance-flow",
    title: "Final Code Handover & IP Assignment Execution",
    description: "Source: Freelance Website Development Agreement",
    date: new Date(Date.now() + 86400000 * 30).toISOString().split("T")[0],
    responsibleParty: "Both Parties",
    type: "deadline",
    status: "pending",
    amount: "$2,500 USD",
  },
  {
    id: "demo-evt-4",
    title: "90-Day Post-Launch Bug Fix Warranty Expiry",
    description: "Source: Freelance Website Development Agreement",
    date: new Date(Date.now() + 86400000 * 90).toISOString().split("T")[0],
    responsibleParty: "Freelancer",
    type: "renewal",
    status: "pending",
  },
];

export const DEMO_GENERATED_DOCUMENT: GeneratedDocumentRow = {
  id: "demo-doc-1",
  user_id: DEMO_USER.id,
  document_type: "freelance_service_agreement",
  title: "Freelance Website Development Agreement",
  input_data: {
    freelancerName: "Alex Rivera",
    clientName: "Acme Digital Corp",
    servicesDescription: "Website UI/UX Redesign and Next.js full-stack development",
    deliverables: "Interactive Figma prototype, production codebase, and deployment setup",
    startDate: "2026-10-01",
    completionDate: "2026-11-01",
    projectFee: "5000",
    paymentStructure: "milestone",
    paymentSchedule: "50% advance retainer upon signing ($2,500), 50% upon final acceptance ($2,500)",
    currency: "USD",
    noticePeriod: "14",
    earlyTerminationWork: "pro_rata",
    ipOwnership: "client_full",
    freelancerReusableMaterials: "standard_tools",
    confidentialityRequired: "yes",
    jurisdiction: "Delaware, USA",
  },
  generated_content: {
    title: "Freelance Website Development Agreement",
    documentType: "freelance_service_agreement",
    parties: {
      freelancerName: "Alex Rivera",
      clientName: "Acme Digital Corp",
      clientAddress: "100 Innovation Way, Suite 400, Wilmington, DE",
    },
    sections: [
      {
        id: "sec-1",
        title: "1. Scope of Work & Deliverables",
        content: "The Freelancer agrees to perform comprehensive website redesign and full-stack development services, including responsive design system architecture, Next.js page development, and production server deployment as outlined in Schedule A.",
        order: 1,
      },
      {
        id: "sec-2",
        title: "2. Payment Milestones & Compensation",
        content: "The Client shall pay a total fixed fee of $5,000 USD. An initial deposit of $2,500 USD (50%) shall be payable upon mutual signature. The remaining balance of $2,500 USD (50%) shall be payable upon final delivery and client acceptance.",
        order: 2,
      },
      {
        id: "sec-3",
        title: "3. Intellectual Property Rights",
        content: "Upon receipt of full payment, all bespoke deliverables, source code, and design assets created under this Agreement shall become the sole and exclusive property of the Client.",
        order: 3,
      },
      {
        id: "sec-4",
        title: "4. Confidentiality & Non-Disclosure",
        content: "Both parties agree to hold proprietary information, technical designs, and business data in strict confidence and shall not disclose such materials without prior written consent.",
        order: 4,
      },
      {
        id: "sec-5",
        title: "5. Termination & Notice",
        content: "Either party may terminate this Agreement with fourteen (14) days written notice. In the event of early termination, the Freelancer shall be compensated pro rata for work verified and delivered up to the effective termination date.",
        order: 5,
      },
    ],
    disclaimer: "This agreement was generated by LegaLese AI Intelligence based on user specifications. For high-stakes engagements, professional legal review is recommended.",
  },
  model: "gemini-2.5-flash",
  status: "completed",
  created_at: new Date(Date.now() - 86400000 * 2).toISOString(),
  updated_at: new Date().toISOString(),
};

export const DEMO_VAULT_DOCUMENTS: VaultDocument[] = [
  {
    id: "demo-doc-1",
    title: "Freelance Website Development Agreement",
    type: "Freelance Agreement",
    status: "Draft",
    isGenerated: true,
    createdAt: new Date(Date.now() - 86400000 * 2).toISOString(),
    updatedAt: new Date().toISOString(),
    flowId: "demo-freelance-flow",
    flowTitle: "Freelance Website Agreement",
    url: "/dashboard/documents/demo-doc-1",
    riskScore: 2,
  },
  {
    id: "demo-doc-2",
    title: "Vendor Master Services Agreement (MSA)",
    type: "Contract",
    status: "Audited",
    isGenerated: false,
    createdAt: new Date(Date.now() - 86400000 * 5).toISOString(),
    updatedAt: new Date(Date.now() - 86400000 * 5).toISOString(),
    riskScore: 3,
    flowId: "demo-freelance-flow",
    flowTitle: "Freelance Website Agreement",
    url: "/dashboard/documents/demo-doc-2",
  },
];

export function getDemoDocumentAndAnalysis(documentId: string): {
  document: Document;
  analysis: DetailedAnalysis;
} {
  const doc: Document = {
    id: documentId,
    user_id: DEMO_USER.id,
    filename: documentId === "demo-doc-2" ? "Vendor Master Services Agreement (MSA).pdf" : "Freelance Website Development Agreement.docx",
    size_bytes: 142850,
    mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    storage_path: "demo/freelance-agreement.docx",
    document_type: "freelance_service_agreement",
    status: "ready",
    created_at: new Date(Date.now() - 86400000 * 2).toISOString(),
    updated_at: new Date().toISOString(),
  };

  const analysis: DetailedAnalysis = {
    id: "demo-analysis-1",
    document_id: documentId,
    user_id: DEMO_USER.id,
    risk_score: 2,
    summary: "Medium Risk — The agreement contains unbalanced liability caps favoring the contractor and a strict 48-hour milestone acceptance window that warrants review.",
    model: "gemini-2.5-flash",
    created_at: new Date().toISOString(),
    clauses: [
      {
        id: "c-1",
        document_id: documentId,
        section: "Scope of Services",
        clause_number: "1",
        text: "The Contractor agrees to provide web design and full-stack development services as outlined in Schedule A.",
        page_number: 1,
        created_at: new Date().toISOString(),
      },
      {
        id: "c-2",
        document_id: documentId,
        section: "Limitation of Liability & Indemnification",
        clause_number: "4",
        text: "Contractor's aggregate liability shall not exceed 10% of total fees paid. Client agrees to indemnify Contractor against all third-party claims.",
        page_number: 1,
        created_at: new Date().toISOString(),
      },
      {
        id: "c-3",
        document_id: documentId,
        section: "Intellectual Property Assignment",
        clause_number: "3",
        text: "Upon receipt of full payment, all intellectual property rights and deliverables shall transfer to Client.",
        page_number: 1,
        created_at: new Date().toISOString(),
      },
    ],
    findings: [
      {
        id: "f-1",
        document_id: documentId,
        category: "Liability Limit & Indemnity",
        risk_level: "high",
        explanation: "Contractor liability is capped at 10% of total fees ($500 max) while client indemnity obligations remain uncapped.",
        why_it_matters: "Creates asymmetric financial exposure for the hiring client in case of developer breach or delay.",
        clause_id: "c-2",
        clause: {
          id: "c-2",
          document_id: documentId,
          section: "Limitation of Liability & Indemnification",
          clause_number: "4",
          text: "Contractor's aggregate liability shall not exceed 10% of total fees paid. Client agrees to indemnify Contractor against all third-party claims.",
          page_number: 1,
          created_at: new Date().toISOString(),
        },
        questions: ["Should the liability limit be equalized to 100% of the contract value?"],
        confidence: 0.96,
        created_at: new Date().toISOString(),
      },
      {
        id: "f-2",
        document_id: documentId,
        category: "Intellectual Property Transfer",
        risk_level: "medium",
        explanation: "Ownership of deliverables does not assign progressively; third-party open-source libraries must be explicitly excluded from exclusive transfer.",
        why_it_matters: "Client may not hold clear copyright over intermediate source code until final balance is settled.",
        clause_id: "c-3",
        clause: {
          id: "c-3",
          document_id: documentId,
          section: "Intellectual Property Assignment",
          clause_number: "3",
          text: "Upon receipt of full payment, all intellectual property rights and deliverables shall transfer to Client.",
          page_number: 1,
          created_at: new Date().toISOString(),
        },
        questions: ["Are third-party open-source components explicitly exempted from exclusive assignment?"],
        confidence: 0.92,
        created_at: new Date().toISOString(),
      },
    ],
    key_terms: [
      {
        id: "kt-1",
        document_id: documentId,
        term: "Total Contract Value",
        value: "$5,000 USD (50% upfront deposit, 50% on final launch)",
        source_clause_id: "c-1",
        clause: null,
        created_at: new Date().toISOString(),
      },
      {
        id: "kt-2",
        document_id: documentId,
        term: "Notice Period",
        value: "14 days written notice for mutual convenience termination",
        source_clause_id: "c-1",
        clause: null,
        created_at: new Date().toISOString(),
      },
    ],
    obligations: [
      {
        id: "obl-1",
        document_id: documentId,
        description: "50% Advance Retainer Payment ($2,500)",
        responsible_party: "Client (You)",
        deadline: "Upon contract signing",
        source_clause_id: "c-1",
        clause: null,
        created_at: new Date().toISOString(),
      },
      {
        id: "obl-2",
        document_id: documentId,
        description: "UI/UX Prototype Delivery & Functional Review",
        responsible_party: "Freelancer",
        deadline: "Within 14 business days",
        source_clause_id: "c-1",
        clause: null,
        created_at: new Date().toISOString(),
      },
      {
        id: "obl-3",
        document_id: documentId,
        description: "Final Code Handover & IP Assignment Clearance",
        responsible_party: "Both Parties",
        deadline: "Within 30 calendar days",
        source_clause_id: "c-3",
        clause: null,
        created_at: new Date().toISOString(),
      },
    ],
    result: {},
  };

  return { document: doc, analysis };
}
