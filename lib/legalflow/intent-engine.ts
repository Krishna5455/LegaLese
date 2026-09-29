import type {
  LegalFlowIntent,
  LegalFlowStep,
  TrustSummary,
} from "@/types/legalflow";

/**
 * Scenario configurations with customized workflow steps and trust evaluations.
 */
interface ScenarioTemplate {
  key: string;
  title: string;
  suggestedDocType?: string;
  defaultTrust: TrustSummary;
  generateSteps: (query?: string) => LegalFlowStep[];
}

const SCENARIOS: Record<string, ScenarioTemplate> = {
  hire_freelancer: {
    key: "hire_freelancer",
    title: "Hire Freelancer / Contractor Agreement",
    suggestedDocType: "freelance_service_agreement",
    defaultTrust: {
      level: "professional_review_recommended",
      label: "Professional Review Recommended for High-Liability Clauses",
      reason: "IP assignment, indemnity, and liability caps in freelance retainers significantly impact ownership of deliverables.",
      formal_notes: [
        "Signatures from both parties are required before work commences.",
        "Ensure clear jurisdiction and dispute resolution clauses match local commercial law.",
      ],
    },
    generateSteps: (): LegalFlowStep[] => [
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
        status: "next",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/create",
        actionLabel: "Generate Agreement",
        category: "Drafting",
        badge: "AI Generator",
      },
      {
        id: "step-review",
        title: "Review Clauses & Risk Traps",
        description: "Audit liability limits, termination notice, and late payment interest clauses for mutual fairness.",
        stage: "review",
        status: "recommended",
        trustLevel: "ai_assistance",
        actionLabel: "Audit Clauses",
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
        formalGuidance: "Recommended if project value exceeds standard thresholds or involves exclusive copyright buyout.",
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
        title: "Store in Document Vault",
        description: "Archive signed agreements and AI audit reports in your secure workspace vault.",
        stage: "complete",
        status: "optional",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/vault",
        actionLabel: "View in Vault",
        category: "Vault",
        badge: "Secure Vault",
      },
    ],
  },

  review_contract: {
    key: "review_contract",
    title: "Contract Pre-Sign Risk Audit",
    suggestedDocType: "general_contract",
    defaultTrust: {
      level: "professional_review_recommended",
      label: "Professional Review Recommended for Flagged Red Flags",
      reason: "High-risk indemnity, unilateral termination, or non-compete clauses often carry substantial exposure.",
      formal_notes: [
        "Pre-sign audit provides risk explanation, not formal legal warranty.",
        "Consult legal counsel if contract contains uncapped liability.",
      ],
    },
    generateSteps: (): LegalFlowStep[] => [
      {
        id: "step-upload",
        title: "Upload Contract Document",
        description: "Upload PDF or DOCX contract for plain-English parsing and clause breakdown.",
        stage: "understand",
        status: "completed",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard#upload",
        actionLabel: "Upload Contract",
        category: "Document Ingestion",
        badge: "Parser",
      },
      {
        id: "step-risk-analysis",
        title: "Analyze Risks & Traps",
        description: "Detect hidden risks, ambiguous termination clauses, and unfair indemnity clauses.",
        stage: "review",
        status: "next",
        trustLevel: "ai_assistance",
        actionLabel: "Audit Risk Score",
        category: "AI Audit",
        badge: "Risk Engine",
      },
      {
        id: "step-trust-check",
        title: "Trust Check: Review High-Risk Findings",
        description: "Determine whether identified medium/high risk findings require human attorney redlining.",
        stage: "trust_check",
        status: "recommended",
        trustLevel: "professional_review_recommended",
        actionLabel: "Verify With Professional",
        category: "Trust Layer",
        badge: "Expert Review",
      },
      {
        id: "step-obligations",
        title: "Extract Deadlines & Key Obligations",
        description: "Automatically identify payment due dates, notice windows, and renewal triggers.",
        stage: "track",
        status: "optional",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/calendar",
        actionLabel: "Track Dates",
        category: "Obligations",
        badge: "Date Extraction",
      },
      {
        id: "step-finalize",
        title: "Archive Audit & Action Plan",
        description: "Save analysis summary and audit export to Document Vault.",
        stage: "complete",
        status: "optional",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/vault",
        actionLabel: "Vault Archive",
        category: "Vault",
        badge: "Vault",
      },
    ],
  },

  create_nda: {
    key: "create_nda",
    title: "Non-Disclosure Agreement (NDA)",
    suggestedDocType: "non_disclosure_agreement",
    defaultTrust: {
      level: "ai_assistance",
      label: "Standard AI Drafting with Formal Execution Guidance",
      reason: "Standard bilateral NDAs follow recognized commercial norms unless trade secret sensitivity is critical.",
      formal_notes: [
        "Both disclosing and receiving parties must sign.",
        "Check definition of Confidential Information and term duration (typically 2-5 years).",
      ],
    },
    generateSteps: (): LegalFlowStep[] => [
      {
        id: "step-nda-understand",
        title: "Define Confidentiality Boundaries",
        description: "Specify unilateral vs mutual disclosure, permitted exclusions, and term duration.",
        stage: "understand",
        status: "completed",
        trustLevel: "ai_assistance",
        actionLabel: "Review Terms",
        category: "Preparation",
      },
      {
        id: "step-nda-prepare",
        title: "Draft NDA Document",
        description: "Generate balanced confidentiality clauses with standard carve-outs and return-of-materials provisions.",
        stage: "prepare",
        status: "next",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/create",
        actionLabel: "Draft NDA",
        category: "Drafting",
      },
      {
        id: "step-nda-review",
        title: "Audit Non-Compete & Restrictive Covenants",
        description: "Ensure the NDA does not inadvertently impose uncompensated non-compete restrictions.",
        stage: "review",
        status: "recommended",
        trustLevel: "ai_assistance",
        category: "Review",
      },
      {
        id: "step-nda-sign",
        title: "Bilateral Execution & Formal Signatures",
        description: "Send for electronic or countersigned execution before sharing sensitive information.",
        stage: "whats_next",
        status: "optional",
        trustLevel: "formal_process",
        formalGuidance: "Execute before disclosing proprietary source code, financial data, or trade secrets.",
        category: "Execution",
      },
      {
        id: "step-nda-track",
        title: "Track NDA Expiry & Return Deadlines",
        description: "Log confidentiality term and material return obligations in your legal calendar.",
        stage: "track",
        status: "optional",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/calendar",
        category: "Tracking",
      },
    ],
  },

  understand_document: {
    key: "understand_document",
    title: "Plain-English Legal Document Breakdown",
    suggestedDocType: "general_document",
    defaultTrust: {
      level: "ai_assistance",
      label: "Plain-English AI Explanations",
      reason: "Translates complex legalese into transparent, understandable summary points.",
      formal_notes: [
        "Summary explanations assist comprehension but do not alter the underlying binding contract text.",
      ],
    },
    generateSteps: (): LegalFlowStep[] => [
      {
        id: "step-und-upload",
        title: "Upload Legal Document",
        description: "Ingest agreement, terms of service, or notice to translate into plain English.",
        stage: "understand",
        status: "completed",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard#upload",
        category: "Input",
      },
      {
        id: "step-und-summary",
        title: "Generate Clause Breakdown & Summary",
        description: "Read plain-English explanations for every section and ask interactive clarifying questions.",
        stage: "understand",
        status: "next",
        trustLevel: "ai_assistance",
        actionLabel: "Explore Summary",
        category: "Explanation",
      },
      {
        id: "step-und-obligations",
        title: "Identify Your Rights & Responsibilities",
        description: "List explicit obligations, rights, and restrictions clearly.",
        stage: "track",
        status: "recommended",
        trustLevel: "ai_assistance",
        actionLabel: "View Rights & Duties",
        category: "Obligations",
      },
      {
        id: "step-und-save",
        title: "Save Summary to Workspace Vault",
        description: "Keep the decoded document and plain-English breakdown readily accessible.",
        stage: "complete",
        status: "optional",
        trustLevel: "ai_assistance",
        category: "Vault",
      },
    ],
  },

  track_obligations: {
    key: "track_obligations",
    title: "Obligation & Deadline Tracking Journey",
    suggestedDocType: "active_agreement",
    defaultTrust: {
      level: "ai_assistance",
      label: "Automated Calendar & Deadline Reminders",
      reason: "Extracts key milestones and dates directly from active agreements.",
      formal_notes: [
        "Notice periods often require written notice via specified delivery channels.",
      ],
    },
    generateSteps: (): LegalFlowStep[] => [
      {
        id: "step-track-extract",
        title: "Extract Contractual Deadlines",
        description: "Scan active documents for payment terms, delivery dates, renewal notices, and milestone deadlines.",
        stage: "understand",
        status: "completed",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/calendar",
        actionLabel: "Extract Deadlines",
        category: "Extraction",
      },
      {
        id: "step-track-calendar",
        title: "Sync with Legal Calendar",
        description: "Map upcoming contractual obligations on a clear chronological timeline.",
        stage: "track",
        status: "next",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/calendar",
        actionLabel: "Open Legal Calendar",
        category: "Calendar",
      },
      {
        id: "step-track-alerts",
        title: "Configure Milestone & Notice Triggers",
        description: "Set reminders for termination notice windows and payment approvals.",
        stage: "whats_next",
        status: "recommended",
        trustLevel: "ai_assistance",
        category: "Reminders",
      },
    ],
  },

  general_custom: {
    key: "general_custom",
    title: "Custom Legal Journey",
    suggestedDocType: "custom_journey",
    defaultTrust: {
      level: "ai_assistance",
      label: "AI Guidance & Structured Journey",
      reason: "LegaLese structures your custom legal task into a clear, manageable sequence of next actions.",
      formal_notes: [
        "Formal legal processes (such as notarization or corporate filing) may require authorized authorities.",
      ],
    },
    generateSteps: (query = ""): LegalFlowStep[] => [
      {
        id: "step-custom-understand",
        title: `Understand: ${query.slice(0, 40)}${query.length > 40 ? "..." : ""}`,
        description: "Clarify core requirements, stakeholders, and governing jurisdiction.",
        stage: "understand",
        status: "completed",
        trustLevel: "ai_assistance",
        category: "Intake",
      },
      {
        id: "step-custom-prepare",
        title: "Prepare or Ingest Relevant Agreement",
        description: "Create a custom agreement or upload an existing document for audit.",
        stage: "prepare",
        status: "next",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/create",
        actionLabel: "Prepare Document",
        category: "Preparation",
      },
      {
        id: "step-custom-review",
        title: "AI Review & Risk Identification",
        description: "Identify high-risk terms, potential pitfalls, and ambiguous clauses.",
        stage: "review",
        status: "recommended",
        trustLevel: "ai_assistance",
        category: "Review",
      },
      {
        id: "step-custom-trust",
        title: "Trust Check: Review Necessity of Professional Judgment",
        description: "Determine if formal attorney verification or notarization is required.",
        stage: "trust_check",
        status: "recommended",
        trustLevel: "professional_review_recommended",
        category: "Trust Check",
      },
      {
        id: "step-custom-action",
        title: "Action Next Steps & Obligations",
        description: "Execute signatures and track key dates in your workspace calendar.",
        stage: "track",
        status: "optional",
        trustLevel: "ai_assistance",
        actionUrl: "/dashboard/calendar",
        category: "Tracking",
      },
    ],
  },
};

/**
 * Classifies a user's natural language intent into a structured LegalFlowIntent.
 */
export function classifyUserIntent(rawQuery: string): LegalFlowIntent {
  const clean = rawQuery.trim().toLowerCase();

  // 1. Keyword-based deterministic classification
  let scenarioKey = "general_custom";

  if (
    clean.includes("freelance") ||
    clean.includes("hire") ||
    clean.includes("contractor") ||
    clean.includes("website") ||
    clean.includes("developer") ||
    clean.includes("designer") ||
    clean.includes("consultant") ||
    clean.includes("service agreement")
  ) {
    scenarioKey = "hire_freelancer";
  } else if (
    clean.includes("review") ||
    clean.includes("audit") ||
    clean.includes("check contract") ||
    clean.includes("analyze") ||
    clean.includes("risk") ||
    clean.includes("upload")
  ) {
    scenarioKey = "review_contract";
  } else if (
    clean.includes("nda") ||
    clean.includes("confidential") ||
    clean.includes("non-disclosure") ||
    clean.includes("secret")
  ) {
    scenarioKey = "create_nda";
  } else if (
    clean.includes("understand") ||
    clean.includes("explain") ||
    clean.includes("plain english") ||
    clean.includes("decode") ||
    clean.includes("what does this mean")
  ) {
    scenarioKey = "understand_document";
  } else if (
    clean.includes("track") ||
    clean.includes("obligation") ||
    clean.includes("deadline") ||
    clean.includes("calendar") ||
    clean.includes("payment due") ||
    clean.includes("renewal")
  ) {
    scenarioKey = "track_obligations";
  }

  const template = SCENARIOS[scenarioKey] || SCENARIOS.general_custom;
  const steps = template.generateSteps(rawQuery);

  // Custom title generation if user provided a descriptive prompt
  let title = template.title;
  if (scenarioKey === "hire_freelancer" && clean.includes("for")) {
    const afterFor = rawQuery.slice(clean.indexOf("for") + 4).trim();
    if (afterFor.length > 2) {
      title = `Freelance Agreement: ${afterFor.charAt(0).toUpperCase() + afterFor.slice(1)}`;
    }
  } else if (scenarioKey === "general_custom" && rawQuery.trim().length > 0) {
    title = rawQuery.trim().slice(0, 60);
  }

  return {
    rawQuery,
    scenarioKey,
    title,
    suggestedDocType: template.suggestedDocType,
    confidence: scenarioKey === "general_custom" ? 0.75 : 0.95,
    recommendedSteps: steps,
    trustRecommendation: template.defaultTrust,
  };
}

/**
 * Calculates current progress percentage and identifies the primary "What Do I Do Next?" step.
 */
export function getFlowNextAction(steps: LegalFlowStep[]): {
  completedCount: number;
  totalCount: number;
  progressPercent: number;
  nextStep: LegalFlowStep | null;
  recommendedSteps: LegalFlowStep[];
} {
  const totalCount = steps.length;
  const completedCount = steps.filter((s) => s.status === "completed").length;
  const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  // Find first step marked as "next"
  let nextStep = steps.find((s) => s.status === "next") || null;

  // If no step is explicitly "next", find first non-completed step
  if (!nextStep) {
    nextStep = steps.find((s) => s.status !== "completed") || null;
  }

  const recommendedSteps = steps.filter((s) => s.status === "recommended");

  return {
    completedCount,
    totalCount,
    progressPercent,
    nextStep,
    recommendedSteps,
  };
}
