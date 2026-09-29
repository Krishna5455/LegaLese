export type LegalFlowStage =
  | "intent"
  | "understand"
  | "prepare"
  | "review"
  | "trust_check"
  | "whats_next"
  | "track"
  | "complete";

export type LegalFlowStatus = "active" | "completed" | "archived";

export type FlowStepStatus =
  | "completed"
  | "next"
  | "recommended"
  | "optional"
  | "blocked";

export type TrustLevel =
  | "ai_assistance"
  | "professional_review_recommended"
  | "formal_process";

export type LegalFlowStep = {
  id: string;
  title: string;
  description: string;
  stage: LegalFlowStage;
  status: FlowStepStatus;
  trustLevel: TrustLevel;
  actionUrl?: string;
  actionLabel?: string;
  category?: string;
  badge?: string;
  formalGuidance?: string;
  metadata?: Record<string, unknown>;
};

export type TrustSummary = {
  level: TrustLevel;
  label: string;
  reason: string;
  formal_notes: string[];
};

export type LegalFlowRow = {
  id: string;
  user_id: string;
  title: string;
  intent_query: string;
  scenario_key: string;
  current_stage: LegalFlowStage;
  status: LegalFlowStatus;
  document_id?: string | null;
  generated_doc_id?: string | null;
  steps: LegalFlowStep[];
  trust_summary: TrustSummary;
  created_at: string;
  updated_at: string;
};

export type LegalFlowIntent = {
  rawQuery: string;
  scenarioKey: string;
  title: string;
  suggestedDocType?: string;
  confidence: number;
  recommendedSteps: LegalFlowStep[];
  trustRecommendation: TrustSummary;
};

export type VerifiedProfessional = {
  id: string;
  name: string;
  title: string;
  barRegistration: string;
  practiceAreas: string[];
  experienceYears: number;
  verificationStatus: "demo_verified" | "prototype";
  verificationDate: string;
  avatarUrl?: string;
  rating?: number;
  reviewTurnaroundHours: number;
  demoNotice: string;
};

export type LegalCalendarEvent = {
  id: string;
  flowId?: string;
  documentId?: string;
  title: string;
  description?: string;
  date: string;
  responsibleParty?: string;
  type: "payment" | "deadline" | "renewal" | "delivery" | "notice";
  status: "pending" | "completed" | "overdue";
  amount?: string;
};
