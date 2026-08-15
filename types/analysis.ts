// ─── Risk / Confidence enumerations ──────────────────────────────────────────

export type RiskLevel = "info" | "low" | "medium" | "high" | "critical";
export type Confidence = "low" | "medium" | "high";
export type OverallRisk = "low" | "medium" | "high" | "critical";
export type AnalysisStatus = "pending" | "analyzing" | "complete" | "failed";

// ─── Database row types ───────────────────────────────────────────────────────

export type Analysis = {
  id: string;
  document_id: string;
  user_id: string;
  status: AnalysisStatus;
  error_message?: string | null;
  summary?: string | null;
  overall_risk?: OverallRisk | null;
  was_truncated: boolean;
  model_used?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  analyzed_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type Finding = {
  id: string;
  analysis_id: string;
  document_id: string;
  user_id: string;
  category: string;
  risk_level: RiskLevel;
  explanation: string;
  why_it_matters?: string | null;
  evidence_text?: string | null;
  source_section?: string | null;
  page_number?: number | null;
  section_index?: number | null;
  confidence?: Confidence | null;
  sort_order: number;
  created_at: string;
};

export type KeyTerm = {
  id: string;
  analysis_id: string;
  document_id: string;
  user_id: string;
  term: string;
  definition: string;
  source_section?: string | null;
  page_number?: number | null;
  section_index?: number | null;
  sort_order: number;
  created_at: string;
};

export type Obligation = {
  id: string;
  analysis_id: string;
  document_id: string;
  user_id: string;
  party?: string | null;
  description: string;
  source_section?: string | null;
  page_number?: number | null;
  section_index?: number | null;
  sort_order: number;
  created_at: string;
};

export type Question = {
  id: string;
  analysis_id: string;
  document_id: string;
  user_id: string;
  question_text: string;
  context?: string | null;
  sort_order: number;
  created_at: string;
};

// ─── Joined type for full analysis display ────────────────────────────────────

export type AnalysisWithDetails = Analysis & {
  findings: Finding[];
  key_terms: KeyTerm[];
  obligations: Obligation[];
  questions: Question[];
};

// ─── AI output types (before DB insertion) ───────────────────────────────────
// These represent the raw structured JSON returned by Gemini,
// validated by Zod before any database writes.

export type AIFinding = {
  category: string;
  riskLevel: RiskLevel;
  explanation: string;
  whyItMatters?: string | null;
  evidenceText?: string | null;
  sourceSection?: string | null;
  pageNumber?: number | null;
  sectionIndex?: number | null;
  confidence: Confidence;
};

export type AIKeyTerm = {
  term: string;
  definition: string;
  sourceSection?: string | null;
  pageNumber?: number | null;
  sectionIndex?: number | null;
};

export type AIObligation = {
  party?: string | null;
  description: string;
  sourceSection?: string | null;
  pageNumber?: number | null;
  sectionIndex?: number | null;
};

export type AIQuestion = {
  questionText: string;
  context?: string | null;
};

export type AIAnalysisOutput = {
  summary: string;
  findings: AIFinding[];
  keyTerms: AIKeyTerm[];
  obligations: AIObligation[];
  questions: AIQuestion[];
};
