// ─── Risk / Confidence enumerations ──────────────────────────────────────────

export type RiskLevel = "informational" | "low" | "medium" | "high";

// ─── Database Row Types (matches exact remote Supabase schema) ───────────────

export type AnalysisRow = {
  id: string;
  document_id: string;
  user_id: string;
  risk_score: number | null;
  summary: string | null;
  result: Record<string, unknown>; // JSONB structured output
  model: string;
  created_at: string;
};

export type ClauseRow = {
  id: string;
  document_id: string;
  section: string;
  clause_number: string | null;
  text: string;
  page_number: number | null;
  created_at: string;
};

export type FindingRow = {
  id: string;
  document_id: string;
  clause_id: string | null;
  risk_level: RiskLevel;
  category: string;
  explanation: string;
  why_it_matters: string | null;
  questions: string[]; // Stored as JSONB in DB
  confidence: number | null;
  created_at: string;
};

export type KeyTermRow = {
  id: string;
  document_id: string;
  term: string;
  value: string;
  source_clause_id: string | null;
  created_at: string;
};

export type ObligationRow = {
  id: string;
  document_id: string;
  description: string;
  responsible_party: string | null;
  deadline: string | null;
  source_clause_id: string | null;
  created_at: string;
};

export type ReportRow = {
  id: string;
  document_id: string;
  user_id: string;
  file_path: string | null;
  created_at: string;
};

// ─── Joined Type for Full Dashboard Display ───────────────────────────────────

export type FindingWithClause = FindingRow & {
  clause?: ClauseRow | null;
};

export type KeyTermWithClause = KeyTermRow & {
  clause?: ClauseRow | null;
};

export type ObligationWithClause = ObligationRow & {
  clause?: ClauseRow | null;
};

export type DetailedAnalysis = AnalysisRow & {
  clauses: ClauseRow[];
  findings: FindingWithClause[];
  key_terms: KeyTermWithClause[];
  obligations: ObligationWithClause[];
};

// ─── AI Output Schema Types (for Gemini response parsing) ────────────────────

export type AIClause = {
  section: string;
  clauseNumber?: string | null;
  text: string;
  pageNumber?: number | null;
};

export type AIFinding = {
  category: string;
  riskLevel: RiskLevel;
  explanation: string;
  whyItMatters?: string | null;
  clauseIndex?: number | null; // 0-based index pointing to AIClause list
  questions?: string[] | null;
  confidence?: number | null; // e.g. 0.0 to 1.0
};

export type AIKeyTerm = {
  term: string;
  value: string; // Plain-English definition/value
  clauseIndex?: number | null;
};

export type AIObligation = {
  description: string;
  responsibleParty?: string | null;
  deadline?: string | null;
  clauseIndex?: number | null;
};

export type AIAnalysisOutput = {
  summary: string;
  clauses: AIClause[];
  findings: AIFinding[];
  keyTerms: AIKeyTerm[];
  obligations: AIObligation[];
};
