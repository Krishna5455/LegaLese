import { z } from "zod";

// ─── Enum schemas ─────────────────────────────────────────────────────────────

export const RiskLevelSchema = z.enum([
  "informational",
  "low",
  "medium",
  "high",
]);

// ─── AI Response Sub-schemas ──────────────────────────────────────────────────

export const AIClauseSchema = z.object({
  section: z.string().min(1).max(200),
  clauseNumber: z.string().max(50).nullable().optional(),
  text: z.string().min(1).max(5000),
  pageNumber: z.number().int().positive().nullable().optional(),
});

export const AIFindingSchema = z.object({
  category: z.string().min(1).max(100),
  riskLevel: RiskLevelSchema,
  explanation: z.string().min(1).max(2000),
  whyItMatters: z.string().max(1000).nullable().optional(),
  clauseIndex: z.number().int().min(0).nullable().optional(),
  questions: z.array(z.string().max(500)).max(10).optional().default([]),
  confidence: z.number().min(0).max(1).nullable().optional(),
});

export const AIKeyTermSchema = z.object({
  term: z.string().min(1).max(200),
  value: z.string().min(1).max(1000),
  clauseIndex: z.number().int().min(0).nullable().optional(),
});

export const AIObligationSchema = z.object({
  description: z.string().min(1).max(1000),
  responsibleParty: z.string().max(100).nullable().optional(),
  deadline: z.string().max(200).nullable().optional(),
  clauseIndex: z.number().int().min(0).nullable().optional(),
});

// ─── Root AI Analysis Output Schema ───────────────────────────────────────────

export const AIAnalysisOutputSchema = z.object({
  summary: z.string().min(1).max(3000),
  clauses: z.array(AIClauseSchema).max(30).default([]),
  findings: z.array(AIFindingSchema).max(30).default([]),
  keyTerms: z.array(AIKeyTermSchema).max(25).default([]),
  obligations: z.array(AIObligationSchema).max(25).default([]),
});

export type AIAnalysisOutputSchema = z.infer<typeof AIAnalysisOutputSchema>;
