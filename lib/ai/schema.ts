import { z } from "zod";

// ─── Enum schemas ─────────────────────────────────────────────────────────────

export const RiskLevelSchema = z.enum(["info", "low", "medium", "high", "critical"]);
export const ConfidenceSchema = z.enum(["low", "medium", "high"]);

// ─── Child schemas ────────────────────────────────────────────────────────────

export const AIFindingSchema = z.object({
  category: z.string().min(1).max(100),
  riskLevel: RiskLevelSchema,
  explanation: z.string().min(1).max(2000),
  whyItMatters: z.string().max(1000).nullable().optional(),
  evidenceText: z.string().max(2000).nullable().optional(),
  sourceSection: z.string().max(200).nullable().optional(),
  // pageNumber is 1-indexed; null for DOCX/TXT documents
  pageNumber: z.number().int().positive().nullable().optional(),
  sectionIndex: z.number().int().min(0).nullable().optional(),
  confidence: ConfidenceSchema,
});

export const AIKeyTermSchema = z.object({
  term: z.string().min(1).max(200),
  definition: z.string().min(1).max(1000),
  sourceSection: z.string().max(200).nullable().optional(),
  pageNumber: z.number().int().positive().nullable().optional(),
  sectionIndex: z.number().int().min(0).nullable().optional(),
});

export const AIObligationSchema = z.object({
  party: z.string().max(100).nullable().optional(),
  description: z.string().min(1).max(1000),
  sourceSection: z.string().max(200).nullable().optional(),
  pageNumber: z.number().int().positive().nullable().optional(),
  sectionIndex: z.number().int().min(0).nullable().optional(),
});

export const AIQuestionSchema = z.object({
  questionText: z.string().min(1).max(500),
  context: z.string().max(500).nullable().optional(),
});

// ─── Root analysis output schema ──────────────────────────────────────────────

export const AIAnalysisOutputSchema = z.object({
  summary: z.string().min(1).max(3000),
  findings: z.array(AIFindingSchema).max(30),
  keyTerms: z.array(AIKeyTermSchema).max(25),
  obligations: z.array(AIObligationSchema).max(25),
  questions: z.array(AIQuestionSchema).max(10),
});

export type AIAnalysisOutputSchema = z.infer<typeof AIAnalysisOutputSchema>;
