import { z } from "zod";

export const ExplanationPartySchema = z.object({
  name: z.string().min(1).max(200),
  role: z.string().min(1).max(200),
});

export const ImportantClauseSchema = z.object({
  section_id: z.string().min(1).max(100),
  section_title: z.string().min(1).max(200),
  explanation: z.string().min(1).max(2000),
});

export const DocumentExplanationSchema = z.object({
  agreement_summary: z.string().min(1).max(3000),
  parties: z.array(ExplanationPartySchema).min(1).max(10),
  key_obligations: z.array(z.string().min(1).max(1000)).min(1).max(20),
  payment_terms: z.string().min(1).max(2000),
  duration_and_termination: z.string().min(1).max(2000),
  confidentiality: z.string().min(1).max(2000),
  intellectual_property: z.string().min(1).max(2000),
  important_clauses: z.array(ImportantClauseSchema).min(1).max(15),
  clarification_questions: z.array(z.string().min(1).max(1000)).min(1).max(15),
});

export type DocumentExplanation = z.infer<typeof DocumentExplanationSchema>;
export type ExplanationParty = z.infer<typeof ExplanationPartySchema>;
export type ImportantClause = z.infer<typeof ImportantClauseSchema>;
