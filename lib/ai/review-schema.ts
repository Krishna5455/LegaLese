import { z } from "zod";

export const ReviewStatusSchema = z.enum(["clear", "attention", "potential_concern"]);

export const ReviewFindingSchema = z.object({
  section_id: z.string().min(1).max(100),
  section_title: z.string().min(1).max(200),
  category: z.string().min(1).max(100),
  status: ReviewStatusSchema,
  clause_excerpt: z.string().min(1).max(500),
  why_it_matters: z.string().min(1).max(1500),
  what_to_clarify: z.string().min(1).max(1500),
});

export const DocumentReviewSchema = z.object({
  overall_summary: z.string().min(1).max(3000),
  findings: z.array(ReviewFindingSchema).min(1).max(25),
});

export type ReviewStatus = z.infer<typeof ReviewStatusSchema>;
export type ReviewFinding = z.infer<typeof ReviewFindingSchema>;
export type DocumentReview = z.infer<typeof DocumentReviewSchema>;
