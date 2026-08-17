import { z } from "zod";

export const GeneratedSectionSchema = z.object({
  id: z.string().min(1).max(100),
  title: z.string().min(1).max(200),
  content: z.string().min(1).max(15000),
  order: z.number().int().min(0).max(100),
});

export const GeneratedDocumentPartiesSchema = z.object({
  freelancerName: z.string().min(1).max(200),
  clientName: z.string().min(1).max(200),
  clientAddress: z.string().max(500).nullable().optional(),
});

export const GeneratedDocumentContentSchema = z.object({
  title: z.string().min(1).max(300),
  documentType: z.literal("freelance_service_agreement"),
  parties: GeneratedDocumentPartiesSchema,
  sections: z.array(GeneratedSectionSchema).min(3).max(20),
  disclaimer: z.string().min(1).max(3000),
});

export type GeneratedDocumentContentValidated = z.infer<
  typeof GeneratedDocumentContentSchema
>;
