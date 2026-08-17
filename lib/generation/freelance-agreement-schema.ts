import { z } from "zod";

export const FreelanceAgreementInputSchema = z.object({
  freelancerName: z
    .string()
    .trim()
    .min(1, "Freelancer name is required.")
    .max(200),
  clientName: z
    .string()
    .trim()
    .min(1, "Client or company name is required.")
    .max(200),
  clientAddress: z.string().trim().max(500).optional(),
  servicesDescription: z
    .string()
    .trim()
    .min(1, "Please describe the services you will provide.")
    .max(5000),
  deliverables: z
    .string()
    .trim()
    .min(1, "Please list the project deliverables.")
    .max(5000),
  startDate: z.string().trim().min(1, "Start date is required."),
  completionDate: z
    .string()
    .trim()
    .min(1, "Expected completion date is required."),
  projectFee: z
    .string()
    .trim()
    .min(1, "Project fee is required.")
    .max(100),
  paymentStructure: z
    .string()
    .trim()
    .min(1, "Payment structure is required.")
    .max(500),
  paymentSchedule: z
    .string()
    .trim()
    .min(1, "Payment due date or schedule is required.")
    .max(500),
  currency: z.string().trim().min(1, "Currency is required.").max(20),
  noticePeriod: z
    .string()
    .trim()
    .min(1, "Notice period is required.")
    .max(200),
  earlyTerminationWork: z
    .string()
    .trim()
    .min(1, "Please describe what happens to completed work if the agreement ends early.")
    .max(2000),
  ipOwnership: z
    .string()
    .trim()
    .min(1, "Please specify who owns the final work after payment.")
    .max(2000),
  freelancerReusableMaterials: z
    .string()
    .trim()
    .min(1, "Please specify whether the freelancer can keep rights to reusable materials.")
    .max(2000),
  confidentialityRequired: z.enum(["yes", "no"], {
    message: "Please indicate whether confidentiality is required.",
  }),
  jurisdiction: z
    .string()
    .trim()
    .min(1, "Please specify which state/city/country should handle legal disputes.")
    .max(300),
});

export type ValidatedFreelanceAgreementInput = z.infer<
  typeof FreelanceAgreementInputSchema
>;

export function parseFreelanceAgreementInput(
  input: unknown,
):
  | { success: true; data: ValidatedFreelanceAgreementInput }
  | { success: false; error: string } {
  const result = FreelanceAgreementInputSchema.safeParse(input);
  if (!result.success) {
    const firstIssue = result.error.issues[0];
    return {
      success: false,
      error: firstIssue?.message ?? "Invalid form input.",
    };
  }
  return { success: true, data: result.data };
}
