import type { VerifiedProfessional } from "@/types/legalflow";

/**
 * Mock Verified Legal Professionals for demonstration/prototype verification layer.
 * Strictly labeled as "Verification Prototype - Demonstration Only".
 */
export const DEMO_VERIFIED_PROFESSIONAL: VerifiedProfessional = {
  id: "prof-01",
  name: "Sarah Jenkins, Esq.",
  title: "Commercial Contracts & Tech IP Specialist",
  barRegistration: "NY-BAR-5928104 [Prototype Demo]",
  practiceAreas: ["Commercial Contracts", "Freelance Agreements", "IP Licensing", "Liability Caps"],
  experienceYears: 10,
  verificationStatus: "demo_verified",
  verificationDate: "Verified Jan 2026",
  avatarUrl: "/avatars/attorney-1.png",
  rating: 4.9,
  reviewTurnaroundHours: 24,
  demoNotice: "Verification Prototype — Demonstration Data Only",
};

export const DEMO_VERIFIED_PROFESSIONALS: VerifiedProfessional[] = [
  DEMO_VERIFIED_PROFESSIONAL,
];
