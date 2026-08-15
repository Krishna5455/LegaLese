import type { AIFinding, RiskLevel } from "@/types/analysis";

const RISK_SCORE_MAP: Record<RiskLevel, number> = {
  informational: 0,
  low: 1,
  medium: 2,
  high: 3,
};

/**
 * Computes a deterministic overall risk score (integer 0–3) from AI findings.
 *
 * Rules:
 * - No findings → 0 (Informational)
 * - Returns the highest numeric score present among findings (informational=0, low=1, medium=2, high=3).
 * - The AI does NOT compute this score — it is strictly deterministic application logic.
 */
export function computeRiskScore(findings: AIFinding[]): number {
  if (!findings || findings.length === 0) return 0;

  let maxScore = 0;
  for (const finding of findings) {
    const score = RISK_SCORE_MAP[finding.riskLevel] ?? 0;
    if (score > maxScore) {
      maxScore = score;
    }
  }

  return maxScore;
}

/**
 * Maps a numeric risk score (0–3) to a human-readable overall risk label for UI display.
 */
export function getRiskLabel(score: number | null | undefined): {
  label: string;
  level: RiskLevel;
} {
  if (score == null || score <= 0) {
    return { label: "Informational", level: "informational" };
  }
  if (score === 1) {
    return { label: "Low Risk", level: "low" };
  }
  if (score === 2) {
    return { label: "Medium Risk", level: "medium" };
  }
  return { label: "High Risk", level: "high" };
}

