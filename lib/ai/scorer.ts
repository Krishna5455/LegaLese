import type { AIFinding, OverallRisk, RiskLevel } from "@/types/analysis";

/**
 * Ordered from lowest to highest severity.
 * The overall risk is determined by the highest-severity finding.
 * The AI does NOT set the overall score — this is deterministic application logic.
 */
const RISK_ORDER: RiskLevel[] = ["info", "low", "medium", "high", "critical"];

/**
 * Computes the overall contract risk level from a list of AI-identified findings.
 *
 * Rules:
 * - No findings → "low"
 * - Only "info" findings → "low"
 * - Otherwise → the highest risk level present among findings
 */
export function computeOverallRisk(findings: AIFinding[]): OverallRisk {
  if (findings.length === 0) return "low";

  let maxIndex = 0;
  for (const finding of findings) {
    const idx = RISK_ORDER.indexOf(finding.riskLevel);
    if (idx > maxIndex) {
      maxIndex = idx;
    }
  }

  const maxRisk = RISK_ORDER[maxIndex];
  // "info" alone does not constitute a meaningful risk
  return maxRisk === "info" ? "low" : (maxRisk as OverallRisk);
}
