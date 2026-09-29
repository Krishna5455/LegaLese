import "server-only";

export function getGeminiConfig(): { apiKey: string; model: string } {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY environment variable is not set. " +
        "Add it to your .env.local file (server-only — never use NEXT_PUBLIC_).",
    );
  }
  const model = process.env.GEMINI_MODEL ?? "gemini-3.5-flash-lite";
  return { apiKey, model };
}

export const GEMINI_GENERATION_TIMEOUT_MS = 120_000;
export const GEMINI_THINKING_BUDGET = 512;
