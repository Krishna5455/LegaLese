import "server-only";

import { GoogleGenerativeAI } from "@google/generative-ai";

import { buildSystemInstruction, buildUserMessage } from "@/lib/ai/prompt";
import { AIAnalysisOutputSchema } from "@/lib/ai/schema";
import type { AIAnalysisOutput } from "@/types/analysis";
import type { ProcessedDocument } from "@/types/processing";
import { prepareContractText } from "./prompt";

// ─── Environment config ───────────────────────────────────────────────────────

function getGeminiConfig(): { apiKey: string; model: string } {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY environment variable is not set. " +
        "Add it to your .env.local file (server-only — never use NEXT_PUBLIC_).",
    );
  }
  const model = process.env.GEMINI_MODEL ?? "gemini-2.0-flash";
  return { apiKey, model };
}

// ─── Token counting helper ────────────────────────────────────────────────────

export type GeminiAnalysisResult = {
  output: AIAnalysisOutput;
  wasTruncated: boolean;
  modelUsed: string;
  inputTokens: number | null;
  outputTokens: number | null;
};

// ─── Main analysis function ───────────────────────────────────────────────────

/**
 * Sends the extracted contract text to Gemini for structured AI analysis.
 *
 * - Server-only: this function must never be called from client components.
 * - Uses JSON output mode to get a structured response.
 * - Validates the response with Zod before returning.
 * - Applies a 25-second timeout via AbortSignal.
 *
 * @throws Error if the API key is missing, the API call fails, the response
 *   times out, or the response fails schema validation.
 */
export async function analyzeContractWithGemini(
  doc: ProcessedDocument,
): Promise<GeminiAnalysisResult> {
  const { apiKey, model: modelName } = getGeminiConfig();

  // Prepare (and potentially truncate) the document text
  const { text: contractText, wasTruncated } = prepareContractText(doc);

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: buildSystemInstruction(),
    generationConfig: {
      responseMimeType: "application/json",
      temperature: 0.1,    // Low temperature for consistent, factual output
      maxOutputTokens: 8192,
    },
  });

  const userMessage = buildUserMessage(doc, contractText);

  // Apply a 25-second timeout to prevent server action hangs
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 25_000);

  let rawResponseText: string;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;

  try {
    const result = await model.generateContent(
      {
        contents: [{ role: "user", parts: [{ text: userMessage }] }],
      },
      { signal: controller.signal } as Parameters<typeof model.generateContent>[1],
    );

    clearTimeout(timeoutId);

    const response = result.response;

    // Extract token usage metadata if available
    const usageMeta = response.usageMetadata;
    if (usageMeta) {
      inputTokens = usageMeta.promptTokenCount ?? null;
      outputTokens = usageMeta.candidatesTokenCount ?? null;
    }

    rawResponseText = response.text();
  } catch (err) {
    clearTimeout(timeoutId);
    const isTimeout =
      err instanceof Error &&
      (err.name === "AbortError" || err.message.includes("aborted"));
    if (isTimeout) {
      throw new Error(
        "AI analysis timed out after 25 seconds. Please try again.",
      );
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Gemini API error: ${msg}`);
  }

  // ─── Parse and validate response ───────────────────────────────────────────
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawResponseText);
  } catch {
    // Log the raw response server-side to aid debugging (not sent to client)
    console.error(
      "[LegaLese/AI] Gemini returned non-JSON response:",
      rawResponseText.slice(0, 500),
    );
    throw new Error(
      "AI returned an unexpected response format (not valid JSON).",
    );
  }

  const validation = AIAnalysisOutputSchema.safeParse(parsed);
  if (!validation.success) {
    console.error(
      "[LegaLese/AI] Gemini response failed schema validation:",
      JSON.stringify(validation.error.flatten(), null, 2),
    );
    throw new Error(
      "AI returned a response that did not match the expected structure. " +
        "Please try again.",
    );
  }

  return {
    output: validation.data as AIAnalysisOutput,
    wasTruncated,
    modelUsed: modelName,
    inputTokens,
    outputTokens,
  };
}
