import "server-only";

import { GoogleGenerativeAI } from "@google/generative-ai";

import { getGeminiConfig } from "@/lib/ai/config";
import { buildSystemInstruction, buildUserMessage } from "@/lib/ai/prompt";
import { prepareContractText } from "@/lib/ai/prompt";
import { AIAnalysisOutputSchema } from "@/lib/ai/schema";
import type { AIAnalysisOutput } from "@/types/analysis";
import type { ProcessedDocument } from "@/types/processing";

export type GeminiAnalysisResult = {
  output: AIAnalysisOutput;
  wasTruncated: boolean;
  modelUsed: string;
};

export async function analyzeContractWithGemini(
  doc: ProcessedDocument,
): Promise<GeminiAnalysisResult> {
  const { apiKey, model: modelName } = getGeminiConfig();
  const { text: contractText, wasTruncated } = prepareContractText(doc);

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: buildSystemInstruction(),
    generationConfig: {
      responseMimeType: "application/json",
    },
  });

  const userMessage = buildUserMessage(doc, contractText);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  let rawResponseText: string;

  try {
    const result = await model.generateContent(
      {
        contents: [{ role: "user", parts: [{ text: userMessage }] }],
      },
      { signal: controller.signal } as Parameters<typeof model.generateContent>[1],
    );

    clearTimeout(timeoutId);
    rawResponseText = result.response.text();
  } catch (err) {
    clearTimeout(timeoutId);
    const isTimeout =
      err instanceof Error &&
      (err.name === "AbortError" || err.message.includes("aborted"));
    if (isTimeout) {
      throw new Error(
        "AI analysis timed out after 30 seconds. Please try again.",
      );
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Gemini API error: ${msg}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawResponseText);
  } catch {
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
      "AI returned a response that did not match the expected structure. Please try again.",
    );
  }

  return {
    output: validation.data as AIAnalysisOutput,
    wasTruncated,
    modelUsed: modelName,
  };
}
