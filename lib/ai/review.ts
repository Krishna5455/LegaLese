import "server-only";

import { GoogleGenerativeAI } from "@google/generative-ai";

import { GEMINI_GENERATION_TIMEOUT_MS, getGeminiConfig } from "@/lib/ai/config";
import {
  buildReviewSystemInstruction,
  buildReviewUserMessage,
} from "@/lib/ai/review-prompt";
import {
  DocumentReviewSchema,
  type DocumentReview,
} from "@/lib/ai/review-schema";
import type { GeneratedDocumentContent } from "@/types/generation";

export type GeminiReviewResult = {
  output: DocumentReview;
  modelUsed: string;
};

export async function reviewGeneratedDocumentWithGemini(
  content: GeneratedDocumentContent,
): Promise<GeminiReviewResult> {
  const { apiKey, model: modelName } = getGeminiConfig();

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: buildReviewSystemInstruction(),
    generationConfig: {
      responseMimeType: "application/json",
    },
  });

  const userMessage = buildReviewUserMessage(content);
  const controller = new AbortController();
  const timeoutId = setTimeout(
    () => controller.abort(),
    GEMINI_GENERATION_TIMEOUT_MS,
  );

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
      throw new Error("Document contract review timed out. Please try again.");
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Contract review failed: ${msg}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawResponseText);
  } catch {
    console.error(
      "[LegaLese/Review] Gemini returned non-JSON response:",
      rawResponseText.slice(0, 500),
    );
    throw new Error("AI returned an invalid contract review response format.");
  }

  const validation = DocumentReviewSchema.safeParse(parsed);
  if (!validation.success) {
    console.error(
      "[LegaLese/Review] Gemini response failed schema validation:",
      JSON.stringify(validation.error.flatten(), null, 2),
    );
    throw new Error(
      "AI returned a review that did not match the expected structure. Please try again.",
    );
  }

  return {
    output: validation.data,
    modelUsed: modelName,
  };
}
