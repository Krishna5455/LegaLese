import "server-only";

import { GoogleGenerativeAI } from "@google/generative-ai";

import { GEMINI_GENERATION_TIMEOUT_MS, getGeminiConfig } from "@/lib/ai/config";
import {
  buildExplanationSystemInstruction,
  buildExplanationUserMessage,
} from "@/lib/ai/explanation-prompt";
import {
  DocumentExplanationSchema,
  type DocumentExplanation,
} from "@/lib/ai/explanation-schema";
import type { GeneratedDocumentContent } from "@/types/generation";

export type GeminiExplanationResult = {
  output: DocumentExplanation;
  modelUsed: string;
};

export async function explainGeneratedDocumentWithGemini(
  content: GeneratedDocumentContent,
): Promise<GeminiExplanationResult> {
  const { apiKey, model: modelName } = getGeminiConfig();

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: buildExplanationSystemInstruction(),
    generationConfig: {
      responseMimeType: "application/json",
    },
  });

  const userMessage = buildExplanationUserMessage(content);
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
      throw new Error("Document analysis for explanation timed out. Please try again.");
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Document explanation failed: ${msg}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawResponseText);
  } catch {
    console.error(
      "[LegaLese/Explanation] Gemini returned non-JSON response:",
      rawResponseText.slice(0, 500),
    );
    throw new Error("AI returned an invalid explanation response format.");
  }

  const validation = DocumentExplanationSchema.safeParse(parsed);
  if (!validation.success) {
    console.error(
      "[LegaLese/Explanation] Gemini response failed schema validation:",
      JSON.stringify(validation.error.flatten(), null, 2),
    );
    throw new Error(
      "AI returned an explanation that did not match the expected structure. Please try again.",
    );
  }

  return {
    output: validation.data,
    modelUsed: modelName,
  };
}
