import "server-only";

import { GoogleGenerativeAI } from "@google/generative-ai";

import { GEMINI_GENERATION_TIMEOUT_MS, getGeminiConfig } from "@/lib/ai/config";
import {
  buildFreelanceAgreementUserMessage,
  buildGenerationSystemInstruction,
} from "@/lib/ai/generation-prompt";
import { GeneratedDocumentContentSchema } from "@/lib/ai/generation-schema";
import type { ValidatedFreelanceAgreementInput } from "@/lib/generation/freelance-agreement-schema";
import type { GeneratedDocumentContent } from "@/types/generation";

export type GeminiGenerationResult = {
  output: GeneratedDocumentContent;
  modelUsed: string;
};

export async function generateFreelanceAgreementWithGemini(
  input: ValidatedFreelanceAgreementInput,
): Promise<GeminiGenerationResult> {
  const { apiKey, model: modelName } = getGeminiConfig();

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: buildGenerationSystemInstruction(),
    generationConfig: {
      responseMimeType: "application/json",
    },
  });

  const userMessage = buildFreelanceAgreementUserMessage(input);
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
      throw new Error(
        "Document generation timed out. Please try again.",
      );
    }
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Document generation failed: ${msg}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawResponseText);
  } catch {
    console.error(
      "[LegaLese/Generation] Gemini returned non-JSON response:",
      rawResponseText.slice(0, 500),
    );
    throw new Error(
      "AI returned an unexpected response format. Please try again.",
    );
  }

  const validation = GeneratedDocumentContentSchema.safeParse(parsed);
  if (!validation.success) {
    console.error(
      "[LegaLese/Generation] Gemini response failed schema validation:",
      JSON.stringify(validation.error.flatten(), null, 2),
    );
    throw new Error(
      "AI returned a response that did not match the expected document structure. Please try again.",
    );
  }

  const sortedSections = [...validation.data.sections].sort(
    (a, b) => a.order - b.order,
  );

  return {
    output: {
      ...validation.data,
      sections: sortedSections,
    },
    modelUsed: modelName,
  };
}
