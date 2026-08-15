"use server";

import { revalidatePath } from "next/cache";

import { analyzeContractWithGemini } from "@/lib/ai/gemini";
import { computeOverallRisk } from "@/lib/ai/scorer";
import { createClient } from "@/lib/supabase/server";
import type {
  AIFinding,
  AIKeyTerm,
  AIObligation,
  AIQuestion,
  Analysis,
  AnalysisWithDetails,
  Finding,
  KeyTerm,
  Obligation,
  Question,
} from "@/types/analysis";
import type { ProcessedDocument } from "@/types/processing";

// ─── Result types ─────────────────────────────────────────────────────────────

export type AnalyzeActionResult = {
  success?: boolean;
  error?: string;
  analysis?: AnalysisWithDetails;
};

export type GetAnalysisResult = {
  analysis?: AnalysisWithDetails | null;
  error?: string;
};

// ─── analyzeDocument ──────────────────────────────────────────────────────────

/**
 * Triggers AI analysis for a document that has completed Phase 4 processing.
 *
 * Security:
 * - Requires authenticated user session
 * - Verifies document ownership before accessing any data
 * - Never exposes GEMINI_API_KEY to the client
 *
 * Duplicate prevention:
 * - If a complete or in-progress analysis exists, returns it without calling the AI
 */
export async function analyzeDocument(
  documentId: string,
): Promise<AnalyzeActionResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to analyze documents." };
  }

  // ── 1. Fetch document and verify ownership ──────────────────────────────────
  const { data: doc, error: docError } = await supabase
    .from("documents")
    .select("*")
    .eq("id", documentId)
    .eq("user_id", user.id)
    .single();

  if (docError || !doc) {
    return {
      error: "Document not found or you do not have permission to analyze it.",
    };
  }

  // ── 2. Guard: document must be fully processed ──────────────────────────────
  const docStatus = (doc.status || "").toLowerCase();
  if (docStatus !== "complete") {
    if (docStatus === "processing") {
      return { error: "Document is still being processed. Please wait." };
    }
    if (docStatus === "failed") {
      return {
        error:
          "Document processing failed. Please re-process the document before analyzing.",
      };
    }
    return {
      error:
        "Document must be fully processed before it can be analyzed. Please process it first.",
    };
  }

  // ── 3. Duplicate prevention: check for existing analysis ───────────────────
  const { data: existingAnalyses } = await supabase
    .from("analyses")
    .select("id, status")
    .eq("document_id", documentId)
    .eq("user_id", user.id)
    .in("status", ["analyzing", "complete"])
    .order("created_at", { ascending: false })
    .limit(1);

  if (existingAnalyses && existingAnalyses.length > 0) {
    const existing = existingAnalyses[0];
    if (existing.status === "analyzing") {
      return { error: "Analysis is already in progress for this document." };
    }
    if (existing.status === "complete") {
      // Return the existing complete analysis
      const full = await fetchAnalysisWithDetails(supabase, existing.id, user.id);
      return { success: true, analysis: full ?? undefined };
    }
  }

  // ── 4. Create analyses row in 'analyzing' state ─────────────────────────────
  const { data: analysisRow, error: insertError } = await supabase
    .from("analyses")
    .insert({
      document_id: documentId,
      user_id: user.id,
      status: "analyzing",
    })
    .select("*")
    .single();

  if (insertError || !analysisRow) {
    return { error: `Failed to create analysis record: ${insertError?.message}` };
  }

  const analysisId: string = analysisRow.id;

  try {
    // ── 5. Download extracted JSON artifact from Storage ──────────────────────
    if (!doc.storage_path) {
      throw new Error("Document storage path is missing.");
    }

    const artifactPath = `${doc.storage_path}.extracted.json`;
    const { data: artifactBlob, error: downloadError } = await supabase.storage
      .from("contracts")
      .download(artifactPath);

    if (downloadError || !artifactBlob) {
      throw new Error(
        "Extracted document artifact not found. Please re-process the document before analyzing.",
      );
    }

    // ── 6. Parse ProcessedDocument ────────────────────────────────────────────
    const artifactText = await artifactBlob.text();
    let processedDoc: ProcessedDocument;
    try {
      processedDoc = JSON.parse(artifactText) as ProcessedDocument;
    } catch {
      throw new Error("Extracted document artifact is corrupted or unreadable.");
    }

    if (!processedDoc.fullText || !processedDoc.fullText.trim()) {
      throw new Error(
        "No text content is available in this document for analysis.",
      );
    }

    // ── 7. Call Gemini API ────────────────────────────────────────────────────
    const geminiResult = await analyzeContractWithGemini(processedDoc);
    const { output, wasTruncated, modelUsed, inputTokens, outputTokens } =
      geminiResult;

    // ── 8. Compute overall risk deterministically ─────────────────────────────
    const overallRisk = computeOverallRisk(output.findings);

    // ── 9. Insert child rows ──────────────────────────────────────────────────
    // findings
    if (output.findings.length > 0) {
      const findingRows = output.findings.map(
        (f: AIFinding, idx: number) => ({
          analysis_id: analysisId,
          document_id: documentId,
          user_id: user.id,
          category: f.category,
          risk_level: f.riskLevel,
          explanation: f.explanation,
          why_it_matters: f.whyItMatters ?? null,
          evidence_text: f.evidenceText ?? null,
          source_section: f.sourceSection ?? null,
          page_number: f.pageNumber ?? null,
          section_index: f.sectionIndex ?? null,
          confidence: f.confidence,
          sort_order: idx,
        }),
      );
      const { error: findErr } = await supabase
        .from("findings")
        .insert(findingRows);
      if (findErr) {
        throw new Error(`Failed to save findings: ${findErr.message}`);
      }
    }

    // key_terms
    if (output.keyTerms.length > 0) {
      const keyTermRows = output.keyTerms.map(
        (kt: AIKeyTerm, idx: number) => ({
          analysis_id: analysisId,
          document_id: documentId,
          user_id: user.id,
          term: kt.term,
          definition: kt.definition,
          source_section: kt.sourceSection ?? null,
          page_number: kt.pageNumber ?? null,
          section_index: kt.sectionIndex ?? null,
          sort_order: idx,
        }),
      );
      const { error: ktErr } = await supabase
        .from("key_terms")
        .insert(keyTermRows);
      if (ktErr) {
        throw new Error(`Failed to save key terms: ${ktErr.message}`);
      }
    }

    // obligations
    if (output.obligations.length > 0) {
      const obligationRows = output.obligations.map(
        (o: AIObligation, idx: number) => ({
          analysis_id: analysisId,
          document_id: documentId,
          user_id: user.id,
          party: o.party ?? null,
          description: o.description,
          source_section: o.sourceSection ?? null,
          page_number: o.pageNumber ?? null,
          section_index: o.sectionIndex ?? null,
          sort_order: idx,
        }),
      );
      const { error: oblErr } = await supabase
        .from("obligations")
        .insert(obligationRows);
      if (oblErr) {
        throw new Error(`Failed to save obligations: ${oblErr.message}`);
      }
    }

    // questions
    if (output.questions.length > 0) {
      const questionRows = output.questions.map(
        (q: AIQuestion, idx: number) => ({
          analysis_id: analysisId,
          document_id: documentId,
          user_id: user.id,
          question_text: q.questionText,
          context: q.context ?? null,
          sort_order: idx,
        }),
      );
      const { error: qErr } = await supabase
        .from("questions")
        .insert(questionRows);
      if (qErr) {
        throw new Error(`Failed to save questions: ${qErr.message}`);
      }
    }

    // ── 10. Mark analysis complete ────────────────────────────────────────────
    const { error: completeErr } = await supabase
      .from("analyses")
      .update({
        status: "complete",
        summary: output.summary,
        overall_risk: overallRisk,
        was_truncated: wasTruncated,
        model_used: modelUsed,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        analyzed_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .eq("id", analysisId)
      .eq("user_id", user.id);

    if (completeErr) {
      throw new Error(
        `Failed to finalize analysis record: ${completeErr.message}`,
      );
    }

    // ── 11. Fetch the full analysis for the return value ──────────────────────
    const fullAnalysis = await fetchAnalysisWithDetails(
      supabase,
      analysisId,
      user.id,
    );

    revalidatePath("/dashboard");

    return { success: true, analysis: fullAnalysis ?? undefined };
  } catch (err) {
    // On any failure: mark analysis as failed
    const errorMessage =
      err instanceof Error ? err.message : "An unexpected error occurred.";

    console.error("[LegaLese/analyzeDocument] Error:", errorMessage);

    await supabase
      .from("analyses")
      .update({
        status: "failed",
        error_message: errorMessage,
        updated_at: new Date().toISOString(),
      })
      .eq("id", analysisId)
      .eq("user_id", user.id);

    revalidatePath("/dashboard");

    return { error: errorMessage };
  }
}

// ─── getAnalysis ──────────────────────────────────────────────────────────────

/**
 * Retrieves the most recent complete analysis for a document.
 * Returns null if no complete analysis exists.
 */
export async function getAnalysis(
  documentId: string,
): Promise<GetAnalysisResult> {
  if (!documentId) {
    return { error: "Document ID is required." };
  }

  const supabase = await createClient();
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return { error: "You must be signed in to view analyses." };
  }

  const { data: analyses } = await supabase
    .from("analyses")
    .select("id")
    .eq("document_id", documentId)
    .eq("user_id", user.id)
    .eq("status", "complete")
    .order("created_at", { ascending: false })
    .limit(1);

  if (!analyses || analyses.length === 0) {
    return { analysis: null };
  }

  const full = await fetchAnalysisWithDetails(supabase, analyses[0].id, user.id);
  return { analysis: full };
}

// ─── Internal helper ──────────────────────────────────────────────────────────

/**
 * Fetches a complete analysis with all child rows (findings, key_terms,
 * obligations, questions). Ownership is enforced via user_id in all queries.
 */
async function fetchAnalysisWithDetails(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  supabase: any,
  analysisId: string,
  userId: string,
): Promise<AnalysisWithDetails | null> {
  const { data: analysis, error } = await supabase
    .from("analyses")
    .select("*")
    .eq("id", analysisId)
    .eq("user_id", userId)
    .single();

  if (error || !analysis) return null;

  const [
    { data: findings },
    { data: keyTerms },
    { data: obligations },
    { data: questions },
  ] = await Promise.all([
    supabase
      .from("findings")
      .select("*")
      .eq("analysis_id", analysisId)
      .eq("user_id", userId)
      .order("sort_order", { ascending: true }),
    supabase
      .from("key_terms")
      .select("*")
      .eq("analysis_id", analysisId)
      .eq("user_id", userId)
      .order("sort_order", { ascending: true }),
    supabase
      .from("obligations")
      .select("*")
      .eq("analysis_id", analysisId)
      .eq("user_id", userId)
      .order("sort_order", { ascending: true }),
    supabase
      .from("questions")
      .select("*")
      .eq("analysis_id", analysisId)
      .eq("user_id", userId)
      .order("sort_order", { ascending: true }),
  ]);

  return {
    ...(analysis as Analysis),
    findings: (findings as Finding[]) ?? [],
    key_terms: (keyTerms as KeyTerm[]) ?? [],
    obligations: (obligations as Obligation[]) ?? [],
    questions: (questions as Question[]) ?? [],
  };
}
