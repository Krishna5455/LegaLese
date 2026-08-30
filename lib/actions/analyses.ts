"use server";

import { revalidatePath } from "next/cache";

import { analyzeContractWithGemini } from "@/lib/ai/gemini";
import { computeRiskScore } from "@/lib/ai/scorer";
import { processDocumentBuffer } from "@/lib/documents/processor";
import { createClient } from "@/lib/supabase/server";
import type {
  AIClause,
  AIFinding,
  AIKeyTerm,
  AIObligation,
  AnalysisRow,
  ClauseRow,
  DetailedAnalysis,
  FindingRow,
  FindingWithClause,
  KeyTermRow,
  KeyTermWithClause,
  ObligationRow,
  ObligationWithClause,
} from "@/types/analysis";
import type { ProcessedDocument } from "@/types/processing";

export type AnalyzeActionResult = {
  success?: boolean;
  error?: string;
  analysis?: DetailedAnalysis;
};

export type GetAnalysisResult = {
  analysis?: DetailedAnalysis | null;
  error?: string;
};

/**
 * Triggers AI analysis for a document that has completed text extraction.
 *
 * Architecture alignment:
 * - Fits exact Supabase schema (analyses, clauses, findings, key_terms, obligations)
 * - Stores evidence via foreign keys to the clauses table
 * - Stores finding questions as JSONB array inside findings.questions
 * - Stores deterministic integer risk_score and full AI result JSON in analyses
 * - Enforces single active analysis per document
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

  // 1. Fetch document and verify ownership
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

  const docStatus = (doc.status || "").toLowerCase();
  if (docStatus !== "complete") {
    if (docStatus === "processing") {
      return { error: "Document is still processing text extraction. Please wait." };
    }
    if (docStatus === "failed") {
      return {
        error:
          "Document extraction failed. Please retry extraction before analyzing.",
      };
    }
    return {
      error:
        "Document must be fully processed before analysis can run.",
    };
  }

  // 2. Duplicate prevention: check if an analysis already exists for this document
  const { data: existingAnalyses } = await supabase
    .from("analyses")
    .select("id")
    .eq("document_id", documentId)
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(1);

  if (existingAnalyses && existingAnalyses.length > 0) {
    // Analysis already exists, fetch and return it
    const full = await fetchAnalysisWithDetails(supabase, documentId, user.id);
    if (full) {
      return { success: true, analysis: full };
    }
  }

  try {
    // 3. Download extracted artifact from Storage
    if (!doc.storage_path) {
      throw new Error("Document storage path is missing.");
    }

    let processedDoc: ProcessedDocument | null = null;
    const txtArtifactPath = `${doc.storage_path}.extracted.txt`;
    const jsonArtifactPath = `${doc.storage_path}.extracted.json`;

    // Attempt to load .extracted.txt first (standard storage MIME format)
    const { data: txtBlob } = await supabase.storage
      .from("contracts")
      .download(txtArtifactPath);

    if (txtBlob) {
      try {
        const text = await txtBlob.text();
        processedDoc = JSON.parse(text) as ProcessedDocument;
      } catch (err) {
        console.warn("[LegaLese/analyzeDocument] Could not parse .txt artifact:", err);
      }
    }

    // Fall back to legacy .extracted.json if .txt was not found
    if (!processedDoc) {
      const { data: jsonBlob } = await supabase.storage
        .from("contracts")
        .download(jsonArtifactPath);

      if (jsonBlob) {
        try {
          const text = await jsonBlob.text();
          processedDoc = JSON.parse(text) as ProcessedDocument;
        } catch (err) {
          console.warn("[LegaLese/analyzeDocument] Could not parse .json artifact:", err);
        }
      }
    }

    // Self-Healing Fallback: If neither artifact exists, download the original document and extract text on-the-fly
    if (!processedDoc) {
      console.log(
        `[LegaLese/analyzeDocument] Artifact not found for doc ${documentId}. Self-healing: extracting directly from original file...`,
      );

      const { data: originalBlob, error: origError } = await supabase.storage
        .from("contracts")
        .download(doc.storage_path);

      if (origError || !originalBlob) {
        console.error("[LegaLese/analyzeDocument] Original file missing from storage:", origError);
        throw new Error(
          "We could not locate the original document file in storage. Please re-upload your contract.",
        );
      }

      const fileBuffer = Buffer.from(await originalBlob.arrayBuffer());
      processedDoc = await processDocumentBuffer({
        documentId: doc.id,
        filename: doc.filename,
        documentType: doc.document_type,
        buffer: fileBuffer,
      });

      // Cache the extracted artifact as .extracted.txt for future fast retrieval
      try {
        const artifactBuffer = Buffer.from(JSON.stringify(processedDoc, null, 2), "utf-8");
        await supabase.storage
          .from("contracts")
          .upload(txtArtifactPath, artifactBuffer, {
            contentType: "text/plain",
            upsert: true,
          });
      } catch (cacheErr) {
        console.warn("[LegaLese/analyzeDocument] Could not cache self-healed artifact:", cacheErr);
      }
    }

    const wordCount = processedDoc.fullText
      ? processedDoc.fullText.trim().split(/\s+/).filter(Boolean).length
      : 0;

    if (wordCount < 10) {
      throw new Error(
        "Document text content is insufficient for AI analysis (fewer than 10 words). Please upload a readable contract.",
      );
    }


    // 4. Call Gemini AI API
    const geminiResult = await analyzeContractWithGemini(processedDoc);
    const { output, modelUsed } = geminiResult;

    // 5. Compute deterministic overall risk score (integer 0–4)
    const riskScore = computeRiskScore(output.findings);

    // 6. Insert Clauses into DB and capture their IDs
    const clauseIdMap: Record<number, string> = {};

    if (output.clauses.length > 0) {
      const clauseInsertRows = output.clauses.map((c: AIClause) => ({
        document_id: documentId,
        section: c.section,
        clause_number: c.clauseNumber ?? null,
        text: c.text,
        page_number: c.pageNumber ?? null,
      }));

      const { data: insertedClauses, error: clauseErr } = await supabase
        .from("clauses")
        .insert(clauseInsertRows)
        .select("id");

      if (clauseErr) {
        throw new Error(`Failed to save document clauses: ${clauseErr.message}`);
      }

      if (insertedClauses) {
        insertedClauses.forEach((row: { id: string }, idx: number) => {
          clauseIdMap[idx] = row.id;
        });
      }
    }

    // 7. Insert Findings with clause_id references & questions JSONB
    if (output.findings.length > 0) {
      const findingInsertRows = output.findings.map((f: AIFinding) => {
        const clauseId =
          f.clauseIndex != null ? clauseIdMap[f.clauseIndex] ?? null : null;
        return {
          document_id: documentId,
          clause_id: clauseId,
          risk_level: f.riskLevel,
          category: f.category,
          explanation: f.explanation,
          why_it_matters: f.whyItMatters ?? null,
          questions: f.questions ?? [],
          confidence: f.confidence ?? null,
        };
      });

      const { error: findErr } = await supabase
        .from("findings")
        .insert(findingInsertRows);

      if (findErr) {
        throw new Error(`Failed to save analysis findings: ${findErr.message}`);
      }
    }

    // 8. Insert Key Terms with source_clause_id
    if (output.keyTerms.length > 0) {
      const keyTermRows = output.keyTerms.map((kt: AIKeyTerm) => {
        const clauseId =
          kt.clauseIndex != null ? clauseIdMap[kt.clauseIndex] ?? null : null;
        return {
          document_id: documentId,
          term: kt.term,
          value: kt.value,
          source_clause_id: clauseId,
        };
      });

      const { error: ktErr } = await supabase
        .from("key_terms")
        .insert(keyTermRows);

      if (ktErr) {
        throw new Error(`Failed to save key terms: ${ktErr.message}`);
      }
    }

    // 9. Insert Obligations with source_clause_id
    if (output.obligations.length > 0) {
      const obligationRows = output.obligations.map((o: AIObligation) => {
        const clauseId =
          o.clauseIndex != null ? clauseIdMap[o.clauseIndex] ?? null : null;
        return {
          document_id: documentId,
          description: o.description,
          responsible_party: o.responsibleParty ?? null,
          deadline: o.deadline ?? null,
          source_clause_id: clauseId,
        };
      });

      const { error: oblErr } = await supabase
        .from("obligations")
        .insert(obligationRows);

      if (oblErr) {
        throw new Error(`Failed to save obligations: ${oblErr.message}`);
      }
    }

    // 10. Insert Analyses Record
    const { error: analysisErr } = await supabase.from("analyses").insert({
      document_id: documentId,
      user_id: user.id,
      risk_score: riskScore,
      summary: output.summary,
      result: output,
      model: modelUsed,
    });

    if (analysisErr) {
      throw new Error(`Failed to create analysis record: ${analysisErr.message}`);
    }

    // 11. Fetch detailed analysis for client view
    const fullAnalysis = await fetchAnalysisWithDetails(
      supabase,
      documentId,
      user.id,
    );

    revalidatePath("/dashboard");

    return {
      success: true,
      analysis: fullAnalysis ?? undefined,
    };
  } catch (err) {
    const rawMessage =
      err instanceof Error ? err.message : "An unexpected error occurred during analysis.";
    console.error("[LegaLese/analyzeDocument] Technical error:", err);

    const isOperational =
      rawMessage.includes("fewer than 10 words") ||
      rawMessage.includes("Scanned PDF") ||
      rawMessage.includes("Password-protected") ||
      rawMessage.includes("insufficient text") ||
      rawMessage.includes("re-upload your contract");

    const clientMessage = isOperational
      ? rawMessage
      : "We could not complete the contract analysis right now. Please try again in a moment.";

    return { error: clientMessage };
  }
}

/**
 * Retrieves the analysis and related child rows for a given document.
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

  const full = await fetchAnalysisWithDetails(supabase, documentId, user.id);
  return { analysis: full };
}

/**
 * Helper to query analyses and child rows (clauses, findings, key_terms, obligations).
 */
async function fetchAnalysisWithDetails(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  supabase: any,
  documentId: string,
  userId: string,
): Promise<DetailedAnalysis | null> {
  // Query analysis row
  const { data: analyses, error: analysisErr } = await supabase
    .from("analyses")
    .select("*")
    .eq("document_id", documentId)
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(1);

  if (analysisErr || !analyses || analyses.length === 0) {
    return null;
  }

  const analysisRow = analyses[0] as AnalysisRow;

  // Query child tables linked by document_id
  const [
    { data: clauses },
    { data: findings },
    { data: keyTerms },
    { data: obligations },
  ] = await Promise.all([
    supabase
      .from("clauses")
      .select("*")
      .eq("document_id", documentId)
      .order("created_at", { ascending: true }),
    supabase
      .from("findings")
      .select("*")
      .eq("document_id", documentId)
      .order("created_at", { ascending: true }),
    supabase
      .from("key_terms")
      .select("*")
      .eq("document_id", documentId)
      .order("created_at", { ascending: true }),
    supabase
      .from("obligations")
      .select("*")
      .eq("document_id", documentId)
      .order("created_at", { ascending: true }),
  ]);

  const clauseMap = new Map<string, ClauseRow>();
  (clauses as ClauseRow[] ?? []).forEach((c) => clauseMap.set(c.id, c));

  const findingsWithClause: FindingWithClause[] = (
    (findings as FindingRow[]) ?? []
  ).map((f) => ({
    ...f,
    clause: f.clause_id ? clauseMap.get(f.clause_id) ?? null : null,
  }));

  const keyTermsWithClause: KeyTermWithClause[] = (
    (keyTerms as KeyTermRow[]) ?? []
  ).map((kt) => ({
    ...kt,
    clause: kt.source_clause_id ? clauseMap.get(kt.source_clause_id) ?? null : null,
  }));

  const obligationsWithClause: ObligationWithClause[] = (
    (obligations as ObligationRow[]) ?? []
  ).map((o) => ({
    ...o,
    clause: o.source_clause_id ? clauseMap.get(o.source_clause_id) ?? null : null,
  }));

  return {
    ...analysisRow,
    clauses: (clauses as ClauseRow[]) ?? [],
    findings: findingsWithClause,
    key_terms: keyTermsWithClause,
    obligations: obligationsWithClause,
  };
}
