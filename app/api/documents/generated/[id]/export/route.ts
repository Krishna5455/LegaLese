import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { exportPdf } from "@/lib/generation/export-pdf";
import { exportDocx } from "@/lib/generation/export-docx";
import {
  generatedDocumentDownloadFilename,
  generatedDocumentToMarkdown,
} from "@/lib/generation/export";
import type { GeneratedDocumentContent } from "@/types/generation";

export const dynamic = "force-dynamic";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  let format = "pdf";
  let documentId = "";

  try {
    const resolvedParams = await params;
    documentId = resolvedParams?.id ?? "";
    format = (
      request.nextUrl.searchParams.get("format") || "pdf"
    ).toLowerCase();

    console.log(
      `[LegaLese/ExportAPI] Incoming export request | docId: ${documentId} | format: ${format}`,
    );

    if (!documentId) {
      console.warn("[LegaLese/ExportAPI] Missing document ID");
      return NextResponse.json(
        { error: "Document ID is required." },
        { status: 400 },
      );
    }

    if (!["pdf", "docx", "md"].includes(format)) {
      console.warn(`[LegaLese/ExportAPI] Invalid format requested: ${format}`);
      return NextResponse.json(
        { error: "Invalid export format requested. Supported formats: pdf, docx, md." },
        { status: 400 },
      );
    }

    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    console.log(
      `[LegaLese/ExportAPI] Auth check | userPresent: ${Boolean(user)} | authError: ${
        authError?.message ?? "none"
      }`,
    );

    if (authError || !user) {
      console.warn("[LegaLese/ExportAPI] Unauthorized export attempt");
      return NextResponse.json(
        { error: "Unauthorized. Please sign in to export documents." },
        { status: 401 },
      );
    }

    const { data: documentRow, error: dbError } = await supabase
      .from("generated_documents")
      .select("*")
      .eq("id", documentId)
      .eq("user_id", user.id)
      .single();

    if (dbError || !documentRow) {
      console.warn(
        `[LegaLese/ExportAPI] Document lookup failed | dbError: ${
          dbError?.message ?? "Not found"
        }`,
      );
      return NextResponse.json(
        { error: "Document not found or access denied." },
        { status: 404 },
      );
    }

    const content = documentRow.generated_content as GeneratedDocumentContent;
    const createdAt = documentRow.created_at;

    if (!content || !content.title || !Array.isArray(content.sections)) {
      console.error("[LegaLese/ExportAPI] Invalid document payload structure");
      return NextResponse.json(
        { error: "Invalid generated document structure." },
        { status: 422 },
      );
    }

    if (format === "pdf") {
      console.log("[LegaLese/ExportAPI] Starting PDF generation...");
      const pdfBuffer = await exportPdf(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "pdf");
      const uint8Array = new Uint8Array(pdfBuffer);

      console.log(
        `[LegaLese/ExportAPI] PDF generation success | bufferSize: ${uint8Array.byteLength} bytes | status: 200`,
      );

      return new Response(uint8Array, {
        status: 200,
        headers: {
          "Content-Type": "application/pdf",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Cache-Control": "no-store, max-age=0",
        },
      });
    }

    if (format === "docx") {
      console.log("[LegaLese/ExportAPI] Starting DOCX generation...");
      const docxBuffer = await exportDocx(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "docx");
      const uint8Array = new Uint8Array(docxBuffer);

      console.log(
        `[LegaLese/ExportAPI] DOCX generation success | bufferSize: ${uint8Array.byteLength} bytes | status: 200`,
      );

      return new Response(uint8Array, {
        status: 200,
        headers: {
          "Content-Type":
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Cache-Control": "no-store, max-age=0",
        },
      });
    }

    // Markdown format
    console.log("[LegaLese/ExportAPI] Starting Markdown export...");
    const markdown = generatedDocumentToMarkdown(content, createdAt);
    const filename = generatedDocumentDownloadFilename(content.title, "md");

    console.log(
      `[LegaLese/ExportAPI] Markdown export success | length: ${markdown.length} chars | status: 200`,
    );

    return new Response(markdown, {
      status: 200,
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (err) {
    const errorMessage =
      err instanceof Error ? err.message : "Unknown server error";
    console.error(
      `[LegaLese/ExportAPI] Export exception caught | docId: ${documentId} | format: ${format} | error: ${errorMessage}`,
    );
    return NextResponse.json(
      { error: `Export failed: ${errorMessage}` },
      { status: 500 },
    );
  }
}
