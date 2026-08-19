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
  try {
    const { id: documentId } = await params;

    if (!documentId) {
      return NextResponse.json(
        { error: "Document ID is required." },
        { status: 400 },
      );
    }

    const format = (
      request.nextUrl.searchParams.get("format") || "pdf"
    ).toLowerCase();

    if (!["pdf", "docx", "md"].includes(format)) {
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

    if (authError || !user) {
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
      return NextResponse.json(
        { error: "Document not found or access denied." },
        { status: 404 },
      );
    }

    const content = documentRow.generated_content as GeneratedDocumentContent;
    const createdAt = documentRow.created_at;

    if (format === "pdf") {
      const pdfBuffer = await exportPdf(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "pdf");

      return new Response(pdfBuffer as unknown as BodyInit, {
        status: 200,
        headers: {
          "Content-Type": "application/pdf",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Cache-Control": "no-store, max-age=0",
        },
      });
    }

    if (format === "docx") {
      const docxBuffer = await exportDocx(content, createdAt);
      const filename = generatedDocumentDownloadFilename(content.title, "docx");

      return new Response(docxBuffer as unknown as BodyInit, {
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
    const markdown = generatedDocumentToMarkdown(content, createdAt);
    const filename = generatedDocumentDownloadFilename(content.title, "md");

    return new Response(markdown, {
      status: 200,
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (err) {
    console.error("[LegaLese/ExportAPI] Export failed:", err);
    return NextResponse.json(
      { error: "An unexpected error occurred while generating export." },
      { status: 500 },
    );
  }
}
