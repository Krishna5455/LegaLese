"use server";

import { revalidatePath } from "next/cache";

import { processDocumentBuffer } from "@/lib/documents/processor";
import { validateDocumentFile } from "@/lib/documents/validation";
import { createClient } from "@/lib/supabase/server";
import type { Document } from "@/types/database";
import type { ProcessedDocument } from "@/types/processing";

export type UploadActionResult = {
  success?: boolean;
  error?: string;
  document?: Document;
};

export async function uploadDocument(
  formData: FormData,
): Promise<UploadActionResult> {
  try {
    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError || !user) {
      return {
        error: "You must be signed in to upload contracts. Please sign in again.",
      };
    }

    const file = formData.get("file") as File | null;
    if (!file || typeof file === "string") {
      return { error: "No file was selected for upload." };
    }

    const validation = validateDocumentFile({
      name: file.name,
      size: file.size,
      type: file.type,
    });

    if (!validation.valid) {
      return { error: validation.error };
    }

    const { extension, mimeType, documentType } = validation;
    const uniqueFileId = crypto.randomUUID();
    const storagePath = `${user.id}/${uniqueFileId}${extension}`;

    const arrayBuffer = await file.arrayBuffer();
    const fileBuffer = Buffer.from(arrayBuffer);

    // 1. Upload raw file to private Supabase Storage bucket 'contracts'
    const { error: storageError } = await supabase.storage
      .from("contracts")
      .upload(storagePath, fileBuffer, {
        contentType: mimeType,
        upsert: false,
      });

    if (storageError) {
      console.error("[LegaLese/uploadDocument] Storage upload error:", storageError);
      return {
        error: "We could not upload your file to storage. Please check your network connection and try again.",
      };
    }

    // 2. Insert metadata row into 'documents' table
    const { data: documentRow, error: dbError } = await supabase
      .from("documents")
      .insert({
        user_id: user.id,
        filename: file.name,
        document_type: documentType,
        mime_type: mimeType,
        size_bytes: file.size,
        storage_path: storagePath,
        status: "uploaded",
      })
      .select("*")
      .single();

    if (dbError) {
      // Clean up orphaned storage object if database insert fails
      try {
        await supabase.storage.from("contracts").remove([storagePath]);
      } catch (cleanupError) {
        console.error(
          "Failed to remove orphaned storage object after database error:",
          cleanupError,
        );
      }

      console.error("[LegaLese/uploadDocument] Database insert error:", dbError);
      return {
        error: "We could not record your document metadata. Please try again.",
      };
    }

    // 3. Immediately trigger document text extraction
    try {
      await processDocumentInternal(supabase, user.id, documentRow as Document, fileBuffer);
    } catch (procErr) {
      console.warn("Initial processing warning:", procErr);
      // Even if initial processing fails, document record exists as 'failed' and can be retried
    }

    revalidatePath("/dashboard");

    return {
      success: true,
      document: documentRow as Document,
    };
  } catch (err) {
    console.error("Unexpected upload error:", err);
    return {
      error:
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while processing your upload.",
    };
  }
}

export type ProcessActionResult = {
  success?: boolean;
  error?: string;
  processed?: ProcessedDocument;
};

export async function processDocument(
  documentId: string,
): Promise<ProcessActionResult> {
  try {
    if (!documentId) {
      return { error: "Document ID is required." };
    }

    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError || !user) {
      return { error: "You must be signed in to process documents." };
    }

    // 1. Fetch document and confirm ownership
    const { data: doc, error: fetchError } = await supabase
      .from("documents")
      .select("*")
      .eq("id", documentId)
      .eq("user_id", user.id)
      .single();

    if (fetchError || !doc) {
      return {
        error:
          "Document not found or you do not have permission to process it.",
      };
    }

    // 2. Download file from storage
    const { data: fileBlob, error: downloadError } = await supabase.storage
      .from("contracts")
      .download(doc.storage_path);

    if (downloadError || !fileBlob) {
      await supabase
        .from("documents")
        .update({
          status: "failed",
          updated_at: new Date().toISOString(),
        })
        .eq("id", doc.id)
        .eq("user_id", user.id);

      revalidatePath("/dashboard");
      return {
        error: `Could not retrieve document from storage: ${downloadError?.message || "File missing"}`,
      };
    }

    const arrayBuffer = await fileBlob.arrayBuffer();
    const fileBuffer = Buffer.from(arrayBuffer);

    // 3. Process document
    const processedDoc = await processDocumentInternal(
      supabase,
      user.id,
      doc as Document,
      fileBuffer,
    );

    revalidatePath("/dashboard");

    return {
      success: true,
      processed: processedDoc,
    };
  } catch (err) {
    console.error("Unexpected process error:", err);
    return {
      error:
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while processing the document.",
    };
  }
}

/**
 * Internal helper to run extraction and save JSON artifact to storage
 */
async function processDocumentInternal(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  supabase: any,
  userId: string,
  doc: Document,
  fileBuffer: Buffer,
): Promise<ProcessedDocument> {
  // 1. Transition status to 'processing'
  await supabase
    .from("documents")
    .update({
      status: "processing",
      updated_at: new Date().toISOString(),
    })
    .eq("id", doc.id)
    .eq("user_id", userId);

  try {
    // 2. Extract and clean text
    const processedDoc = await processDocumentBuffer({
      documentId: doc.id,
      filename: doc.filename,
      documentType: doc.document_type,
      buffer: fileBuffer,
    });

    // 3. Upload extracted JSON text artifact to private 'contracts' storage bucket
    // Note: The bucket permits text/plain, so we save the extracted text artifact with .txt extension
    const artifactPath = `${doc.storage_path}.extracted.txt`;
    const artifactBuffer = Buffer.from(JSON.stringify(processedDoc, null, 2), "utf-8");

    const { error: artifactUploadErr } = await supabase.storage
      .from("contracts")
      .upload(artifactPath, artifactBuffer, {
        contentType: "text/plain",
        upsert: true,
      });

    if (artifactUploadErr) {
      console.warn(
        `[LegaLese/processDocumentInternal] Artifact storage notice for ${doc.id}: ${artifactUploadErr.message}`,
      );
    }

    // 4. Update status to 'complete'
    await supabase
      .from("documents")
      .update({
        status: "complete",
        updated_at: new Date().toISOString(),
      })
      .eq("id", doc.id)
      .eq("user_id", userId);

    return processedDoc;
  } catch (error) {
    // Transition status to 'failed'
    await supabase
      .from("documents")
      .update({
        status: "failed",
        updated_at: new Date().toISOString(),
      })
      .eq("id", doc.id)
      .eq("user_id", userId);

    throw error;
  }
}

export type DeleteActionResult = {
  success?: boolean;
  error?: string;
};

export async function deleteDocument(
  documentId: string,
): Promise<DeleteActionResult> {
  try {
    if (!documentId) {
      return { error: "Document ID is required." };
    }

    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError || !user) {
      return { error: "You must be signed in to delete documents." };
    }

    // 1. Fetch document to obtain storage_path and confirm ownership
    const { data: doc, error: fetchError } = await supabase
      .from("documents")
      .select("id, user_id, storage_path")
      .eq("id", documentId)
      .eq("user_id", user.id)
      .single();

    if (fetchError || !doc) {
      return {
        error:
          "Document not found or you do not have permission to delete it.",
      };
    }

    // 2. Remove storage objects (raw file, extracted JSON, and report MD) from 'contracts' bucket
    if (doc.storage_path) {
      const filesToRemove = [
        doc.storage_path,
        `${doc.storage_path}.extracted.json`,
        `${user.id}/reports/${documentId}_report.md`,
      ];

      const { error: storageRemoveError } = await supabase.storage
        .from("contracts")
        .remove(filesToRemove);

      if (storageRemoveError) {
        console.warn("Storage removal warning:", storageRemoveError.message);
      }
    }

    // 2b. Clean up associated reports DB records
    await supabase.from("reports").delete().eq("document_id", documentId).eq("user_id", user.id);

    // 3. Delete database record

    const { error: dbDeleteError } = await supabase
      .from("documents")
      .delete()
      .eq("id", documentId)
      .eq("user_id", user.id);

    if (dbDeleteError) {
      return {
        error: `Failed to delete document record: ${dbDeleteError.message}`,
      };
    }

    revalidatePath("/dashboard");

    return { success: true };
  } catch (err) {
    console.error("Unexpected delete error:", err);
    return {
      error:
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while deleting the document.",
    };
  }
}
