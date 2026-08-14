"use server";

import { revalidatePath } from "next/cache";

import {
  validateDocumentFile,
} from "@/lib/documents/validation";
import { createClient } from "@/lib/supabase/server";
import type { Document } from "@/types/database";

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

    // 1. Upload file to private Supabase Storage bucket 'contracts'
    const { error: storageError } = await supabase.storage
      .from("contracts")
      .upload(storagePath, fileBuffer, {
        contentType: mimeType,
        upsert: false,
      });

    if (storageError) {
      return {
        error: `Failed to upload file to storage: ${storageError.message}`,
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

      return {
        error: `Failed to record document in database: ${dbError.message}`,
      };
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
      return { error: "Document not found or you do not have permission to delete it." };
    }

    // 2. Remove storage object from 'contracts' bucket if storage_path exists
    if (doc.storage_path) {
      const { error: storageRemoveError } = await supabase.storage
        .from("contracts")
        .remove([doc.storage_path]);

      if (storageRemoveError) {
        console.warn(
          "Storage removal warning:",
          storageRemoveError.message,
        );
      }
    }

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
