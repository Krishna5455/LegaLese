export const MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

export const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"] as const;
export type AllowedExtension = (typeof ALLOWED_EXTENSIONS)[number];

export const EXTENSION_TO_MIME: Record<AllowedExtension, string> = {
  ".pdf": "application/pdf",
  ".docx":
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".txt": "text/plain",
};

export const EXTENSION_TO_TYPE: Record<AllowedExtension, string> = {
  ".pdf": "PDF",
  ".docx": "DOCX",
  ".txt": "TXT",
};

export function getFileExtension(filename: string): string {
  const lastDotIndex = filename.lastIndexOf(".");
  if (lastDotIndex === -1) return "";
  return filename.slice(lastDotIndex).toLowerCase();
}

export type ValidationResult =
  | {
      valid: true;
      extension: AllowedExtension;
      mimeType: string;
      documentType: string;
    }
  | {
      valid: false;
      error: string;
    };

export function validateDocumentFile(file: {
  name: string;
  size: number;
  type?: string;
}): ValidationResult {
  if (!file || !file.name) {
    return { valid: false, error: "Please select a valid file to upload." };
  }

  if (file.size <= 0) {
    return { valid: false, error: "The selected file is empty." };
  }

  if (file.size > MAX_DOCUMENT_SIZE_BYTES) {
    return {
      valid: false,
      error: `File size exceeds the 50 MB limit (${formatBytes(file.size)}).`,
    };
  }

  const ext = getFileExtension(file.name) as AllowedExtension;
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      error:
        "Unsupported file type. Only PDF (.pdf), Word (.docx), and plain text (.txt) files are supported.",
    };
  }

  const expectedMime = EXTENSION_TO_MIME[ext];
  const providedMime = file.type?.trim();
  // Use provided mime if compatible, otherwise fallback to standard mapped mime
  const finalMime =
    providedMime && providedMime !== "application/octet-stream"
      ? providedMime
      : expectedMime;

  return {
    valid: true,
    extension: ext,
    mimeType: finalMime,
    documentType: EXTENSION_TO_TYPE[ext],
  };
}

export function formatBytes(bytes?: number | null): string {
  if (bytes === null || bytes === undefined || isNaN(bytes)) {
    return "0 B";
  }
  if (bytes === 0) return "0 B";

  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}
