"use client";

import { useRef, useState, useTransition } from "react";

import { Button } from "@/components/Button";
import { uploadDocument } from "@/lib/actions/documents";
import {
  formatBytes,
  validateDocumentFile,
} from "@/lib/documents/validation";

export function ContractUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isPending, startTransition] = useTransition();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File | null) => {
    setErrorMessage(null);
    setSuccessMessage(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const validation = validateDocumentFile({
      name: file.name,
      size: file.size,
      type: file.type,
    });

    if (!validation.valid) {
      setErrorMessage(validation.error);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleClearSelection = () => {
    setSelectedFile(null);
    setErrorMessage(null);
    setSuccessMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUpload = () => {
    if (!selectedFile) {
      setErrorMessage("Please select a file first.");
      return;
    }

    setErrorMessage(null);
    setSuccessMessage(null);

    startTransition(async () => {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const result = await uploadDocument(formData);

      if (result.error) {
        setErrorMessage(result.error);
      } else {
        setSuccessMessage(`"${selectedFile.name}" uploaded successfully.`);
        setSelectedFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    });
  };

  return (
    <div className="rounded-xl border border-border bg-surface p-5 transition-all">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">
            Upload New Agreement
          </h2>
          <p className="text-xs text-muted">
            Supported files: Digital PDF, Word (.docx), or Text (.txt) up to 50 MB.
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-muted border border-border px-2 py-0.5 rounded bg-surface-inset">
          <span>PDF</span>
          <span>•</span>
          <span>DOCX</span>
          <span>•</span>
          <span>TXT</span>
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        id="contract-file-input"
        accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
        className="sr-only"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleFileChange(e.target.files[0]);
          }
        }}
        disabled={isPending}
      />

      {/* Drop Zone */}
      {!selectedFile ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer items-center justify-between rounded-lg border border-dashed px-5 py-4 transition-colors ${
            isDragOver
              ? "border-accent bg-accent/10"
              : "border-border bg-surface-inset hover:border-border-strong hover:bg-surface-hover"
          } ${isPending ? "pointer-events-none opacity-50" : ""}`}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md bg-accent/10 border border-accent/20 text-accent">
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <div>
              <p className="text-xs font-semibold text-foreground">
                Drop your contract file here or <span className="text-accent underline">browse</span>
              </p>
              <p className="text-[11px] text-muted">
                Private & secure storage scoped to your account
              </p>
            </div>
          </div>

          <Button type="button" variant="secondary" className="px-3 py-1.5 text-xs pointer-events-none">
            Choose File
          </Button>
        </div>
      ) : (
        /* Selected File Card */
        <div className="rounded-lg border border-border bg-surface-inset p-3.5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded bg-accent/15 border border-accent/30 font-mono font-bold text-accent text-xs">
                {selectedFile.name.split(".").pop()?.toUpperCase() || "DOC"}
              </div>
              <div className="min-w-0">
                <p className="truncate text-xs font-medium text-foreground">
                  {selectedFile.name}
                </p>
                <p className="text-[11px] font-mono text-muted">
                  {formatBytes(selectedFile.size)}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleClearSelection}
                disabled={isPending}
                className="text-xs font-medium text-muted hover:text-red-400 disabled:opacity-50 px-2 py-1"
              >
                Cancel
              </button>
              <Button
                type="button"
                variant="primary"
                onClick={handleUpload}
                disabled={isPending}
                className="px-3.5 py-1.5 text-xs"
              >
                {isPending ? "Uploading..." : "Upload Contract"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {errorMessage ? (
        <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
          {errorMessage}
        </div>
      ) : null}

      {/* Success Message */}
      {successMessage ? (
        <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-400">
          {successMessage}
        </div>
      ) : null}
    </div>
  );
}
