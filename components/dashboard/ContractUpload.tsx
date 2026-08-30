"use client";

import { useRef, useState, useTransition } from "react";

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
    <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 transition-all shadow-xs">
      <div className="mb-3.5 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">
            Upload Existing Contract
          </h2>
          <p className="text-xs text-secondary">
            Supported files: Digital PDF, Word (.docx), or Text (.txt) up to 50 MB.
          </p>
        </div>
        <div className="hidden sm:flex items-center gap-1.5 text-[10px] font-mono text-muted border border-border px-2 py-0.5 rounded bg-background">
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
          className={`flex cursor-pointer items-center justify-between rounded-xl border border-dashed px-6 py-5 transition-all ${
            isDragOver
              ? "border-[#059669] bg-[#059669]/5"
              : "border-[#E7E5E2] bg-white hover:border-[#D4D2CD] hover:bg-[#F7F7F5]"
          } ${isPending ? "pointer-events-none opacity-50" : ""}`}
        >
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-[#059669]/10 border border-[#059669]/20 text-[#059669]">
              <svg
                className="h-5 w-5"
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
              <p className="text-xs sm:text-sm font-semibold text-[#171717]">
                Drop your contract file here or <span className="text-[#059669] underline font-semibold">browse computer</span>
              </p>
              <p className="text-[11px] text-[#8A8F98]">
                Supports PDF and DOCX files up to 10MB • Private & secure
              </p>
            </div>
          </div>

          <span className="hidden sm:inline-flex rounded-lg border border-[#E7E5E2] bg-white px-3.5 py-1.5 text-xs font-medium text-[#171717] shadow-2xs hover:bg-[#F7F7F5] transition-colors">
            Choose File
          </span>
        </div>
      ) : (
        /* Selected File Card */
        <div className="rounded-xl border border-[#E7E5E2] bg-white p-4 shadow-2xs">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/20 font-mono font-bold text-[#059669] text-xs">
                {selectedFile.name.split(".").pop()?.toUpperCase() || "DOC"}
              </div>
              <div className="min-w-0">
                <p className="truncate text-xs font-semibold text-[#171717]">
                  {selectedFile.name}
                </p>
                <p className="text-[11px] font-mono text-[#8A8F98]">
                  {formatBytes(selectedFile.size)}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleClearSelection}
                disabled={isPending}
                className="text-xs font-medium text-[#5F6368] hover:text-[#B91C1C] disabled:opacity-50 px-2.5 py-1.5 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={isPending}
                className="rounded-lg bg-[#171717] px-4 py-2 text-xs font-medium text-white hover:bg-[#262626] transition-colors shadow-xs"
              >
                {isPending ? "Uploading..." : "Upload Contract"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {errorMessage ? (
        <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs font-medium text-rose-700">
          {errorMessage}
        </div>
      ) : null}

      {/* Success Message */}
      {successMessage ? (
        <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs font-medium text-emerald-700">
          {successMessage}
        </div>
      ) : null}
    </div>
  );
}

