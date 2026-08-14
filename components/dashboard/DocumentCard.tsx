"use client";

import { useState, useTransition } from "react";

import { deleteDocument } from "@/lib/actions/documents";
import { formatBytes } from "@/lib/documents/validation";
import type { Document } from "@/types/database";

function formatDate(value: string) {
  try {
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function getFileType(doc: Document): string {
  if (doc.document_type) return doc.document_type.toUpperCase();
  const ext = doc.filename?.split(".").pop()?.toUpperCase();
  return ext || "FILE";
}

function getStatusBadge(status?: string | null) {
  const s = (status || "uploaded").toLowerCase();
  if (s === "uploaded") {
    return (
      <span className="inline-flex items-center rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
        Uploaded
      </span>
    );
  }
  if (s === "processing") {
    return (
      <span className="inline-flex items-center rounded-full bg-yellow-50 px-2.5 py-0.5 text-xs font-medium text-yellow-700">
        Processing
      </span>
    );
  }
  if (s === "complete") {
    return (
      <span className="inline-flex items-center rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700">
        Complete
      </span>
    );
  }
  if (s === "failed") {
    return (
      <span className="inline-flex items-center rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-700">
      {status}
    </span>
  );
}

export function DocumentCard({ document }: { document: Document }) {
  const [isConfirming, setIsConfirming] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const handleDelete = () => {
    setDeleteError(null);
    startTransition(async () => {
      const result = await deleteDocument(document.id);
      if (result.error) {
        setDeleteError(result.error);
        setIsConfirming(false);
      }
    });
  };

  return (
    <li className="rounded-lg border border-border bg-surface p-4 transition-shadow hover:shadow-xs">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3 min-w-0">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md bg-accent/10 text-xs font-bold text-accent">
            {getFileType(document)}
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-foreground">
              {document.filename}
            </p>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
              <span>{formatBytes(document.size_bytes)}</span>
              <span>•</span>
              <span>Added {formatDate(document.created_at)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 sm:justify-end">
          <div>{getStatusBadge(document.status)}</div>

          {!isConfirming ? (
            <button
              type="button"
              onClick={() => setIsConfirming(true)}
              disabled={isPending}
              className="rounded p-1.5 text-xs font-medium text-muted hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
              title="Delete contract"
              aria-label={`Delete ${document.filename}`}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleDelete}
                disabled={isPending}
                className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                {isPending ? "Deleting..." : "Confirm"}
              </button>
              <button
                type="button"
                onClick={() => setIsConfirming(false)}
                disabled={isPending}
                className="rounded border border-border px-2 py-1 text-xs font-medium text-muted hover:text-foreground disabled:opacity-50"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {deleteError ? (
        <p className="mt-2 rounded bg-red-50 p-2 text-xs text-red-700">
          Delete failed: {deleteError}
        </p>
      ) : null}
    </li>
  );
}
