"use client";

import { useState } from "react";

import { DocumentTypeSelector } from "@/components/create/DocumentTypeSelector";
import { FreelanceAgreementForm } from "@/components/create/FreelanceAgreementForm";
import type { DocumentTypeDefinition } from "@/types/generation";

type CreateDocumentSectionProps = {
  types: DocumentTypeDefinition[];
};

export function CreateDocumentSection({ types }: CreateDocumentSectionProps) {
  const [selectedId, setSelectedId] = useState<string>("freelance_service_agreement");

  return (
    <div className="rounded-xl border border-border bg-surface p-6 space-y-6">
      <DocumentTypeSelector
        types={types}
        selectedId={selectedId}
        onSelect={(id) => setSelectedId(id)}
      />

      <div className="border-t border-border pt-6">
        {selectedId === "freelance_service_agreement" ? (
          <FreelanceAgreementForm />
        ) : (
          <p className="text-xs text-muted">
            Please select a supported document type above.
          </p>
        )}
      </div>
    </div>
  );
}
