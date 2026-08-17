"use client";

import type { DocumentTypeDefinition } from "@/types/generation";

type DocumentTypeSelectorProps = {
  types: DocumentTypeDefinition[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function DocumentTypeSelector({
  types,
  selectedId,
  onSelect,
}: DocumentTypeSelectorProps) {
  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Document type</h2>
      <ul className="space-y-2">
        {types.map((type) => {
          const isSelected = selectedId === type.id;
          return (
            <li key={type.id}>
              <button
                type="button"
                onClick={() => onSelect(type.id)}
                className={`w-full rounded-lg border px-4 py-3 text-left transition-colors ${
                  isSelected
                    ? "border-accent bg-accent/5"
                    : "border-border bg-surface hover:border-accent/40"
                }`}
              >
                <p className="text-sm font-semibold text-foreground">
                  {type.label}
                </p>
                <p className="mt-1 text-xs text-muted">{type.description}</p>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
