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
  const roadmapTypes = [
    {
      id: "nda",
      label: "Non-Disclosure Agreement (NDA)",
      description: "Mutual or one-way confidentiality agreement to protect proprietary information.",
    },
    {
      id: "employment",
      label: "Employment Agreement",
      description: "Standard employment contract covering duties, compensation, and workplace policies.",
    },
    {
      id: "vendor",
      label: "Vendor / Supplier Contract",
      description: "Service level agreement for procurement and ongoing B2B vendor management.",
    },
  ];

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-bold text-foreground">Select Agreement Type</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        {types.map((type) => {
          const isSelected = selectedId === type.id;
          return (
            <button
              key={type.id}
              type="button"
              onClick={() => onSelect(type.id)}
              className={`rounded-xl border p-4 text-left transition-all ${
                isSelected
                  ? "border-accent bg-accent-soft ring-1 ring-accent/30 shadow-xs"
                  : "border-border bg-surface hover:border-slate-300"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-foreground">
                  {type.label}
                </span>
                <span className="rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 text-[10px] font-semibold">
                  Active
                </span>
              </div>
              <p className="mt-1 text-xs text-secondary leading-relaxed">
                {type.description}
              </p>
            </button>
          );
        })}

        {roadmapTypes.map((item) => (
          <div
            key={item.id}
            className="rounded-xl border border-border/60 bg-slate-50/50 p-4 text-left opacity-75 cursor-not-allowed select-none"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-secondary">
                {item.label}
              </span>
              <span className="rounded bg-accent-soft text-accent border border-accent/20 px-2 py-0.5 text-[10px] font-semibold">
                Coming Soon
              </span>
            </div>
            <p className="mt-1 text-xs text-muted leading-relaxed">
              {item.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

