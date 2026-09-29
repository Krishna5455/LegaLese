import Link from "next/link";
import { FileText, ExternalLink, FilePlus } from "lucide-react";

interface ConnectedDocCardProps {
  document?: {
    id: string;
    title: string;
    type: string;
    status: string;
    isGenerated?: boolean;
    url: string;
  } | null;
  scenarioKey?: string;
}

export function ConnectedDocCard({ document, scenarioKey }: ConnectedDocCardProps) {
  if (!document) {
    const isCreateScenario =
      scenarioKey === "hire_freelancer" || scenarioKey === "create_nda";

    return (
      <div className="rounded-2xl border border-[#E7E5E2] bg-white p-5 sm:p-6 space-y-4 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#F7F7F5] border border-[#E7E5E2] text-[#8A8F98]">
              <FileText className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-bold text-[#171717]">Connected Document</h3>
          </div>
        </div>

        <div className="rounded-xl border border-dashed border-[#E7E5E2] bg-[#F7F7F5]/50 p-4 text-center space-y-2">
          <p className="text-xs text-[#5F6368]">No document connected yet.</p>
          <Link
            href={isCreateScenario ? "/dashboard/create" : "/dashboard#upload"}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#059669] hover:underline"
          >
            <FilePlus className="w-3.5 h-3.5" />
            <span>{isCreateScenario ? "Generate agreement" : "Upload document"}</span>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-[#E7E5E2] bg-white p-5 sm:p-6 space-y-4 shadow-xs">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/20 text-[#059669]">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98]">
              Connected Document
            </p>
            <h3 className="text-sm font-bold text-[#171717] truncate max-w-[200px]">
              {document.title}
            </h3>
          </div>
        </div>

        <span className="rounded-full bg-[#F7F7F5] border border-[#E7E5E2] px-2 py-0.5 text-[10px] font-mono font-medium text-[#5F6368] capitalize">
          {document.status}
        </span>
      </div>

      <div className="rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-3 text-xs space-y-1">
        <div className="flex items-center justify-between text-[#8A8F98] text-[11px]">
          <span>Type</span>
          <span className="font-medium text-[#171717] capitalize">
            {document.type.replace(/_/g, " ")}
          </span>
        </div>
        <div className="flex items-center justify-between text-[#8A8F98] text-[11px]">
          <span>Source</span>
          <span className="font-medium text-[#171717]">
            {document.isGenerated ? "AI Drafted Agreement" : "Uploaded Contract"}
          </span>
        </div>
      </div>

      <Link
        href={document.url}
        className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-[#171717] px-4 py-2.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
      >
        <span>Open Document Workspace</span>
        <ExternalLink className="w-3.5 h-3.5 text-[#059669]" />
      </Link>
    </div>
  );
}
