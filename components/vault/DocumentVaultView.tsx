"use client";

import { useState } from "react";
import Link from "next/link";
import {
  FolderLock,
  Search,
  FileText,
  FilePlus,
  ExternalLink,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

export type VaultDocument = {
  id: string;
  title: string;
  type: string;
  status: string;
  isGenerated: boolean;
  createdAt: string;
  updatedAt?: string;
  riskScore?: number | null;
  flowId?: string;
  flowTitle?: string;
  url: string;
};

interface DocumentVaultViewProps {
  documents: VaultDocument[];
}

export function DocumentVaultView({ documents }: DocumentVaultViewProps) {
  const [filter, setFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredDocs = documents.filter((doc) => {
    let matchesFilter = true;
    if (filter === "generated") matchesFilter = doc.isGenerated;
    else if (filter === "uploaded") matchesFilter = !doc.isGenerated;
    else if (filter === "attention") matchesFilter = (doc.riskScore ?? 0) >= 2;

    const matchesSearch =
      searchQuery.trim() === "" ||
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (doc.flowTitle && doc.flowTitle.toLowerCase().includes(searchQuery.toLowerCase()));

    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#E7E5E2] pb-6">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-1.5 rounded-full bg-[#059669]/10 px-2.5 py-0.5 text-[11px] font-semibold text-[#059669]">
            <FolderLock className="w-3 h-3" />
            <span>Document Vault</span>
          </div>
          <h1 className="heading-page text-[#171717]">Document Vault</h1>
          <p className="text-xs sm:text-[14px] text-[#5F6368]">
            Unified secure repository of all AI-generated agreements, analyzed contracts, and connected LegalFlows.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link
            href="/dashboard/create"
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
          >
            <FilePlus className="w-3.5 h-3.5 text-[#059669]" />
            <span>Create New</span>
          </Link>
          <span className="rounded-md bg-[#F7F7F5] border border-[#E7E5E2] px-2.5 py-1 text-xs font-semibold text-[#5F6368]">
            {documents.length} Files
          </span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Category Filters */}
        <div className="flex flex-wrap items-center gap-1.5">
          {[
            { id: "all", label: "All Documents" },
            { id: "generated", label: "AI Generated" },
            { id: "uploaded", label: "Uploaded Contracts" },
            { id: "attention", label: "Needs Attention" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setFilter(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                filter === tab.id
                  ? "bg-[#171717] text-white shadow-2xs"
                  : "bg-white border border-[#E7E5E2] text-[#5F6368] hover:text-[#171717] hover:bg-[#F7F7F5]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative min-w-[240px]">
          <Search className="absolute left-3 top-2.5 w-3.5 h-3.5 text-[#8A8F98]" />
          <input
            type="text"
            placeholder="Search vault documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-[#E7E5E2] bg-white pl-8.5 pr-3 py-1.5 text-xs text-[#171717] placeholder:text-[#8A8F98] focus:border-[#171717] focus:outline-none focus:ring-1 focus:ring-[#171717]"
          />
        </div>
      </div>

      {/* Document Grid / List */}
      {filteredDocs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-[#E7E5E2] bg-white p-12 text-center space-y-4 shadow-2xs">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#F7F7F5] border border-[#E7E5E2] text-[#8A8F98]">
            <FileText className="w-6 h-6" />
          </div>
          <div className="space-y-1 max-w-sm mx-auto">
            <h3 className="text-base font-bold text-[#171717]">
              Your legal documents will appear here
            </h3>
            <p className="text-xs text-[#5F6368] leading-relaxed">
              Create an AI agreement or upload a contract for pre-sign risk audit to populate your vault.
            </p>
          </div>
          <div className="flex justify-center gap-2 pt-2">
            <Link
              href="/dashboard/create"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
            >
              <FilePlus className="w-3.5 h-3.5 text-[#059669]" />
              <span>Create Agreement</span>
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {filteredDocs.map((doc) => {
            const hasAttention = (doc.riskScore ?? 0) >= 2;

            return (
              <div
                key={doc.id}
                className="group rounded-xl border border-[#E7E5E2] bg-white p-5 space-y-4 shadow-2xs hover:border-[#D4D2CD] hover:shadow-xs transition-all flex flex-col justify-between"
              >
                <div className="space-y-3">
                  {/* Top Header */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`flex h-8 w-8 items-center justify-center rounded-lg shrink-0 ${
                          doc.isGenerated
                            ? "bg-[#059669]/10 text-[#059669] border border-[#059669]/20"
                            : "bg-[#F7F7F5] text-[#171717] border border-[#E7E5E2]"
                        }`}
                      >
                        <FileText className="w-4 h-4" />
                      </div>
                      <div className="min-w-0">
                        <span className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98]">
                          {doc.isGenerated ? "AI Drafted" : "Audit Contract"}
                        </span>
                        <h3 className="text-sm font-bold text-[#171717] truncate group-hover:text-[#059669] transition-colors">
                          {doc.title}
                        </h3>
                      </div>
                    </div>

                    <span className="rounded-full bg-[#F7F7F5] border border-[#E7E5E2] px-2 py-0.5 text-[10px] font-mono text-[#5F6368] capitalize shrink-0">
                      {doc.status}
                    </span>
                  </div>

                  {/* Metadata Box */}
                  <div className="rounded-lg bg-[#F7F7F5] border border-[#E7E5E2] p-3 text-xs space-y-1">
                    <div className="flex items-center justify-between text-[11px] text-[#8A8F98]">
                      <span>Type</span>
                      <span className="font-medium text-[#171717] capitalize truncate max-w-[160px]">
                        {doc.type.replace(/_/g, " ")}
                      </span>
                    </div>

                    {doc.flowTitle && (
                      <div className="flex items-center justify-between text-[11px] text-[#8A8F98]">
                        <span>LegalFlow</span>
                        <span className="font-medium text-[#059669] truncate max-w-[160px]">
                          {doc.flowTitle}
                        </span>
                      </div>
                    )}

                    <div className="flex items-center justify-between text-[11px] text-[#8A8F98]">
                      <span>Updated</span>
                      <span className="font-mono text-[#5F6368]">
                        {new Date(doc.createdAt).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Footer Action */}
                <div className="pt-2 flex items-center justify-between border-t border-[#F0EFEA]">
                  <div className="flex items-center gap-1 text-[11px]">
                    {hasAttention ? (
                      <span className="flex items-center gap-1 text-[#B45309] font-medium">
                        <ShieldAlert className="w-3.5 h-3.5" />
                        <span>Needs Attention</span>
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[#059669] font-medium">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        <span>Ready</span>
                      </span>
                    )}
                  </div>

                  <Link
                    href={doc.url}
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#171717] hover:text-[#059669] transition-colors"
                  >
                    <span>Open Workspace</span>
                    <ExternalLink className="w-3.5 h-3.5 text-[#059669]" />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
