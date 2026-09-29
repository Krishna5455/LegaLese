"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Calendar as CalendarIcon,
  Search,
  FileText,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import type { LegalCalendarEvent } from "@/types/legalflow";

interface LegalCalendarViewProps {
  initialEvents: LegalCalendarEvent[];
}

export function LegalCalendarView({ initialEvents }: LegalCalendarViewProps) {
  const [filterType, setFilterType] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredEvents = initialEvents.filter((evt) => {
    const matchesFilter = filterType === "all" || evt.type === filterType;
    const matchesSearch =
      searchQuery.trim() === "" ||
      evt.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (evt.description && evt.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (evt.responsibleParty && evt.responsibleParty.toLowerCase().includes(searchQuery.toLowerCase()));

    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#E7E5E2] pb-6">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-1.5 rounded-full bg-[#059669]/10 px-2.5 py-0.5 text-[11px] font-semibold text-[#059669]">
            <CalendarIcon className="w-3 h-3" />
            <span>Legal Deadlines &amp; Obligations</span>
          </div>
          <h1 className="heading-page text-[#171717]">Legal Calendar</h1>
          <p className="text-xs sm:text-[14px] text-[#5F6368]">
            Centralized timeline of extracted payment dates, delivery milestones, renewal windows, and notice periods.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-md bg-[#059669]/10 border border-[#059669]/20 px-2.5 py-1 text-xs font-semibold text-[#059669]">
            {initialEvents.length} Total Dates
          </span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Category Filter Chips */}
        <div className="flex flex-wrap items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          {[
            { id: "all", label: "All Dates" },
            { id: "payment", label: "Payments" },
            { id: "delivery", label: "Deliveries" },
            { id: "renewal", label: "Renewals & Expirations" },
            { id: "notice", label: "Notices" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setFilterType(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                filterType === tab.id
                  ? "bg-[#171717] text-white shadow-2xs"
                  : "bg-white border border-[#E7E5E2] text-[#5F6368] hover:text-[#171717] hover:bg-[#F7F7F5]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Search Bar */}
        <div className="relative min-w-[240px]">
          <Search className="absolute left-3 top-2.5 w-3.5 h-3.5 text-[#8A8F98]" />
          <input
            type="text"
            placeholder="Search dates or parties..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-[#E7E5E2] bg-white pl-8.5 pr-3 py-1.5 text-xs text-[#171717] placeholder:text-[#8A8F98] focus:border-[#171717] focus:outline-none focus:ring-1 focus:ring-[#171717]"
          />
        </div>
      </div>

      {/* Events Timeline / List Container */}
      {filteredEvents.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-[#E7E5E2] bg-white p-12 text-center space-y-4 shadow-2xs">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#F7F7F5] border border-[#E7E5E2] text-[#8A8F98]">
            <CalendarIcon className="w-6 h-6" />
          </div>
          <div className="space-y-1 max-w-sm mx-auto">
            <h3 className="text-base font-bold text-[#171717]">
              No upcoming legal dates
            </h3>
            <p className="text-xs text-[#5F6368] leading-relaxed">
              Important dates extracted from your legal documents and active LegalFlows will appear here automatically.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-4 py-2 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
            >
              <Sparkles className="w-3.5 h-3.5 text-[#059669]" />
              <span>Start a LegalFlow</span>
            </Link>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredEvents.map((evt) => {
            const isPayment = evt.type === "payment";
            const isDelivery = evt.type === "delivery";
            const isRenewal = evt.type === "renewal";

            return (
              <div
                key={evt.id}
                className="group rounded-xl border border-[#E7E5E2] bg-white p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-2xs hover:border-[#D4D2CD] hover:shadow-xs transition-all"
              >
                {/* Left: Date & Details */}
                <div className="flex items-start gap-3.5 min-w-0">
                  {/* Date badge */}
                  <div className="flex flex-col items-center justify-center rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] px-3 py-2 shrink-0 min-w-[72px] text-center">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98] font-bold">
                      {isPayment ? "Payment" : isDelivery ? "Milestone" : isRenewal ? "Renewal" : "Deadline"}
                    </span>
                    <span className="text-xs font-bold text-[#171717] mt-0.5">
                      {evt.date}
                    </span>
                  </div>

                  {/* Info */}
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="text-sm font-bold text-[#171717] group-hover:text-[#059669] transition-colors">
                        {evt.title}
                      </h3>
                      {evt.amount && (
                        <span className="rounded bg-[#059669]/10 text-[#059669] border border-[#059669]/20 px-2 py-0.2 text-[11px] font-mono font-bold">
                          {evt.amount}
                        </span>
                      )}
                      <span className="rounded-full bg-[#F7F7F5] border border-[#E7E5E2] px-2 py-0.2 text-[10px] font-mono text-[#5F6368] uppercase">
                        {evt.status}
                      </span>
                    </div>

                    <p className="text-xs text-[#5F6368]">
                      {evt.description}
                    </p>

                    {evt.responsibleParty && (
                      <p className="text-[11px] text-[#8A8F98]">
                        Responsible: <span className="text-[#171717] font-medium">{evt.responsibleParty}</span>
                      </p>
                    )}
                  </div>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-2 shrink-0">
                  {evt.flowId ? (
                    <Link
                      href={`/dashboard/flow/${evt.flowId}`}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs"
                    >
                      <span>Open LegalFlow</span>
                      <ArrowRight className="w-3.5 h-3.5 text-[#059669]" />
                    </Link>
                  ) : evt.documentId ? (
                    <Link
                      href={`/dashboard/documents/${evt.documentId}`}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-3.5 py-1.5 text-xs font-semibold text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-2xs"
                    >
                      <FileText className="w-3.5 h-3.5 text-[#8A8F98]" />
                      <span>View Document</span>
                    </Link>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
