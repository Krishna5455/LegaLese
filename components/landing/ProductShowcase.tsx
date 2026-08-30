"use client";

import React from "react";
import Link from "next/link";
import { Check, AlertTriangle, ArrowRight, Download, Sparkles, SlidersHorizontal } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

export function ProductShowcase() {
  return (
    <section className="py-24 sm:py-32 bg-white text-[#171717] border-b border-[#E7E5E2] overflow-hidden">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-12 sm:space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F7F7F5] border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Complete Workspace
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              An all-in-one studio for your contracts.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              Drafting, plain-English translation, pre-sign risk audits, and multi-format exports united in a single focused workspace.
            </p>
          </div>
        </ScrollReveal>

        {/* Huge Full-Width Product Workspace Composition */}
        <div className="rounded-2xl border border-[#E7E5E2] bg-[#F7F7F5] p-3 sm:p-5 shadow-xl perspective-1000">
          <div className="rounded-xl border border-[#E7E5E2] bg-white shadow-sm overflow-hidden">
            {/* Mockup Application Navigation Bar */}
            <div className="h-12 border-b border-[#E7E5E2] bg-[#F7F7F5]/80 px-4 flex items-center justify-between text-xs">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#E7E5E2]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#E7E5E2]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#E7E5E2]" />
                </div>
                <span className="h-4 w-px bg-[#E7E5E2]" />
                <div className="flex items-center gap-2 text-[#171717] font-semibold">
                  <span className="h-5 w-5 rounded bg-[#059669]/10 text-[#059669] flex items-center justify-center font-bold text-[10px]">
                    §
                  </span>
                  <span>LegaLese Workspace</span>
                  <span className="text-[#8A8F98] font-normal">/ agreements / master-services-v1.pdf</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-[#F0FDF4] text-[#166534] border border-[#BBF7D0]">
                  <Check className="w-3 h-3" /> Audit Complete
                </span>
                <span className="text-[11px] font-mono text-[#8A8F98]">Auto-saved 2m ago</span>
              </div>
            </div>

            {/* 3-Column Studio Grid */}
            <div className="grid grid-cols-1 md:grid-cols-12 min-h-[500px]">
              {/* Column 1: Document Tree (2 Cols) */}
              <div className="hidden lg:block md:col-span-3 border-r border-[#E7E5E2] p-4 space-y-4 bg-[#F7F7F5]/40 text-xs">
                <div className="font-mono text-[10px] uppercase tracking-wider text-[#8A8F98]">
                  Contract Index
                </div>
                <div className="space-y-1">
                  <div className="p-2 rounded-lg bg-white border border-[#E7E5E2] font-semibold text-[#171717] flex items-center justify-between shadow-2xs">
                    <span className="truncate">1. Recitals & Parties</span>
                    <Check className="w-3 h-3 text-[#059669]" />
                  </div>
                  <div className="p-2 rounded-lg hover:bg-white text-[#5F6368] flex items-center justify-between transition-colors">
                    <span className="truncate">2. Scope of Services</span>
                    <Check className="w-3 h-3 text-[#059669]" />
                  </div>
                  <div className="p-2 rounded-lg bg-[#FFF7ED] text-[#B45309] font-medium flex items-center justify-between border border-[#FED7AA]">
                    <span className="truncate">3. Payment & Retainer</span>
                    <AlertTriangle className="w-3 h-3 text-[#B45309]" />
                  </div>
                  <div className="p-2 rounded-lg hover:bg-white text-[#5F6368] flex items-center justify-between transition-colors">
                    <span className="truncate">4. IP Assignment</span>
                    <Check className="w-3 h-3 text-[#059669]" />
                  </div>
                  <div className="p-2 rounded-lg hover:bg-white text-[#5F6368] flex items-center justify-between transition-colors">
                    <span className="truncate">5. Termination Terms</span>
                    <AlertTriangle className="w-3 h-3 text-[#B45309]" />
                  </div>
                </div>
              </div>

              {/* Column 2: Interactive Document Canvas (6 Cols) */}
              <div className="md:col-span-8 lg:col-span-6 p-6 sm:p-8 space-y-6 text-xs sm:text-[13px] leading-relaxed text-[#5F6368] overflow-y-auto">
                <div className="border-b border-[#E7E5E2] pb-4">
                  <h3 className="text-base sm:text-lg font-bold uppercase tracking-wider text-[#171717]">
                    Master Services Agreement (MSA)
                  </h3>
                  <p className="text-xs text-[#8A8F98] mt-1 font-mono">
                    Jurisdiction: India • Effective Date: September 2026
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-[#BBF7D0] bg-[#F0FDF4]/40 space-y-2">
                  <span className="font-bold text-xs uppercase tracking-wide text-[#166534] flex items-center gap-1.5">
                    <Check className="w-3.5 h-3.5" /> Clause 3.1: Milestone Payment Safeguard
                  </span>
                  <p className="text-[#171717]">
                    Client shall remit 50% upfront retainer prior to deliverable commencement. The remaining balance shall be payable upon formal milestone acceptance.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-[#FED7AA] bg-[#FFF7ED]/50 space-y-2">
                  <span className="font-bold text-xs uppercase tracking-wide text-[#B45309] flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5" /> Clause 5.2: Termination Cure Period
                  </span>
                  <p className="text-[#171717]">
                    Either party may terminate upon 15 calendar days written notice. Contractor shall be paid for all hours worked up to the termination date.
                  </p>
                </div>

                <div className="flex items-center gap-3 pt-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] text-xs font-medium text-[#171717]">
                    <Download className="w-3.5 h-3.5 text-[#059669]" /> Export Signed PDF
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] text-xs font-medium text-[#171717]">
                    <Download className="w-3.5 h-3.5 text-[#059669]" /> Export DOCX
                  </span>
                </div>
              </div>

              {/* Column 3: Live Inspector & Plain-Language Sidebar (3 Cols) */}
              <div className="md:col-span-4 lg:col-span-3 border-t md:border-t-0 md:border-l border-[#E7E5E2] p-5 space-y-5 bg-[#F7F7F5]/40 text-xs">
                <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-3">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-[#8A8F98]">
                    Pre-Sign Audit
                  </span>
                  <span className="font-mono text-xs font-bold text-[#059669]">
                    Score 88/100
                  </span>
                </div>

                <div className="space-y-2.5">
                  <div className="p-3 rounded-lg bg-white border border-[#E7E5E2] shadow-2xs space-y-1">
                    <span className="font-semibold text-[#171717] block flex items-center gap-1">
                      <Sparkles className="w-3 h-3 text-[#059669]" /> Plain English
                    </span>
                    <p className="text-[#5F6368] text-[11px] leading-relaxed">
                      You are fully protected against uncompensated revisions with a clear 50% upfront deposit.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-white border border-[#E7E5E2] shadow-2xs space-y-1">
                    <span className="font-semibold text-[#171717] block flex items-center gap-1">
                      <SlidersHorizontal className="w-3 h-3 text-[#059669]" /> Verified Fair
                    </span>
                    <p className="text-[#5F6368] text-[11px] leading-relaxed">
                      Intellectual property automatically transfers upon final receipt of payment.
                    </p>
                  </div>
                </div>

                <div className="pt-2">
                  <Link
                    href="/dashboard"
                    className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-[#171717] py-2 text-xs font-medium text-white hover:bg-[#262626] transition-all shadow-xs"
                  >
                    <span>Launch Workspace</span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#059669]" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
