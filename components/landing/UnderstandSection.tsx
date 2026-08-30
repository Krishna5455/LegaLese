"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Sparkles, Check } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

const sampleClauses = [
  {
    id: "indemnity",
    clauseTitle: "Indemnification & Unlimited Liability",
    risk: "High Risk",
    riskColor: "text-[#B91C1C] bg-[#FEF2F2] border-[#FECACA]",
    originalText:
      "Contractor covenants, warrants, and agrees to indemnify, defend, and hold harmless Client and its respective affiliates against any and all foreseeable and unforeseeable claims, liabilities, judgments, settlements, attorney fees, or losses whatsoever arising out of or related to the Services provided hereunder.",
    plainTranslation:
      "If anything goes wrong, you are personally and financially responsible for all client losses, court costs, and legal bills, with no dollar limit whatsoever.",
    suggestedFix:
      "Amend clause to cap liability to 100% of the total fees actually paid to you under this statement of work.",
  },
  {
    id: "acceptance",
    clauseTitle: "Milestone Acceptance & Deferred Remittance",
    risk: "Medium Risk",
    riskColor: "text-[#B45309] bg-[#FFF7ED] border-[#FED7AA]",
    originalText:
      "Payment of the deferred portion of aggregate remuneration shall be contingent upon Client's sole and subjective determination of deliverable completeness, with Client retaining the right to request unlimited iterations without additional compensation.",
    plainTranslation:
      "The client can reject your work for any subjective reason and force you to do infinite free revisions before releasing your final milestone payment.",
    suggestedFix:
      "Enforce maximum 2 revision rounds and establish a 7-day automatic acceptance period upon deliverable submission.",
  },
  {
    id: "ip-transfer",
    clauseTitle: "Intellectual Property Transfer Upon Final Remittance",
    risk: "Low Risk",
    riskColor: "text-[#166534] bg-[#F0FDF4] border-[#BBF7D0]",
    originalText:
      "All intellectual property, proprietary discoveries, copyright interests, and design assets shall automatically vest in Client solely upon full and irrevocable remittance of all outstanding invoicing obligations.",
    plainTranslation:
      "You retain 100% ownership of your work until the client pays every invoice in full. Ownership transfers only after payment clears.",
    suggestedFix:
      "Standard balanced clause. No amendments required.",
  },
];

export function UnderstandSection() {
  const [activeClause, setActiveClause] = useState(sampleClauses[0]);

  return (
    <section id="understand" className="py-24 sm:py-32 bg-white text-[#171717] border-b border-[#E7E5E2]">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-12 sm:space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F7F7F5] border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Plain English Translation
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              Understand what you&apos;re signing before you commit.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              Traditional contracts are written in archaic jargon that conceals one-sided liabilities and payment delays. LegaLese translates dense legal clauses into everyday terms with actionable advice.
            </p>
          </div>
        </ScrollReveal>

        {/* Split Screen Workspace Canvas */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Interactive Clause Picker + Legal Advisory Image (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-[#8A8F98]">
              Select A Clause To Translate
            </div>

            <div className="space-y-2.5">
              {sampleClauses.map((clause) => {
                const isSelected = activeClause.id === clause.id;
                return (
                  <button
                    key={clause.id}
                    type="button"
                    onClick={() => setActiveClause(clause)}
                    className={`w-full p-4 rounded-xl border text-left transition-all ${
                      isSelected
                        ? "bg-white border-[#059669] ring-1 ring-[#059669]/20 shadow-sm"
                        : "bg-[#F7F7F5] border-[#E7E5E2] hover:bg-white hover:border-[#D4D2CD]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <span className="font-semibold text-xs text-[#171717]">
                        {clause.clauseTitle}
                      </span>
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded border shrink-0 ${clause.riskColor}`}
                      >
                        {clause.risk}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#5F6368] line-clamp-1 font-serif">
                      {clause.originalText}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Large Photographic Legal Advisory Context Card */}
            <div className="relative rounded-2xl overflow-hidden border border-[#E7E5E2] shadow-sm h-52 group">
              <Image
                src="/images/legal_advisory_consult.jpg"
                alt="Corporate counsel and founder reviewing legal agreement on tablet"
                fill
                sizes="(max-width: 1024px) 100vw, 500px"
                className="object-cover object-center filter brightness-[0.95] transition-transform duration-700 ease-out group-hover:scale-102"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent flex items-end p-4">
                <div className="text-white text-xs font-medium">
                  <span className="font-mono text-[10px] uppercase text-[#10B981] font-bold block">
                    Collaboration Mode
                  </span>
                  <span>Clear, transparent legal terms that both parties can sign with confidence.</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: High-Fidelity Translation & Counter-Proposal Card (7 cols) */}
          <div className="lg:col-span-7 rounded-2xl border border-[#E7E5E2] bg-[#F7F7F5] p-6 sm:p-8 space-y-6 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-4">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#8A8F98] block">
                  Clause Translation
                </span>
                <h3 className="text-base sm:text-lg font-bold text-[#171717]">
                  {activeClause.clauseTitle}
                </h3>
              </div>
              <span
                className={`text-xs font-semibold px-2.5 py-1 rounded-md border ${activeClause.riskColor}`}
              >
                {activeClause.risk}
              </span>
            </div>

            {/* Original Text */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#8A8F98] flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-[#8A8F98]" /> Original Contract Text
              </span>
              <div className="p-4 rounded-xl bg-white border border-[#E7E5E2] font-serif text-xs leading-relaxed text-[#5F6368] italic">
                &ldquo;{activeClause.originalText}&rdquo;
              </div>
            </div>

            {/* Plain English Translation */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#059669] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" /> What It Actually Means For You
              </span>
              <div className="p-4 rounded-xl bg-white border border-[#BBF7D0] text-xs leading-relaxed text-[#171717] font-medium shadow-2xs">
                {activeClause.plainTranslation}
              </div>
            </div>

            {/* Actionable Remedy */}
            <div className="p-4 rounded-xl bg-[#F0FDF4] border border-[#BBF7D0] space-y-1.5">
              <span className="text-xs font-bold text-[#166534] flex items-center gap-1.5">
                <Check className="w-3.5 h-3.5 text-[#059669]" /> Recommended Counter-Proposal
              </span>
              <p className="text-xs text-[#5F6368] leading-relaxed">
                {activeClause.suggestedFix}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
