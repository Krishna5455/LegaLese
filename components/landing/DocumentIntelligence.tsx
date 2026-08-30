"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Check, AlertTriangle, FileText, Sparkles } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

interface Finding {
  id: string;
  clauseTitle: string;
  status: "Clear" | "Needs attention" | "High risk";
  statusColor: string;
  badgeBg: string;
  badgeBorder: string;
  summary: string;
  clauseExcerpt: string;
  explanation: string;
  recommendation: string;
}

const findingsData: Finding[] = [
  {
    id: "payment",
    clauseTitle: "Payment Terms",
    status: "Needs attention",
    statusColor: "text-[#B45309]",
    badgeBg: "bg-[#FFF7ED]",
    badgeBorder: "border-[#FED7AA]",
    summary: "Payment deadline is unclear and open to client discretion",
    clauseExcerpt:
      "Client shall remit milestone payments within sixty (60) business days following subjective verification of final deliverables.",
    explanation:
      "A 60-day window tied to 'subjective verification' allows the client to withhold payment indefinitely. Standard terms require 14 to 30 calendar days.",
    recommendation: "Request Net-15 or Net-30 payment schedule with clear milestone sign-off criteria.",
  },
  {
    id: "confidentiality",
    clauseTitle: "Confidentiality",
    status: "Clear",
    statusColor: "text-[#166534]",
    badgeBg: "bg-[#F0FDF4]",
    badgeBorder: "border-[#BBF7D0]",
    summary: "Standard mutual non-disclosure obligations",
    clauseExcerpt:
      "Both parties agree to hold proprietary trade secrets in strict confidence for a period of two (2) years post-termination.",
    explanation:
      "Standard bilateral confidentiality agreement protecting both contractor techniques and client proprietary data.",
    recommendation: "Balanced term. No modifications required.",
  },
  {
    id: "termination",
    clauseTitle: "Termination Notice",
    status: "Needs attention",
    statusColor: "text-[#B45309]",
    badgeBg: "bg-[#FFF7ED]",
    badgeBorder: "border-[#FED7AA]",
    summary: "Notice period may be unfavorable without compensation guarantee",
    clauseExcerpt:
      "Client may terminate this Agreement immediately upon written email notice without obligation to pay for unaccepted work-in-progress.",
    explanation:
      "Immediate termination without compensation leaves your ongoing hours completely unremunerated.",
    recommendation: "Add requirement for 15-day written notice and pro-rata payment for all work completed.",
  },
];

export function DocumentIntelligence() {
  const [activeFinding, setActiveFinding] = useState<Finding>(findingsData[0]);

  return (
    <section id="analyze" className="py-24 sm:py-32 bg-[#F7F7F5] text-[#171717] border-b border-[#E7E5E2]">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Automated Contract Analysis
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              Pre-sign intelligence that spots what you missed.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              Upload any agreement in PDF or DOCX format. LegaLese audits obligations, surfaces high-risk liability traps, and calculates an actionable contract risk score.
            </p>
          </div>
        </ScrollReveal>

        {/* Visual Composition: Realistic Contract Viewer on Left, Findings on Right */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Realistic Contract Viewer with Highlighted Clauses */}
          <div className="lg:col-span-7 rounded-2xl border border-[#E7E5E2] bg-white shadow-sm overflow-hidden">
            {/* Viewer Top Bar */}
            <div className="px-5 py-3.5 bg-[#F7F7F5] border-b border-[#E7E5E2] flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 font-mono text-[#5F6368]">
                <FileText className="w-3.5 h-3.5 text-[#059669]" />
                <span>Freelance_Service_Agreement_Draft.pdf</span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[11px] text-[#8A8F98]">
                <span>100% Zoom</span>
                <span>•</span>
                <span>Page 2 of 4</span>
              </div>
            </div>

            {/* Document Body with Selectable Clauses */}
            <div className="p-6 sm:p-8 space-y-4 text-xs sm:text-[13px] leading-relaxed text-[#5F6368]">
              <div className="font-mono text-[10px] uppercase tracking-wider text-[#8A8F98] pb-1 border-b border-[#E7E5E2]">
                CONTRACT CLAUSES & RISK HIGHLIGHTS
              </div>

              {findingsData.map((f) => {
                const isSelected = activeFinding.id === f.id;
                return (
                  <div
                    key={f.id}
                    onClick={() => setActiveFinding(f)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-[#F7F7F5] border-[#171717] ring-1 ring-[#171717]/20 shadow-xs"
                        : "bg-white border-[#E7E5E2] hover:bg-[#F7F7F5]/60 hover:border-[#D4D2CD]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-bold text-[#171717] text-xs">
                        {f.clauseTitle}
                      </span>
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${f.badgeBg} ${f.statusColor} ${f.badgeBorder}`}
                      >
                        {f.status}
                      </span>
                    </div>
                    <p className="text-[12px] text-[#5F6368] line-clamp-2">
                      &ldquo;{f.clauseExcerpt}&rdquo;
                    </p>
                  </div>
                );
              })}

              <div className="pt-2 flex items-center justify-between text-xs text-[#8A8F98]">
                <span>3 risk areas flagged for review</span>
                <span className="font-mono text-[11px] text-[#059669]">AI Analysis Verified ✓</span>
              </div>
            </div>
          </div>

          {/* Right Column: High-Fidelity Risk Score & Inspector Panel */}
          <div className="lg:col-span-5 space-y-6">
            <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-6 shadow-sm">
              {/* Header with Risk Gauge */}
              <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-5">
                <div>
                  <span className="text-[11px] font-mono uppercase tracking-wider text-[#8A8F98] block">
                    Contract Health
                  </span>
                  <h3 className="text-base sm:text-lg font-bold text-[#171717]">
                    Contract Analysis
                  </h3>
                </div>

                <div className="text-right">
                  <div className="flex items-baseline gap-1 justify-end">
                    <span className="text-2xl sm:text-3xl font-extrabold text-[#B45309]">72</span>
                    <span className="text-xs text-[#8A8F98] font-mono">/ 100</span>
                  </div>
                  <span className="text-[10px] font-semibold text-[#B45309] uppercase tracking-wide block">
                    Needs Attention
                  </span>
                </div>
              </div>

              {/* Finding Details */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm text-[#171717]">
                    {activeFinding.clauseTitle}
                  </span>
                  <span
                    className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded border ${activeFinding.badgeBg} ${activeFinding.statusColor} ${activeFinding.badgeBorder}`}
                  >
                    {activeFinding.status === "Clear" ? (
                      <Check className="w-3 h-3" />
                    ) : (
                      <AlertTriangle className="w-3 h-3" />
                    )}
                    {activeFinding.status}
                  </span>
                </div>

                <div className="p-3.5 rounded-lg bg-[#F7F7F5] border border-[#E7E5E2] text-xs text-[#5F6368] space-y-1">
                  <span className="font-semibold text-[#171717] block">Finding Summary:</span>
                  <p>{activeFinding.summary}</p>
                </div>

                <div className="p-3.5 rounded-lg bg-[#F7F7F5] border border-[#E7E5E2] text-xs text-[#5F6368] space-y-1">
                  <span className="font-semibold text-[#171717] block">Explanation:</span>
                  <p>{activeFinding.explanation}</p>
                </div>

                <div className="p-3.5 rounded-lg bg-[#059669]/10 border border-[#059669]/20 text-xs text-[#171717] space-y-1">
                  <span className="font-semibold text-[#059669] block flex items-center gap-1">
                    <Sparkles className="w-3 h-3" /> Recommended Remedy:
                  </span>
                  <p>{activeFinding.recommendation}</p>
                </div>

                {/* Forensic Audit Desk Photography */}
                <div className="relative rounded-xl overflow-hidden border border-[#E7E5E2] h-40 group">
                  <Image
                    src="/images/forensic_legal_audit.jpg"
                    alt="Forensic contract audit with redline corrections and magnifying glass"
                    fill
                    sizes="(max-width: 1024px) 100vw, 450px"
                    className="object-cover object-center filter brightness-[0.96] transition-transform duration-700 ease-out group-hover:scale-102"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-transparent to-transparent flex items-end p-3">
                    <div className="text-white text-xs flex items-center justify-between w-full">
                      <span className="font-mono text-[10px] uppercase font-bold text-[#FED7AA]">
                        Forensic Pre-Sign Scrutiny
                      </span>
                      <span className="text-[10px] text-white/80">Every Redline Analyzed</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
