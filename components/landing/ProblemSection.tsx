"use client";

import React, { useState } from "react";
import Image from "next/image";
import { AlertTriangle, Sparkles, Check } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

export function ProblemSection() {
  const [showSimplified, setShowSimplified] = useState(true);

  return (
    <section id="problem" className="py-24 sm:py-32 bg-[#F7F7F5] text-[#171717] border-b border-[#E7E5E2] relative overflow-hidden">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-12 sm:space-y-16">
        {/* Editorial Section Headline */}
        <ScrollReveal>
          <div className="max-w-4xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              The Fundamental Problem
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              Contracts are everywhere.
              <br />
              <span className="text-[#5F6368]">Understanding them shouldn&apos;t require a law degree.</span>
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6] max-w-2xl">
              Traditional contracts bury uncapped indemnity, hidden revision cycles, and delayed payment terms beneath pages of boilerplate. When disputes happen, ambiguity always favors the party with more legal leverage.
            </p>
          </div>
        </ScrollReveal>

        {/* Large Visual Boardroom Dispute Composition */}
        <div className="relative rounded-2xl overflow-hidden border border-[#E7E5E2] shadow-md h-[340px] sm:h-[440px] group">
          <Image
            src="/images/contract_dispute_room.jpg"
            alt="Corporate boardroom contract negotiation and dispute review"
            fill
            sizes="(max-width: 1024px) 100vw, 1200px"
            className="object-cover object-center filter brightness-[0.92] transition-transform duration-700 ease-out group-hover:scale-101"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent" />
          <div className="absolute bottom-6 left-6 right-6 sm:bottom-8 sm:left-8 sm:right-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4 text-white">
            <div className="max-w-xl space-y-1.5">
              <span className="text-[11px] font-mono uppercase tracking-widest text-[#FED7AA] font-bold">
                The Reality of Unchecked Boilerplate
              </span>
              <h3 className="text-lg sm:text-xl font-bold">
                73% of independent contractors sign agreements with uncapped liability without realizing it.
              </h3>
            </div>
            <span className="text-xs font-mono text-white/70">
              Source: LegalTech Commercial Risk Survey
            </span>
          </div>
        </div>

        {/* Large Visual Contract Comparison Canvas */}
        <div className="relative rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-10 shadow-sm space-y-6">
          {/* Interactive Mode Switcher Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#E7E5E2] pb-5">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-[#171717]" />
              <span className="font-mono text-xs font-semibold text-[#171717] uppercase tracking-wider">
                Exhibit A: Standard Master Services Agreement
              </span>
            </div>

            {/* Toggle Switch */}
            <div className="inline-flex items-center p-1 rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] self-start sm:self-auto">
              <button
                type="button"
                suppressHydrationWarning
                onClick={() => setShowSimplified(false)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  !showSimplified
                    ? "bg-white text-[#171717] font-semibold shadow-2xs border border-[#E7E5E2]"
                    : "text-[#5F6368] hover:text-[#171717]"
                }`}
              >
                Confusing Legalese
              </button>
              <button
                type="button"
                suppressHydrationWarning
                onClick={() => setShowSimplified(true)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  showSimplified
                    ? "bg-[#171717] text-white font-semibold shadow-2xs"
                    : "text-[#5F6368] hover:text-[#171717]"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 text-[#059669]" />
                <span>LegaLese Simplified</span>
              </button>
            </div>
          </div>

          {/* Document Content Viewport */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            {/* Left: Intimidating Contract Page */}
            <div className="lg:col-span-7 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5]/70 p-6 sm:p-8 space-y-4 font-serif text-xs sm:text-[13px] leading-relaxed text-[#5F6368] relative">
              <div className="border-b border-[#E7E5E2] pb-3 font-sans">
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#8A8F98] block">
                  Original Legal Text • Section 8.3
                </span>
                <h4 className="font-bold text-sm text-[#171717] mt-0.5">
                  Limitation of Liability & Indemnification Covenants
                </h4>
              </div>

              <p className={!showSimplified ? "text-[#171717]" : "text-[#8A8F98] line-through opacity-60"}>
                &ldquo;Contractor shall indemnify, defend, and hold harmless Client, its parent companies, subsidiaries, officers, agents, and successors from and against any and all claims, liabilities, obligations, judgments, costs, and attorney fees arising directly or indirectly out of Contractor&apos;s performance or non-performance under this Statement of Work, without any monetary cap, dollar limitation, or statutory expiration period whatsoever.&rdquo;
              </p>

              <p className={!showSimplified ? "text-[#171717]" : "text-[#8A8F98] line-through opacity-60"}>
                &ldquo;Client reserves the unilateral right to withhold milestone disbursements pending sole, subjective satisfaction of deliverable revisions for an indefinite cure period not to exceed ninety (90) business days from submission date.&rdquo;
              </p>

              {!showSimplified && (
                <div className="p-3 rounded-lg bg-[#FEF2F2] border border-[#FECACA] font-sans text-xs text-[#991B1B] flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Archaic clause structure conceals unlimited personal financial liability.</span>
                </div>
              )}
            </div>

            {/* Right: LegaLese Real-Time Plain English Extraction */}
            <div className="lg:col-span-5 rounded-xl border border-[#059669]/30 bg-[#F0FDF4]/40 p-6 sm:p-8 space-y-5 flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-[#BBF7D0] pb-3">
                  <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#059669] flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5" />
                    Plain English Reality
                  </span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-[#FFF7ED] text-[#B45309] border border-[#FED7AA]">
                    Action Required
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="p-3.5 rounded-lg bg-white border border-[#E7E5E2] space-y-1">
                    <span className="text-xs font-bold text-[#B91C1C] flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> 1. Uncapped Personal Liability
                    </span>
                    <p className="text-xs text-[#5F6368] leading-relaxed">
                      You are personally responsible for paying all client losses without any dollar ceiling.
                    </p>
                  </div>

                  <div className="p-3.5 rounded-lg bg-white border border-[#E7E5E2] space-y-1">
                    <span className="text-xs font-bold text-[#B45309] flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> 2. 90-Day Payment Delay
                    </span>
                    <p className="text-xs text-[#5F6368] leading-relaxed">
                      The client can freeze your final payout for up to 90 days on subjective grounds.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-white border border-[#059669]/25 text-xs text-[#171717] space-y-1.5">
                <span className="font-semibold text-[#059669] flex items-center gap-1">
                  <Check className="w-3.5 h-3.5" /> Recommended Fix:
                </span>
                <p className="text-xs text-[#5F6368]">
                  Cap liability to 100% of fees paid, and enforce a 7-day automatic deemed acceptance window.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
