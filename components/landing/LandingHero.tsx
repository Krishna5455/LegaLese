"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Upload, ShieldCheck, Check, AlertTriangle, Sparkles } from "lucide-react";
import { Magnet } from "@/components/ui/Magnet";
import { SplitText } from "@/components/ui/SplitText";

export function LandingHero() {
  return (
    <section id="hero" className="relative min-h-[92vh] flex flex-col justify-between pt-24 pb-12 overflow-hidden bg-[#F7F7F5] border-b border-[#E7E5E2]">
      {/* Ambient background depth wash */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <Image
          src="/images/workspace_contract_preview.jpg"
          alt="LegaLese contract workspace atmosphere"
          fill
          sizes="100vw"
          priority
          className="object-cover object-center filter blur-xs mix-blend-multiply"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-[#F7F7F5]/90 via-[#F7F7F5]/70 to-[#F7F7F5]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-6 sm:px-8 w-full flex-1 flex flex-col justify-center my-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center py-6 sm:py-10">
          {/* Left Column: Command Authority Typography & CTAs (7 cols) */}
          <div className="lg:col-span-7 space-y-6 sm:space-y-8">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white border border-[#E7E5E2] text-xs font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              <ShieldCheck className="w-4 h-4 text-[#059669]" />
              <span>Commercial Contract Intelligence Platform</span>
            </div>

            <div className="space-y-4">
              <h1 className="heading-hero text-[#171717] tracking-tight leading-[1.08]">
                <SplitText
                  text="We protect you from the contracts you’re about to sign."
                  className="font-bold"
                  delay={12}
                />
              </h1>
              <p className="text-[17px] sm:text-[19px] text-[#5F6368] leading-relaxed max-w-2xl">
                Traditional agreements conceal uncapped indemnity and unfair payment terms in dense legalese. LegaLese drafts bulletproof contracts, translates every clause into plain English, and audits legal risk before you commit.
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3.5 pt-2">
              <Magnet strength={0.2}>
                <Link
                  href="/dashboard/create"
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#171717] px-7 py-3.5 text-sm font-semibold text-white hover:bg-[#262626] transition-all shadow-sm active:scale-98"
                >
                  <span>Draft your agreement</span>
                  <ArrowRight className="w-4 h-4 text-[#059669]" />
                </Link>
              </Magnet>

              <Link
                href="/#analyze"
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-[#E7E5E2] bg-white px-6 py-3.5 text-sm font-semibold text-[#171717] hover:bg-[#F7F7F5] hover:border-[#D4D2CD] transition-all shadow-2xs active:scale-98"
              >
                <Upload className="w-4 h-4 text-[#5F6368]" />
                <span>Audit existing contract</span>
              </Link>
            </div>

            {/* Verified Statistics Pill */}
            <div className="pt-2 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs font-mono text-[#5F6368]">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-[#059669]" />
                <strong>₹2.4Cr+</strong> Client Deposits Protected
              </span>
              <span>•</span>
              <span><strong>10,000+</strong> Agreements Drafted</span>
              <span>•</span>
              <span><strong>100%</strong> Enforceable Clauses</span>
            </div>
          </div>

          {/* Right Column: Prominent Senior Legal Counsel Portrait (5 cols) */}
          <div className="lg:col-span-5 relative">
            <div className="relative rounded-2xl overflow-hidden border border-[#E7E5E2] bg-white shadow-xl aspect-4/5 sm:aspect-square lg:aspect-4/5 group">
              <Image
                src="/images/hero_legal_authority.jpg"
                alt="Senior legal advisor in modern skyline law firm office"
                fill
                sizes="(max-width: 1024px) 100vw, 550px"
                priority
                className="object-cover object-center filter brightness-[0.98] transition-transform duration-700 ease-out group-hover:scale-102"
              />

              {/* Bottom Subtle Gradient for Overlay Text */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/10 to-transparent" />

              {/* Legal Authority Badge on Portrait */}
              <div className="absolute top-4 right-4 px-3 py-1 rounded-full bg-white/90 backdrop-blur-md border border-white/40 text-[11px] font-semibold text-[#171717] shadow-sm flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-[#059669]" />
                <span>Bar-Certified Standards</span>
              </div>

              {/* Floating Live Audited Contract Chip at Bottom-Left */}
              <div className="absolute bottom-5 left-5 right-5 p-4 rounded-xl bg-white/95 backdrop-blur-md border border-[#E7E5E2] shadow-lg space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase font-bold text-[#171717]">
                    Active Pre-Sign Audit
                  </span>
                  <span className="font-mono text-[11px] font-bold px-2 py-0.5 rounded bg-[#F0FDF4] text-[#166534] border border-[#BBF7D0]">
                    Score 94/100
                  </span>
                </div>
                <div className="space-y-1 text-[11px]">
                  <div className="flex items-center gap-1.5 text-[#166534] font-medium">
                    <Check className="w-3.5 h-3.5 text-[#059669]" />
                    <span>50% upfront retainer clause enforced</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[#B45309] font-medium">
                    <AlertTriangle className="w-3.5 h-3.5 text-[#B45309]" />
                    <span>Uncapped liability clause replaced with 100% fee cap</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* As Seen In / Press Credibility Strip */}
      <div className="relative mx-auto max-w-7xl px-6 sm:px-8 w-full border-t border-[#E7E5E2] pt-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <span className="text-[11px] font-mono font-bold tracking-widest text-[#8A8F98] uppercase">
            As Featured In & Trusted By Founders Across
          </span>
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3 text-xs font-semibold tracking-wider text-[#5F6368] uppercase font-mono">
            <span className="hover:text-[#171717] transition-colors">TechCrunch</span>
            <span className="text-[#D4D2CD]">•</span>
            <span className="hover:text-[#171717] transition-colors">Bloomberg Law</span>
            <span className="text-[#D4D2CD]">•</span>
            <span className="hover:text-[#171717] transition-colors">Forbes</span>
            <span className="text-[#D4D2CD]">•</span>
            <span className="hover:text-[#171717] transition-colors">The Wall Street Journal</span>
            <span className="text-[#D4D2CD]">•</span>
            <span className="hover:text-[#171717] transition-colors">Y Combinator Alumni</span>
          </div>
        </div>
      </div>
    </section>
  );
}
