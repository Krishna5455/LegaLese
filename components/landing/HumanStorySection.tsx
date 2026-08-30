import React from "react";
import Image from "next/image";
import Link from "next/link";
import { ArrowRight, ShieldCheck } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

export function HumanStorySection() {
  return (
    <section className="py-24 sm:py-32 bg-[#F7F7F5] text-[#171717] border-b border-[#E7E5E2] relative overflow-hidden">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-12 sm:space-y-16">
        {/* Editorial Narrative Split Header */}
        <ScrollReveal>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-end">
            <div className="lg:col-span-8 space-y-4">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
                Built For Real Work
              </div>
              <h2 className="heading-landing-section text-[#171717] tracking-tight">
                Never wonder what you agreed to again.
              </h2>
            </div>
            <div className="lg:col-span-4">
              <p className="text-[15px] sm:text-[16px] text-[#5F6368] leading-relaxed">
                From initial statements of work to master services agreements, LegaLese gives founders and independent professionals complete contractual clarity without lawyer retainer fees.
              </p>
            </div>
          </div>
        </ScrollReveal>

        {/* Cinematic Large Photographic Composition */}
        <div className="relative rounded-2xl overflow-hidden border border-[#E7E5E2] shadow-md h-[400px] sm:h-[520px] lg:h-[600px] group">
          <Image
            src="/images/human_contract_story.jpg"
            alt="Founder reviewing legal contract on laptop in bright architectural studio"
            fill
            sizes="(max-width: 1024px) 100vw, 1200px"
            className="object-cover object-center filter brightness-[0.98] transition-transform duration-700 ease-out group-hover:scale-102"
          />

          {/* Subtle Ambient Vignette Overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />

          {/* Floating High-Contrast Editorial Story Banner */}
          <div className="absolute bottom-6 left-6 right-6 sm:bottom-10 sm:left-10 sm:right-10 flex flex-col sm:flex-row sm:items-end justify-between gap-6">
            <div className="max-w-xl text-white space-y-2">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/20 backdrop-blur-md text-white text-xs font-semibold">
                <ShieldCheck className="w-4 h-4 text-[#10B981]" />
                Fair Terms Enforced
              </div>
              <h3 className="text-xl sm:text-2xl lg:text-3xl font-bold tracking-tight text-white leading-snug">
                &ldquo;We signed our largest enterprise client contract in 48 hours without spending ₹50,000 on outside counsel.&rdquo;
              </h3>
              <p className="text-xs sm:text-sm text-white/80 font-mono">
                Elena Rostova • Design Director & Agency Founder
              </p>
            </div>

            <div className="shrink-0">
              <Link
                href="/dashboard/create"
                className="inline-flex items-center gap-2 rounded-xl bg-white px-5 py-3 text-xs sm:text-sm font-semibold text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-lg active:scale-98"
              >
                <span>Draft your agreement</span>
                <ArrowRight className="w-4 h-4 text-[#059669]" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
