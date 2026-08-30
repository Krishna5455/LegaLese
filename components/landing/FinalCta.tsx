import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Upload, ShieldCheck } from "lucide-react";
import { Magnet } from "@/components/ui/Magnet";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

export function FinalCta() {
  return (
    <section className="py-28 sm:py-36 bg-[#0B0F19] text-white border-b border-[#1E293B] relative overflow-hidden">
      {/* Background Architectural Twilight Layer */}
      <div className="absolute inset-0 pointer-events-none opacity-35">
        <Image
          src="/images/enterprise_legal_future.jpg"
          alt="Modern legal tech architectural glass headquarters at twilight"
          fill
          sizes="100vw"
          className="object-cover object-center filter brightness-[0.7] mix-blend-screen scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#0B0F19] via-[#0B0F19]/80 to-[#0B0F19]" />
      </div>

      <div className="relative mx-auto max-w-4xl px-6 sm:px-8 text-center space-y-8">
        <ScrollReveal>
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-[12px] font-semibold tracking-wider text-[#10B981] uppercase shadow-2xs">
              <ShieldCheck className="w-3.5 h-3.5" />
              Contract Certainty
            </div>
            <h2 className="heading-hero text-white tracking-tight max-w-3xl mx-auto">
              Your next contract shouldn&apos;t be a guessing game.
            </h2>
            <p className="text-[17px] sm:text-[18px] text-white/70 leading-[1.6] max-w-xl mx-auto">
              Draft your first enforceable agreement or audit an existing contract in less than two minutes.
            </p>
          </div>
        </ScrollReveal>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 pt-2">
          <Magnet strength={0.25}>
            <Link
              href="/dashboard/create"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-white px-7 py-3.5 text-sm font-semibold text-[#0B0F19] hover:bg-[#F1F5F9] transition-all shadow-lg active:scale-98"
            >
              <span>Get Started Now</span>
              <ArrowRight className="w-4 h-4 text-[#059669]" />
            </Link>
          </Magnet>

          <Link
            href="/dashboard#upload"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-white/20 bg-white/10 backdrop-blur-sm px-7 py-3.5 text-sm font-semibold text-white hover:bg-white/20 transition-all shadow-2xs active:scale-98"
          >
            <Upload className="w-4 h-4 text-white/80" />
            <span>Analyze an existing contract</span>
          </Link>
        </div>

        <div className="pt-4 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-white/60">
          <span>No credit card required</span>
          <span>•</span>
          <span>Instant PDF & DOCX export</span>
          <span>•</span>
          <span>Zero attorney retainers</span>
        </div>
      </div>
    </section>
  );
}
