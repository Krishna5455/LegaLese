import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";

export function FinalCta() {
  return (
    <section className="py-24 bg-[#111827] text-white border-b border-[#1F2937] relative overflow-hidden">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 text-center space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-xs font-semibold text-[#10B981]">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Complete LegalFlow</span>
        </div>

        <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white max-w-2xl mx-auto leading-tight">
          Don&apos;t just generate the legal document. <br />
          <span className="text-[#10B981]">Complete the legal journey.</span>
        </h2>

        <p className="text-sm sm:text-base text-white/70 max-w-xl mx-auto leading-relaxed">
          From natural language intent to drafted agreements, AI risk auditing, and automated milestone tracking.
        </p>

        <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link
            href="/dashboard"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-white px-7 py-3.5 text-sm font-semibold text-[#111827] hover:bg-[#F3F4F6] transition-all shadow-md active:scale-98"
          >
            <span>Start a LegalFlow</span>
            <ArrowRight className="w-4 h-4 text-[#059669]" />
          </Link>

          <Link
            href="/dashboard#upload"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-white/20 bg-white/5 px-6 py-3.5 text-sm font-semibold text-white hover:bg-white/10 transition-all shadow-2xs active:scale-98"
          >
            <span>Analyze a contract</span>
          </Link>
        </div>
      </div>
    </section>
  );
}
