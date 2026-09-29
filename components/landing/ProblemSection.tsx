"use client";

import { AlertTriangle, Clock, FileQuestion } from "lucide-react";
import { ScrollReveal } from "@/components/landing/ScrollReveal";

export function ProblemSection() {
  return (
    <section className="py-20 bg-white border-b border-[#E5E5E3]">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">
        <ScrollReveal className="text-center space-y-3 max-w-2xl mx-auto">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827]">
            Why standard legal tools fail businesses
          </h2>
          <p className="text-sm text-[#4B5563]">
            Traditional legal software either generates a static template or acts as a dumb PDF archive. Neither solves the actual journey.
          </p>
        </ScrollReveal>

        <ScrollReveal className="grid grid-cols-1 md:grid-cols-3 gap-6" delay={0.15}>
          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-amber-50 text-amber-700 w-fit">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">Concealed Risk Traps</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Uncapped indemnities, automatic renewals, and one-sided termination windows hide in dense clauses before you sign.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-rose-50 text-rose-700 w-fit">
              <FileQuestion className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">The Disconnected Journey</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Drafting happens in one tool, review in another, and signing via email. You lose context at every handoff.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-blue-50 text-blue-700 w-fit">
              <Clock className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">Forgotten Post-Sign Duties</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Once signed, milestone payments, renewal notice windows, and deliverable deadlines disappear into forgotten folders.
            </p>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
