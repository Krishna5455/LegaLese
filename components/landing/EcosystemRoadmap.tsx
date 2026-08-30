import React from "react";
import { Users, UserCheck, SlidersHorizontal, Sparkles } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

const roadmapItems = [
  {
    title: "Community Agreement Library",
    description:
      "Peer-reviewed contract templates tailored for modern software developers, designers, agencies, and cross-border consultants.",
    icon: Users,
    phase: "Q4 2026",
    status: "COMING SOON",
  },
  {
    title: "1-Click Legal Expert Review",
    description:
      "Seamless escalation to vetted commercial attorneys for nuanced high-value transactions, dispute arbitration, or equity grants.",
    icon: UserCheck,
    phase: "Q1 2027",
    status: "COMING SOON",
  },
  {
    title: "Custom Clause Presets",
    description:
      "Save your bespoke business terms, preferred late fee penalties, and tailored IP retention clauses across all generated documents.",
    icon: SlidersHorizontal,
    phase: "Q1 2027",
    status: "COMING SOON",
  },
];

export function EcosystemRoadmap() {
  return (
    <section className="py-24 sm:py-32 bg-white text-[#171717] border-b border-[#E7E5E2]">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F7F7F5] border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Product Roadmap
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              The evolving legal-tech ecosystem.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              LegaLese is expanding beyond automated drafting and audits to build a collaborative legal workspace for growing businesses.
            </p>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {roadmapItems.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.title}
                className="rounded-2xl border border-[#E7E5E2] bg-[#F7F7F5]/60 p-6 sm:p-7 space-y-5 card-hover shadow-2xs"
              >
                <div className="flex items-center justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-[#171717] border border-[#E7E5E2] shadow-2xs">
                    <Icon className="w-5 h-5 text-[#059669]" />
                  </div>
                  <span className="text-[10px] font-mono font-semibold uppercase tracking-wider px-2 py-0.5 rounded border border-[#E7E5E2] bg-white text-[#5F6368]">
                    {item.status}
                  </span>
                </div>

                <div className="space-y-2">
                  <h3 className="text-base font-semibold text-[#171717]">
                    {item.title}
                  </h3>
                  <p className="text-xs sm:text-[13px] text-[#5F6368] leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="pt-3 border-t border-[#E7E5E2] flex items-center justify-between text-xs text-[#8A8F98] font-mono">
                  <span>Target: {item.phase}</span>
                  <span className="text-[#059669] flex items-center gap-1 font-sans font-medium text-[11px]">
                    <Sparkles className="w-3 h-3" /> In Development
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
