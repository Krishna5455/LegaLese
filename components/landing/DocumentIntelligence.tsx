"use client";

import { useEffect, useRef } from "react";
import { Sparkles, CheckCircle2 } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function DocumentIntelligence() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const section = sectionRef.current;

    if (!section) return;

    const ctx = gsap.context(() => {
      gsap.from(".intelligence-content", {
        opacity: 0,
        x: -50,
        duration: 0.8,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      gsap.from(".intelligence-card", {
        opacity: 0,
        x: 50,
        duration: 0.8,
        delay: 0.15,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      gsap.from(".intelligence-feature", {
        opacity: 0,
        y: 15,
        duration: 0.5,
        stagger: 0.12,
        ease: "power2.out",
        scrollTrigger: {
          trigger: section,
          start: "top 75%",
          toggleActions: "play none none reverse",
        },
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="intelligence"
      className="py-20 bg-[#F9F9F8] border-b border-[#E5E5E3]"
    >
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">

          {/* Left Column: Intelligence Value */}
          <div className="intelligence-content lg:col-span-6 space-y-5">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E5E5E3] text-xs font-semibold text-[#059669]">
              <Sparkles className="w-3.5 h-3.5" />
              <span>LegaLese Intelligence</span>
            </div>

            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827] leading-snug">
              Understand every clause before you commit.
            </h2>

            <p className="text-sm text-[#4B5563] leading-relaxed">
              Our AI engine audits your contract to detect asymmetric liability, unfair payment terms, and vague milestone criteria. It doesn&apos;t just summarize; it generates actionable negotiation points.
            </p>

            <ul className="space-y-2.5 text-xs text-[#374151]">
              <li className="intelligence-feature flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>
                  Plain-English translation for complex legal terminology
                </span>
              </li>

              <li className="intelligence-feature flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>
                  Automated risk scoring (Low, Moderate, High Exposure)
                </span>
              </li>

              <li className="intelligence-feature flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>
                  Bilateral IP transfer and deliverable acceptance verification
                </span>
              </li>
            </ul>
          </div>

          {/* Right Column: Interactive Card Preview */}
          <div className="intelligence-card lg:col-span-6">
            <div className="p-6 rounded-2xl bg-white border border-[#E5E5E3] shadow-sm space-y-4">

              <div className="flex items-center justify-between pb-3 border-b border-[#E5E5E3]">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#111827]">
                  Pre-Sign Risk Audit
                </span>

                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                  Risk Level: Low
                </span>
              </div>

              <div className="space-y-3 text-xs">

                <div className="p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#111827]">
                      Intellectual Property Clause
                    </span>

                    <span className="text-[10px] text-[#059669] font-medium font-mono">
                      Standard
                    </span>
                  </div>

                  <p className="text-[#6B7280] text-[11px]">
                    Full copyright and source code transfer directly upon receipt of final milestone payment.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#111827]">
                      Liability & Indemnity Limitation
                    </span>

                    <span className="text-[10px] text-[#059669] font-medium font-mono">
                      Capped
                    </span>
                  </div>

                  <p className="text-[#6B7280] text-[11px]">
                    Aggregate liability is strictly capped at 100% of total compensation paid under the agreement.
                  </p>
                </div>

              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}