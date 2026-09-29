"use client";

import { useEffect, useRef } from "react";
import { Scale, ArrowRight } from "lucide-react";
import Link from "next/link";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function TrustSection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const section = sectionRef.current;

    if (!section) return;

    const ctx = gsap.context(() => {
      gsap.from(".trust-header", {
        opacity: 0,
        y: 35,
        duration: 0.7,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      gsap.from(".trust-card", {
        opacity: 0,
        y: 40,
        duration: 0.6,
        stagger: 0.15,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 70%",
          toggleActions: "play none none reverse",
        },
      });

      gsap.from(".trust-briefing", {
        opacity: 0,
        y: 25,
        duration: 0.7,
        delay: 0.2,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 60%",
          toggleActions: "play none none reverse",
        },
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="py-20 bg-white border-b border-[#E5E5E3]"
    >
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">

        {/* Header */}
        <div className="trust-header text-center space-y-3 max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F9F9F8] border border-[#E5E5E3] text-xs font-semibold text-[#111827]">
            <Scale className="w-3.5 h-3.5 text-[#059669]" />
            <span>The Trust Layer</span>
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827]">
            AI handles automation. Humans provide judgment.
          </h2>

          <p className="text-sm text-[#4B5563]">
            LegaLese identifies when a deal warrants human legal counsel and automatically compiles a complete briefing package so your attorney starts with full context.
          </p>
        </div>

        {/* Trust Tiers */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          <div className="trust-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
              Tier 1
            </span>

            <h3 className="text-base font-bold text-[#111827]">
              AI Assistance
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              Standard clauses, plain-English translations, and routine drafting tasks are handled instantly by AI.
            </p>
          </div>

          <div className="trust-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              Tier 2
            </span>

            <h3 className="text-base font-bold text-[#111827]">
              Professional Review
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              Flagged high-exposure terms trigger a clear recommendation for independent legal counsel review.
            </p>
          </div>

          <div className="trust-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
              Tier 3
            </span>

            <h3 className="text-base font-bold text-[#111827]">
              Formal Handoff
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              One-click briefing compilation generates intent, document, AI findings, and specific questions for counsel.
            </p>
          </div>

        </div>

        {/* Briefing Package */}
        <div className="trust-briefing p-6 rounded-2xl bg-[#FAFAF9] border border-[#E5E5E3] flex flex-col sm:flex-row items-center justify-between gap-4">

          <div className="space-y-1 text-center sm:text-left">
            <p className="text-sm font-bold text-[#111827]">
              Briefing Package Prototype
            </p>

            <p className="text-xs text-[#6B7280]">
              Eliminates the hours spent re-explaining contract requirements to external counsel.
            </p>
          </div>

          <Link
            href="/dashboard"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#111827] text-white text-xs font-semibold hover:bg-[#1F2937] transition-all shrink-0"
          >
            <span>Explore Trust Experience</span>
            <ArrowRight className="w-3.5 h-3.5 text-[#10B981]" />
          </Link>

        </div>

      </div>
    </section>
  );
}