"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function FinalCta() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const section = sectionRef.current;

    if (!section) return;

    const ctx = gsap.context(() => {
      // Subtle background glow
      gsap.from(".cta-glow", {
        opacity: 0,
        scale: 0.7,
        duration: 1.4,
        ease: "power2.out",
        scrollTrigger: {
          trigger: section,
          start: "top 85%",
          toggleActions: "play none none reverse",
        },
      });

      // Badge
      gsap.from(".cta-badge", {
        opacity: 0,
        y: -20,
        duration: 0.6,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      // Main heading
      gsap.from(".cta-heading", {
        opacity: 0,
        y: 40,
        duration: 0.8,
        delay: 0.15,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      // Description
      gsap.from(".cta-description", {
        opacity: 0,
        y: 25,
        duration: 0.6,
        delay: 0.3,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 75%",
          toggleActions: "play none none reverse",
        },
      });

      // Buttons
      gsap.from(".cta-buttons", {
        opacity: 0,
        y: 25,
        scale: 0.97,
        duration: 0.7,
        delay: 0.4,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 70%",
          toggleActions: "play none none reverse",
        },
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="py-24 bg-[#111827] text-white border-b border-[#1F2937] relative overflow-hidden"
    >
      {/* Decorative background glow */}
      <div className="cta-glow absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] rounded-full bg-[#059669]/10 blur-3xl pointer-events-none" />

      <div className="mx-auto max-w-4xl px-4 sm:px-6 text-center space-y-6 relative z-10">

        {/* Badge */}
        <div className="cta-badge inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-xs font-semibold text-[#10B981]">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Complete LegalFlow</span>
        </div>

        {/* Heading */}
        <h2 className="cta-heading text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white max-w-2xl mx-auto leading-tight">
          Don&apos;t just generate the legal document. <br />
          <span className="text-[#10B981]">
            Complete the legal journey.
          </span>
        </h2>

        {/* Description */}
        <p className="cta-description text-sm sm:text-base text-white/70 max-w-xl mx-auto leading-relaxed">
          From natural language intent to drafted agreements, AI risk auditing, and automated milestone tracking.
        </p>

        {/* Buttons */}
        <div className="cta-buttons pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
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