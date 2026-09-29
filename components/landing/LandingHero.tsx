"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowRight, Upload, Sparkles } from "lucide-react";
import gsap from "gsap";

export function LandingHero() {
  const heroRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const hero = heroRef.current;

    if (!hero) return;

    const ctx = gsap.context(() => {
      const timeline = gsap.timeline({
        defaults: {
          ease: "power3.out",
        },
      });

      timeline
        .from(".hero-pill", {
          opacity: 0,
          y: -20,
          duration: 0.6,
        })
        .from(
          ".hero-title",
          {
            opacity: 0,
            y: 30,
            duration: 0.8,
          },
          "-=0.3"
        )
        .from(
          ".hero-description",
          {
            opacity: 0,
            y: 20,
            duration: 0.6,
          },
          "-=0.4"
        )
        .from(
          ".hero-buttons",
          {
            opacity: 0,
            y: 20,
            duration: 0.6,
          },
          "-=0.3"
        )
        .from(
          ".hero-journey",
          {
            opacity: 0,
            y: 40,
            scale: 0.98,
            duration: 0.8,
          },
          "-=0.2"
        )
        .from(
          ".hero-stage",
          {
            opacity: 0,
            y: 20,
            duration: 0.5,
            stagger: 0.08,
          },
          "-=0.3"
        );
    }, hero);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={heroRef}
      className="relative pt-28 pb-16 overflow-hidden bg-[#F9F9F8] border-b border-[#E5E5E3]"
    >
      <div className="mx-auto max-w-5xl px-4 sm:px-6 text-center space-y-6">

        {/* Category Pill */}
        <div className="hero-pill inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E5E5E3] text-xs font-semibold text-[#059669] shadow-2xs">
          <Sparkles className="w-3.5 h-3.5 text-[#059669]" />
          <span>AI-Powered LegalFlow Platform</span>
        </div>

        {/* Hero Headline */}
        <h1 className="hero-title text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-[#111827] max-w-3xl mx-auto leading-[1.12]">
          Your legal task. <br />
          <span className="text-[#059669]">One guided journey.</span>
        </h1>

        {/* Subheadline */}
        <p className="hero-description text-base sm:text-lg text-[#4B5563] max-w-2xl mx-auto leading-relaxed">
          From understanding a document to preparing it, reviewing risk, getting professional guidance when needed, and tracking what happens next.
        </p>

        {/* Primary CTAs */}
        <div className="hero-buttons flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
          <Link
            href="/dashboard"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-[#111827] px-6 py-3 text-sm font-semibold text-white hover:bg-[#1F2937] transition-all shadow-sm active:scale-98"
          >
            <span>Start a LegalFlow</span>
            <ArrowRight className="w-4 h-4 text-[#10B981]" />
          </Link>

          <Link
            href="/dashboard#upload"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-[#E5E5E3] bg-white px-6 py-3 text-sm font-semibold text-[#111827] hover:bg-[#F3F4F6] transition-all shadow-2xs active:scale-98"
          >
            <Upload className="w-4 h-4 text-[#6B7280]" />
            <span>Analyze a document</span>
          </Link>
        </div>

        {/* Canonical Journey Interactive Demo Showcase */}
        <div id="journey" className="hero-journey pt-10 max-w-4xl mx-auto text-left">
          <div className="rounded-2xl bg-white border border-[#E5E5E3] shadow-md p-5 sm:p-7 space-y-6">

            {/* Header: User Intent Input */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E5E5E3]">
              <div className="flex items-center gap-2.5">
                <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#059669] bg-[#ECFDF5] px-2 py-0.5 rounded border border-[#A7F3D0]">
                  User Intent
                </span>

                <p className="text-sm font-semibold text-[#111827]">
                  &quot;I want to hire a freelancer for my website.&quot;
                </p>
              </div>

              <span className="text-xs text-[#6B7280] font-medium">
                Canonical LegalFlow Journey
              </span>
            </div>

            {/* 6-Stage Progression Flow */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-center text-xs">

              <div className="hero-stage p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 1
                </span>
                <p className="font-semibold text-[#111827]">Understand</p>
                <p className="text-[11px] text-[#6B7280]">IP Scope & Roles</p>
              </div>

              <div className="hero-stage p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 2
                </span>
                <p className="font-semibold text-[#111827]">Prepare</p>
                <p className="text-[11px] text-[#6B7280]">Freelance Contract</p>
              </div>

              <div className="hero-stage p-3 rounded-xl bg-[#ECFDF5] border border-[#A7F3D0] space-y-1 shadow-2xs">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 3
                </span>
                <p className="font-semibold text-[#111827]">Review</p>
                <p className="text-[11px] text-[#065F46] font-medium">
                  AI Risk Audit
                </p>
              </div>

              <div className="hero-stage p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 4
                </span>
                <p className="font-semibold text-[#111827]">Trust</p>
                <p className="text-[11px] text-[#6B7280]">Attorney Briefing</p>
              </div>

              <div className="hero-stage p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 5
                </span>
                <p className="font-semibold text-[#111827]">Track</p>
                <p className="text-[11px] text-[#6B7280]">Milestones & Pay</p>
              </div>

              <div className="hero-stage p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-1">
                <span className="font-mono text-[10px] text-[#059669] font-bold uppercase">
                  Stage 6
                </span>
                <p className="font-semibold text-[#111827]">Complete</p>
                <p className="text-[11px] text-[#6B7280]">Document Archive</p>
              </div>

            </div>

            {/* Dominant Next Action Snippet */}
            <div className="p-4 rounded-xl bg-[#FAFAF9] border border-[#E5E5E3] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">

              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-[#10B981] animate-pulse" />

                  <span className="text-xs font-bold uppercase tracking-wider text-[#111827]">
                    Current Step: Review Liability & IP Transfer
                  </span>
                </div>

                <p className="text-xs text-[#4B5563]">
                  AI detected a standard 100% fee liability cap and bilateral IP transfer upon final invoice payment.
                </p>
              </div>

              <Link
                href="/dashboard"
                className="shrink-0 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#111827] text-white text-xs font-semibold hover:bg-[#1F2937] transition-colors"
              >
                <span>Open Workspace</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#10B981]" />
              </Link>

            </div>
          </div>
        </div>
      </div>
    </section>
  );
}