"use client";

import { useEffect, useRef } from "react";
import { Lock, ShieldCheck, Database, Key } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function SecuritySection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const section = sectionRef.current;

    if (!section) return;

    const ctx = gsap.context(() => {
      // Header animation
      gsap.from(".security-header", {
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

      // Security cards animation
      gsap.from(".security-card", {
        opacity: 0,
        y: 45,
        duration: 0.7,
        stagger: 0.15,
        ease: "power3.out",
        scrollTrigger: {
          trigger: section,
          start: "top 70%",
          toggleActions: "play none none reverse",
        },
      });

      // Icons slightly scale into place
      gsap.from(".security-icon", {
        opacity: 0,
        scale: 0.7,
        duration: 0.5,
        stagger: 0.15,
        ease: "back.out(1.7)",
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
      id="security"
      className="py-20 bg-white border-b border-[#E5E5E3]"
    >
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">

        {/* Header */}
        <div className="security-header text-center space-y-3 max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F9F9F8] border border-[#E5E5E3] text-xs font-semibold text-[#111827]">
            <Lock className="w-3.5 h-3.5 text-[#059669]" />
            <span>Security & Privacy</span>
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827]">
            Enterprise-grade data protection
          </h2>

          <p className="text-sm text-[#4B5563]">
            Your legal documents contain sensitive commercial agreements. We ensure strict tenant isolation and zero public model training.
          </p>
        </div>

        {/* Security Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* Card 1 */}
          <div className="security-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="security-icon p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <ShieldCheck className="w-5 h-5" />
            </div>

            <h3 className="text-base font-bold text-[#111827]">
              Zero AI Training
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              Your uploaded contracts and generated clauses are never used to train public AI foundation models.
            </p>
          </div>

          {/* Card 2 */}
          <div className="security-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="security-icon p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <Database className="w-5 h-5" />
            </div>

            <h3 className="text-base font-bold text-[#111827]">
              Row-Level Security (RLS)
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              Postgres database enforced tenant isolation guarantees that only authorized users access their documents.
            </p>
          </div>

          {/* Card 3 */}
          <div className="security-card p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="security-icon p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <Key className="w-5 h-5" />
            </div>

            <h3 className="text-base font-bold text-[#111827]">
              Encrypted Storage
            </h3>

            <p className="text-xs text-[#4B5563] leading-relaxed">
              All contract binaries, extracted text tokens, and export artifacts are encrypted in transit and at rest.
            </p>
          </div>

        </div>
      </div>
    </section>
  );
}