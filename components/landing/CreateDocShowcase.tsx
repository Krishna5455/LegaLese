"use client";

import React, { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Sparkles, Download, FileText, Check } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

const presets = [
  { name: "Web Development", fee: "75,000", client: "Acme Technologies Pvt. Ltd.", deliverable: "Full-stack web application development & cloud deployment" },
  { name: "UI/UX Design", fee: "60,000", client: "Helix Design Studio", deliverable: "End-to-end design system & interactive mobile prototypes" },
  { name: "Digital Marketing", fee: "45,000", client: "Orbit Growth Partners", deliverable: "Quarterly performance marketing and paid growth strategy" },
];

export function CreateDocShowcase() {
  const [selectedPreset, setSelectedPreset] = useState(presets[0]);
  const [clientName, setClientName] = useState(presets[0].client);
  const [freelancerName, setFreelancerName] = useState("Rahul Sharma");
  const [fee, setFee] = useState(presets[0].fee);
  const [retainer, setRetainer] = useState("50%");

  const handleSelectPreset = (p: typeof presets[0]) => {
    setSelectedPreset(p);
    setClientName(p.client);
    setFee(p.fee);
  };

  return (
    <section id="create" className="py-24 sm:py-32 bg-[#F7F7F5] text-[#171717] border-b border-[#E7E5E2]">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Guided Contract Authoring
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              Create legal documents in minutes.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              No generic templates or confusing boilerplate. Answer clear guided questions to generate legally structured contracts with fair payment milestones and IP protection.
            </p>
          </div>
        </ScrollReveal>

        {/* Visual Workspace Mockup */}
        <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-10 shadow-sm grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left: Interactive Form Controls */}
          <div className="lg:col-span-6 space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[#8A8F98]">
                  Industry Presets
                </span>
                <span className="text-[11px] text-[#059669] font-medium flex items-center gap-1">
                  <Sparkles className="w-3 h-3" /> One-click autofill
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    suppressHydrationWarning
                    onClick={() => handleSelectPreset(preset)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                      selectedPreset.name === preset.name
                        ? "bg-[#171717] text-white border-[#171717] font-semibold"
                        : "bg-[#F7F7F5] text-[#5F6368] border-[#E7E5E2] hover:border-[#D4D2CD]"
                    }`}
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-[#171717] block mb-1">
                    Client Entity
                  </label>
                  <input
                    type="text"
                    suppressHydrationWarning
                    value={clientName}
                    onChange={(e) => setClientName(e.target.value)}
                    className="w-full rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] px-3 py-2 text-xs sm:text-sm text-[#171717] input-focus"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-[#171717] block mb-1">
                    Freelancer / Studio
                  </label>
                  <input
                    type="text"
                    suppressHydrationWarning
                    value={freelancerName}
                    onChange={(e) => setFreelancerName(e.target.value)}
                    className="w-full rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] px-3 py-2 text-xs sm:text-sm text-[#171717] input-focus"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-[#171717] block mb-1">
                    Total Compensation
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-2 text-xs font-bold text-[#8A8F98]">₹</span>
                    <input
                      type="text"
                      suppressHydrationWarning
                      value={fee}
                      onChange={(e) => setFee(e.target.value)}
                      className="w-full rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] pl-7 pr-3 py-2 text-xs sm:text-sm font-semibold text-[#171717] input-focus"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-[#171717] block mb-1">
                    Upfront Retainer Deposit
                  </label>
                  <div className="grid grid-cols-3 gap-1.5">
                    {["25%", "50%", "100%"].map((pct) => (
                      <button
                        key={pct}
                        type="button"
                        suppressHydrationWarning
                        onClick={() => setRetainer(pct)}
                        className={`py-2 rounded-lg text-center text-xs font-medium border transition-colors ${
                          retainer === pct
                            ? "bg-[#059669] text-white border-[#059669]"
                            : "bg-[#F7F7F5] text-[#5F6368] border-[#E7E5E2] hover:border-[#D4D2CD]"
                        }`}
                      >
                        {pct}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-2">
              <Link
                href="/dashboard/create"
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-[#171717] px-6 py-3 text-sm font-medium text-white hover:bg-[#262626] transition-all shadow-xs active:scale-98"
              >
                <span>Draft custom agreement now</span>
                <ArrowRight className="w-4 h-4 text-[#059669]" />
              </Link>
            </div>
          </div>

          {/* Right: Live Generated Agreement Output Preview */}
          <div className="lg:col-span-6 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5] p-5 sm:p-7 space-y-5 shadow-2xs">
            <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-3">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#8A8F98] flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-[#059669]" />
                Generated Agreement Preview
              </span>
              <span className="text-[11px] font-medium text-[#166534] bg-[#F0FDF4] border border-[#BBF7D0] px-2 py-0.5 rounded">
                ✓ Ready to Export
              </span>
            </div>

            <div className="space-y-4 bg-white p-5 rounded-xl border border-[#E7E5E2] text-xs leading-relaxed text-[#5F6368] shadow-2xs">
              <div className="border-b border-[#E7E5E2] pb-3">
                <h4 className="font-bold text-sm text-[#171717] uppercase">
                  Freelance Service Agreement
                </h4>
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-[#8A8F98]">
                  <div>
                    <span className="block font-mono uppercase text-[9px]">Client</span>
                    <strong className="text-[#171717] font-semibold">{clientName || "Client"}</strong>
                  </div>
                  <div>
                    <span className="block font-mono uppercase text-[9px]">Freelancer</span>
                    <strong className="text-[#171717] font-semibold">{freelancerName || "Contractor"}</strong>
                  </div>
                </div>
              </div>

              <div className="space-y-1">
                <strong className="text-[#171717] block font-semibold">1. Scope of Work & Deliverables:</strong>
                <p>{selectedPreset.deliverable}.</p>
              </div>

              <div className="space-y-1">
                <strong className="text-[#171717] block font-semibold">2. Compensation & Payment Terms:</strong>
                <p>
                  Total compensation is <strong className="text-[#171717]">₹{fee} INR</strong>, structured with a {retainer} upfront retainer deposit and balance upon milestone completion.
                </p>
              </div>

              <div className="pt-2 flex items-center gap-2">
                <button
                  type="button"
                  suppressHydrationWarning
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] text-[11px] font-medium text-[#171717] hover:bg-white transition-colors"
                >
                  <Download className="w-3 h-3 text-[#059669]" />
                  <span>Download PDF</span>
                </button>
                <button
                  type="button"
                  suppressHydrationWarning
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] text-[11px] font-medium text-[#171717] hover:bg-white transition-colors"
                >
                  <Download className="w-3 h-3 text-[#059669]" />
                  <span>Download DOCX</span>
                </button>
              </div>

              {/* Real Contract Signing Execution Snapshot */}
              <div className="relative rounded-xl overflow-hidden border border-[#E7E5E2] h-28 group">
                <Image
                  src="/images/executive_contract_signing.jpg"
                  alt="Execution of legal agreement with executive fountain pen"
                  fill
                  sizes="(max-width: 1024px) 100vw, 450px"
                  className="object-cover object-center filter brightness-95 transition-transform duration-700 ease-out group-hover:scale-102"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/20 to-transparent flex items-end p-2.5">
                  <div className="text-white text-[11px] font-medium flex items-center justify-between w-full">
                    <span>Legally Enforceable Execution Block</span>
                    <span className="font-mono text-[10px] text-[#10B981] bg-black/40 px-2 py-0.5 rounded">Verified</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-[#8A8F98] pt-1">
              <span className="flex items-center gap-1 text-[#166534]">
                <Check className="w-3.5 h-3.5" /> Plain language verified
              </span>
              <span className="font-mono text-[11px]">Format: Legal Standard</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
