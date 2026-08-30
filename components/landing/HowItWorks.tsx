"use client";

import React, { useState } from "react";
import { PenTool, Search, CheckSquare, Download, FileText, Sparkles } from "lucide-react";
import { ScrollReveal } from "@/components/ui/ScrollReveal";

const workflowStages = [
  {
    step: "01",
    id: "create",
    label: "Create",
    title: "Guided Questionnaire",
    description: "Answer basic project questions. LegaLese compiles enforceable clauses customized to your jurisdiction.",
    icon: PenTool,
    previewBadge: "Step 1 • Draft Created",
    previewSnippet: "Client: Acme Tech • Fee: ₹75,000 • Retainer: 50% upfront",
    statusTag: "Draft Ready",
  },
  {
    step: "02",
    id: "understand",
    label: "Understand",
    title: "Plain-Language Translation",
    description: "Every sentence of boilerplate is translated into simple, transparent English so you know what you're signing.",
    icon: Search,
    previewBadge: "Step 2 • Plain English",
    previewSnippet: "Translation: You keep ownership of work until the final balance is fully paid.",
    statusTag: "Zero Jargon",
  },
  {
    step: "03",
    id: "review",
    label: "Review",
    title: "Pre-Sign Risk Audit",
    description: "Our engine scans for uncapped liability, unfair termination clauses, and ambiguous revision cycles.",
    icon: CheckSquare,
    previewBadge: "Step 3 • Audited",
    previewSnippet: "Fairness Score: 88/100 • 2 minor clauses flagged for counter-proposal",
    statusTag: "Audited",
  },
  {
    step: "04",
    id: "download",
    label: "Download",
    title: "Export & Execute",
    description: "Download court-ready PDF or editable DOCX formats ready for execution and client delivery.",
    icon: Download,
    previewBadge: "Step 4 • Ready to Sign",
    previewSnippet: "PDFKit & DOCX Compiled • Legally formatted with execution signature blocks",
    statusTag: "Execution Ready",
  },
];

export function HowItWorks() {
  const [activeStage, setActiveStage] = useState(workflowStages[0]);

  return (
    <section id="workflow" className="py-24 sm:py-32 bg-[#F7F7F5] text-[#171717] border-b border-[#E7E5E2]">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 space-y-12 sm:space-y-16">
        <ScrollReveal>
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E2] text-[12px] font-semibold tracking-wider text-[#059669] uppercase shadow-2xs">
              Continuous Workflow
            </div>
            <h2 className="heading-landing-section text-[#171717] tracking-tight">
              From blank page to signed contract in minutes.
            </h2>
            <p className="text-[16px] sm:text-[17px] text-[#5F6368] leading-[1.6]">
              Four seamless steps that replace costly attorney retainers with fast, reliable contract certainty.
            </p>
          </div>
        </ScrollReveal>

        {/* Interactive Step Selector Bar */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          {workflowStages.map((stage) => {
            const isSelected = activeStage.id === stage.id;
            const Icon = stage.icon;
            return (
              <button
                key={stage.id}
                type="button"
                suppressHydrationWarning
                onClick={() => setActiveStage(stage)}
                className={`p-4 sm:p-5 rounded-xl border text-left transition-all ${
                  isSelected
                    ? "bg-white border-[#171717] ring-1 ring-[#171717]/10 shadow-sm"
                    : "bg-white/60 border-[#E7E5E2] hover:bg-white hover:border-[#D4D2CD]"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-lg border ${
                      isSelected
                        ? "bg-[#171717] text-white border-[#171717]"
                        : "bg-[#F7F7F5] text-[#5F6368] border-[#E7E5E2]"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="font-mono text-xs font-bold text-[#8A8F98]">
                    {stage.step}
                  </span>
                </div>

                <div className="space-y-1">
                  <div className="text-xs font-mono font-semibold uppercase tracking-wider text-[#059669]">
                    {stage.label}
                  </div>
                  <h4 className="text-sm font-semibold text-[#171717]">
                    {stage.title}
                  </h4>
                </div>
              </button>
            );
          })}
        </div>

        {/* Visual Transforming Contract Artifact Canvas */}
        <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-10 shadow-sm space-y-6">
          <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-4 text-xs">
            <div className="flex items-center gap-2 text-[#171717] font-semibold">
              <FileText className="w-4 h-4 text-[#059669]" />
              <span>{activeStage.previewBadge}</span>
            </div>
            <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-[#F0FDF4] text-[#166534] border border-[#BBF7D0]">
              {activeStage.statusTag}
            </span>
          </div>

          <div className="p-5 sm:p-6 rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] space-y-3">
            <h5 className="font-semibold text-sm text-[#171717]">
              {activeStage.title}
            </h5>
            <p className="text-xs sm:text-sm text-[#5F6368] leading-relaxed">
              {activeStage.description}
            </p>
            <div className="p-3 bg-white rounded-lg border border-[#E7E5E2] font-mono text-xs text-[#171717] flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-[#059669] shrink-0" />
              <span className="truncate">{activeStage.previewSnippet}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
