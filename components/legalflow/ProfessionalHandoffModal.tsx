"use client";

import { useState, useEffect } from "react";
import {
  X,
  ShieldCheck,
  Scale,
  AlertTriangle,
  HelpCircle,
  Copy,
  Check,
  Calendar,
  Sparkles,
  ArrowRight,
  UserCheck,
} from "lucide-react";
import { DEMO_VERIFIED_PROFESSIONAL } from "@/lib/legalflow/professionals-mock";
import type { LegalCalendarEvent, LegalFlowRow } from "@/types/legalflow";

interface ProfessionalHandoffModalProps {
  isOpen: boolean;
  onClose: () => void;
  flow: LegalFlowRow;
  connectedDocTitle?: string;
  obligations?: LegalCalendarEvent[];
  findings?: Array<{
    category: string;
    risk_level: string;
    explanation: string;
    questions?: string[];
  }>;
}

export function ProfessionalHandoffModal({
  isOpen,
  onClose,
  flow,
  connectedDocTitle = "Document Draft",
  obligations = [],
  findings = [],
}: ProfessionalHandoffModalProps) {
  const selectedProf = DEMO_VERIFIED_PROFESSIONAL;
  const [copied, setCopied] = useState(false);
  const [packageShared, setPackageShared] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Derive relevant review questions from findings or scenario defaults
  const reviewQuestions: string[] = [];
  if (findings.length > 0) {
    findings.forEach((f) => {
      if (f.questions && f.questions.length > 0) {
        reviewQuestions.push(...f.questions.slice(0, 1));
      } else {
        reviewQuestions.push(`Is the ${f.category} clause appropriately balanced?`);
      }
    });
  } else if (flow.scenario_key === "hire_freelancer") {
    reviewQuestions.push(
      "Is the intellectual property assignment clause broad enough to cover deliverables?",
      "Are liability limits and indemnities mutually fair?",
      "Does the termination notice period align with commercial norms?"
    );
  } else if (flow.scenario_key === "create_nda") {
    reviewQuestions.push(
      "Does the definition of confidential information contain standard commercial exclusions?",
      "Is the confidentiality duration appropriate for this jurisdiction?"
    );
  } else {
    reviewQuestions.push(
      "Are there uncapped liability provisions or one-sided indemnity traps?",
      "Are payment milestones and dispute jurisdiction enforceable?"
    );
  }

  const handleCopyPackage = () => {
    const textSummary = `LEGALESE REVIEW PACKAGE\n--------------------------------\nGoal: ${flow.title}\nIntent: "${flow.intent_query}"\nDocument: ${connectedDocTitle}\n\nAI PREPARATION SUMMARY:\n${findings.map((f, i) => `${i + 1}. [${f.risk_level.toUpperCase()}] ${f.category}: ${f.explanation}`).join("\n") || "AI drafting and standard clause breakdown completed."}\n\nOBLIGATIONS/DEADLINES:\n${obligations.map((o, i) => `${i + 1}. ${o.title} (${o.date})`).join("\n") || "Standard commercial delivery milestones."}\n\nKEY QUESTIONS FOR COUNSEL:\n${reviewQuestions.map((q, i) => `${i + 1}. ${q}`).join("\n")}\n\n[Prepared via LegaLese Context-Preserving LegalFlow]`;

    navigator.clipboard.writeText(textSummary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl glass-modal p-6 sm:p-8 space-y-6 animate-scale-in border border-[#E7E5E2] shadow-xl">
        {/* Close button */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-lg text-[#8A8F98] hover:text-[#171717] hover:bg-[#F7F7F5] transition-colors cursor-pointer"
          aria-label="Close dialog"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Prototype Banner */}
        <div className="rounded-xl bg-[#FEF3C7] border border-[#FDE68A] p-3 text-xs text-[#92400E] flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Scale className="w-4 h-4 text-[#B45309] shrink-0" />
            <span className="font-semibold">
              Verification Prototype — Demonstration Data Only
            </span>
          </div>
          <span className="rounded bg-white/80 px-2 py-0.5 text-[10px] font-mono font-bold text-[#B45309]">
            DEMO ONLY
          </span>
        </div>

        {/* Header */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#059669] font-bold">
              Context-Preserving Handoff
            </span>
          </div>
          <h2 id="modal-title" className="text-xl font-bold text-[#171717]">
            Request Professional Review Package
          </h2>
          <p className="text-xs sm:text-[13px] text-[#5F6368] leading-relaxed">
            LegaLese prepares the goal, document clauses, and AI findings so the human professional can focus on what actually requires legal judgment.
          </p>
        </div>

        {/* Visual Workflow Handoff Diagram */}
        <div className="rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-3.5">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-[#5F6368] gap-1 overflow-x-auto">
            <span className="text-[#171717]">1. User Intent</span>
            <ArrowRight className="w-3 h-3 text-[#8A8F98]" />
            <span className="text-[#171717]">2. AI Preparation</span>
            <ArrowRight className="w-3 h-3 text-[#8A8F98]" />
            <span className="text-[#059669] font-bold">3. Professional Review</span>
            <ArrowRight className="w-3 h-3 text-[#8A8F98]" />
            <span className="text-[#171717]">4. Protected User</span>
          </div>
        </div>

        {/* Demo Verified Professional Card */}
        <div className="rounded-xl border border-[#E7E5E2] bg-white p-4 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98]">
              Assigned Demonstration Counsel
            </p>
            <span className="text-[10px] font-mono text-[#059669] font-semibold bg-[#ECFDF5] px-2 py-0.5 rounded border border-[#A7F3D0]">
              Independent Reviewer
            </span>
          </div>

          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#171717] text-white font-bold text-sm shrink-0">
              {selectedProf.name.slice(0, 2).toUpperCase()}
            </div>
            <div className="space-y-0.5 min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <h4 className="text-sm font-bold text-[#171717]">{selectedProf.name}</h4>
                <span className="inline-flex items-center gap-1 text-[10px] font-medium text-[#059669] bg-[#059669]/10 rounded-md px-1.5 py-0.2">
                  <UserCheck className="w-3 h-3" />
                  <span>Credentials Verified [Mock]</span>
                </span>
              </div>
              <p className="text-xs text-[#5F6368]">{selectedProf.title}</p>
              <p className="text-[11px] font-mono text-[#8A8F98]">
                Reg: {selectedProf.barRegistration}
              </p>
            </div>
          </div>

          {/* Practice Areas */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {selectedProf.practiceAreas.map((area) => (
              <span
                key={area}
                className="rounded bg-[#F7F7F5] border border-[#E7E5E2] px-2 py-0.5 text-[10px] text-[#5F6368]"
              >
                {area}
              </span>
            ))}
          </div>
        </div>

        {/* Prepared Briefing Package Preview */}
        <div className="rounded-xl border border-[#E7E5E2] bg-white p-4 space-y-3 text-xs">
          <div className="flex items-center justify-between border-b border-[#F0EFEA] pb-2">
            <span className="font-bold text-[#171717] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#059669]" />
              <span>Prepared Briefing Package</span>
            </span>
            <span className="text-[11px] font-mono text-[#8A8F98]">Auto-Compiled</span>
          </div>

          <div className="grid sm:grid-cols-2 gap-3 text-[11px]">
            <div className="rounded-lg bg-[#F7F7F5] p-2.5 space-y-1">
              <span className="font-mono text-[#8A8F98] uppercase">LegalFlow</span>
              <p className="font-semibold text-[#171717] truncate">{flow.title}</p>
            </div>
            <div className="rounded-lg bg-[#F7F7F5] p-2.5 space-y-1">
              <span className="font-mono text-[#8A8F98] uppercase">Document</span>
              <p className="font-semibold text-[#171717] truncate">{connectedDocTitle}</p>
            </div>
          </div>

          {/* AI Pre-Audit Findings */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#8A8F98] font-semibold flex items-center gap-1">
              <AlertTriangle className="w-3 h-3 text-[#B45309]" />
              <span>Flagged Items for Professional Evaluation:</span>
            </span>
            <ul className="space-y-1 text-xs text-[#5F6368]">
              {findings.length > 0 ? (
                findings.slice(0, 3).map((f, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-[#B45309] font-bold">•</span>
                    <span>
                      <strong className="text-[#171717]">{f.category}:</strong> {f.explanation}
                    </span>
                  </li>
                ))
              ) : (
                <>
                  <li className="flex items-start gap-1.5">
                    <span className="text-[#059669] font-bold">•</span>
                    <span>Liability limits and indemnification boundary review</span>
                  </li>
                  <li className="flex items-start gap-1.5">
                    <span className="text-[#059669] font-bold">•</span>
                    <span>Intellectual property assignment and preexisting work ownership</span>
                  </li>
                </>
              )}
            </ul>
          </div>

          {/* Extracted Obligations if present */}
          {obligations.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[11px] font-mono uppercase tracking-wider text-[#8A8F98] font-semibold flex items-center gap-1">
                <Calendar className="w-3 h-3 text-[#059669]" />
                <span>Extracted Deadlines &amp; Payment Obligations:</span>
              </span>
              <ul className="space-y-1 text-xs text-[#5F6368]">
                {obligations.slice(0, 2).map((o, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-[#059669] font-bold">•</span>
                    <span>
                      {o.title} — <strong className="text-[#171717]">{o.date}</strong>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Questions */}
          <div className="space-y-1.5 pt-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#8A8F98] font-semibold flex items-center gap-1">
              <HelpCircle className="w-3 h-3 text-[#059669]" />
              <span>Pre-Formulated Questions for Attorney:</span>
            </span>
            <ul className="space-y-1 text-xs text-[#5F6368]">
              {reviewQuestions.slice(0, 3).map((q, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <span className="text-[#059669] font-bold">{i + 1}.</span>
                  <span>{q}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 border-t border-[#E7E5E2]">
          <button
            type="button"
            onClick={handleCopyPackage}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-[#E7E5E2] bg-white px-4 py-2.5 text-xs font-semibold text-[#171717] hover:bg-[#F7F7F5] transition-all shadow-2xs cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-[#059669]" />
                <span>Package Copied to Clipboard</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-[#8A8F98]" />
                <span>Copy Briefing Summary</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={() => setPackageShared(true)}
            disabled={packageShared}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#171717] px-5 py-2.5 text-xs font-semibold text-white hover:bg-[#262626] transition-all shadow-xs cursor-pointer disabled:bg-[#059669]"
          >
            {packageShared ? (
              <>
                <Check className="w-3.5 h-3.5 text-white" />
                <span>Review Request Prepared [Prototype]</span>
              </>
            ) : (
              <>
                <ShieldCheck className="w-3.5 h-3.5 text-[#059669]" />
                <span>Confirm Review Package</span>
              </>
            )}
          </button>
        </div>

        {packageShared && (
          <div className="p-3.5 rounded-xl bg-[#F0FDF4] border border-[#BBF7D0] text-xs text-[#166534] space-y-1 animate-fade-in">
            <p className="font-semibold flex items-center gap-1.5">
              <Check className="w-4 h-4 text-[#059669]" />
              <span>Context Briefing Package Ready for Professional Review</span>
            </p>
            <p className="text-[11px] text-[#15803D]">
              In the live deployment, this package is securely dispatched directly to your retained counsel with zero re-explanation overhead.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
