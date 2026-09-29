"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { ArrowLeft, Sparkles, RefreshCw } from "lucide-react";
import { updateLegalFlowStep } from "@/lib/actions/legalflow";
import { getFlowNextAction } from "@/lib/legalflow/intent-engine";
import { WhatNextCard } from "@/components/legalflow/WhatNextCard";
import { JourneyTimeline } from "@/components/legalflow/JourneyTimeline";
import { TrustSignalCard } from "@/components/legalflow/TrustSignalCard";
import { ConnectedDocCard } from "@/components/legalflow/ConnectedDocCard";
import { UpcomingObligationsCard } from "@/components/legalflow/UpcomingObligationsCard";
import { ProfessionalHandoffModal } from "@/components/legalflow/ProfessionalHandoffModal";
import type {
  FlowStepStatus,
  LegalCalendarEvent,
  LegalFlowRow,
  LegalFlowStep,
} from "@/types/legalflow";

interface LegalFlowWorkspaceProps {
  initialFlow: LegalFlowRow;
  connectedDoc?: {
    id: string;
    title: string;
    type: string;
    status: string;
    isGenerated?: boolean;
    url: string;
  } | null;
  obligations?: LegalCalendarEvent[];
  findings?: Array<{
    category: string;
    risk_level: string;
    explanation: string;
    questions?: string[];
  }>;
}

export function LegalFlowWorkspace({
  initialFlow,
  connectedDoc,
  obligations = [],
  findings = [],
}: LegalFlowWorkspaceProps) {
  const [flow, setFlow] = useState<LegalFlowRow>(initialFlow);
  const [isPending, startTransition] = useTransition();
  const [isHandoffModalOpen, setIsHandoffModalOpen] = useState(false);

  const { completedCount, totalCount, progressPercent, nextStep } =
    getFlowNextAction(flow.steps || []);

  const isComplete =
    flow.status === "completed" ||
    (totalCount > 0 && completedCount === totalCount);

  // Toggle step completion status
  const handleToggleStep = (stepId: string) => {
    const currentStep = (flow.steps || []).find((s) => s.id === stepId);
    if (!currentStep) return;

    const newStatus: FlowStepStatus =
      currentStep.status === "completed" ? "next" : "completed";

    // Optimistic state update
    const updatedSteps: LegalFlowStep[] = (flow.steps || []).map((s) => {
      if (s.id === stepId) {
        return { ...s, status: newStatus };
      }
      return s;
    });

    const nextAction = getFlowNextAction(updatedSteps);

    setFlow((prev) => ({
      ...prev,
      steps: updatedSteps,
      status:
        nextAction.completedCount === updatedSteps.length
          ? "completed"
          : prev.status,
    }));

    // Server-side persistence
    startTransition(async () => {
      const res = await updateLegalFlowStep(flow.id, stepId, newStatus);
      if (res.flow) {
        setFlow(res.flow);
      }
    });
  };

  const handleMarkStepComplete = (stepId: string) => {
    handleToggleStep(stepId);
  };

  const hasHighRiskFindings = findings.some(
    (f) => f.risk_level === "high" || f.risk_level === "medium"
  );

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Back Navigation & Status Ribbon */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-[#E7E5E2] pb-4">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#5F6368] hover:text-[#171717] transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Workspace</span>
        </Link>

        <div className="flex items-center gap-2">
          {isPending && (
            <span className="flex items-center gap-1 text-[11px] font-mono text-[#8A8F98]">
              <RefreshCw className="w-3 h-3 animate-spin text-[#059669]" />
              <span>Saving...</span>
            </span>
          )}

          <span
            className={`rounded-full px-3 py-0.5 text-xs font-semibold ${
              isComplete
                ? "bg-[#059669]/10 text-[#059669] border border-[#059669]/20"
                : "bg-[#171717] text-white"
            }`}
          >
            {isComplete ? "Journey Complete" : "In Progress"}
          </span>
        </div>
      </div>

      {/* Primary Workspace Header */}
      <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-8 space-y-5 shadow-xs">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-1.5 rounded-full bg-[#F7F7F5] border border-[#E7E5E2] px-2.5 py-0.5 text-[11px] font-medium text-[#5F6368]">
            <Sparkles className="w-3 h-3 text-[#059669]" />
            <span>Goal: &ldquo;{flow.intent_query}&rdquo;</span>
          </div>
          <h1 className="heading-page text-[#171717]">{flow.title}</h1>
          <p className="text-xs sm:text-[14px] text-[#5F6368]">
            Guided legal journey. Complete each phase to safely draft, audit, and track this matter.
          </p>
        </div>

        {/* Progress Summary Bar */}
        <div className="rounded-xl bg-[#F7F7F5] border border-[#E7E5E2] p-4 sm:p-5 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-[#171717]">
            <span className="flex items-center gap-2">
              <span className="capitalize">{flow.current_stage.replace("_", " ")}</span>
              <span className="text-[#8A8F98] font-normal">
                • {completedCount} of {totalCount} steps complete
              </span>
            </span>
            <span className="text-[#059669] font-mono font-bold">{progressPercent}%</span>
          </div>

          <div className="h-2 w-full overflow-hidden rounded-full bg-[#E7E5E2]">
            <div
              className="h-full rounded-full bg-[#059669] transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main 2-Column Responsive Grid */}
      <div className="grid gap-8 lg:grid-cols-3">
        {/* Left Column (2/3 width on desktop): What Next & Journey Timeline */}
        <div className="lg:col-span-2 space-y-8">
          {/* Primary "What Do I Do Next?" Section */}
          <section id="what-next">
            <WhatNextCard
              nextStep={nextStep}
              isComplete={isComplete}
              onMarkStepComplete={handleMarkStepComplete}
              connectedDocUrl={connectedDoc?.url}
            />
          </section>

          {/* Journey Steps Roadmap */}
          <section id="journey-steps">
            <JourneyTimeline
              steps={flow.steps || []}
              onToggleStep={handleToggleStep}
            />
          </section>
        </div>

        {/* Right Column (1/3 width on desktop): Trust, Connected Document, Obligations */}
        <div className="space-y-6">
          {/* Trust Signal Card with Expandable Rationale and Handoff Trigger */}
          <TrustSignalCard
            trustSummary={flow.trust_summary}
            onRequestReview={() => setIsHandoffModalOpen(true)}
            scenarioKey={flow.scenario_key}
            hasHighRiskFindings={hasHighRiskFindings}
          />

          {/* Connected Document Card */}
          <ConnectedDocCard
            document={connectedDoc}
            scenarioKey={flow.scenario_key}
          />

          {/* Obligations & Dates Card */}
          <UpcomingObligationsCard obligations={obligations} />
        </div>
      </div>

      {/* Context-Preserving Professional Handoff Modal */}
      <ProfessionalHandoffModal
        isOpen={isHandoffModalOpen}
        onClose={() => setIsHandoffModalOpen(false)}
        flow={flow}
        connectedDocTitle={connectedDoc?.title || flow.title}
        obligations={obligations}
        findings={findings}
      />
    </div>
  );
}
