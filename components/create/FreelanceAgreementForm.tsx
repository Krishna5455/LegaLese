"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/Button";
import { generateFreelanceAgreement } from "@/lib/actions/generated-documents";
import { FreelanceAgreementInputSchema } from "@/lib/generation/freelance-agreement-schema";
import type { FreelanceAgreementInput } from "@/types/generation";

const defaultValues: FreelanceAgreementInput = {
  freelancerName: "",
  clientName: "",
  clientAddress: "",
  servicesDescription: "",
  deliverables: "",
  startDate: "",
  completionDate: "",
  projectFee: "",
  paymentStructure: "",
  paymentSchedule: "",
  currency: "INR",
  noticePeriod: "",
  earlyTerminationWork: "",
  ipOwnership: "",
  freelancerReusableMaterials: "",
  confidentialityRequired: "yes",
  jurisdiction: "",
};

// Realistic Indian Legal-Tech Showcase Presets
const demoPresets: Record<string, { label: string; data: FreelanceAgreementInput }> = {
  web_dev: {
    label: "Web Development",
    data: {
      freelancerName: "Rahul Sharma",
      clientName: "Acme Technologies Pvt. Ltd.",
      clientAddress: "Ahmedabad, Gujarat, India",
      servicesDescription:
        "Full-stack web application development including responsive UI development, backend API development, database integration, testing, and deployment.",
      deliverables:
        "Responsive web application, admin dashboard, REST API, database setup, deployment configuration, and basic technical documentation.",
      startDate: "2026-09-01",
      completionDate: "2026-11-30",
      projectFee: "75000",
      currency: "INR",
      paymentStructure: "50% upfront, 50% on final delivery",
      paymentSchedule: "Final payment due within 15 days of invoice",
      noticePeriod: "15 days written notice",
      earlyTerminationWork:
        "Freelancer shall be paid for all work performed up to the date of termination. Completed work products shall be transferred to Client upon payment.",
      ipOwnership:
        "Client shall own all final custom deliverables upon full and final payment.",
      freelancerReusableMaterials:
        "Freelancer retains ownership of pre-existing development tools, libraries, and reusable code modules.",
      confidentialityRequired: "yes",
      jurisdiction: "Ahmedabad, Gujarat, India",
    },
  },
  ui_design: {
    label: "UI/UX Design",
    data: {
      freelancerName: "Ananya Roy",
      clientName: "Nexus Digital Studio Pvt. Ltd.",
      clientAddress: "Bengaluru, Karnataka, India",
      servicesDescription:
        "End-to-end UI/UX product design, user research, wireframing, high-fidelity mobile app screen designs, and interactive Figma prototyping.",
      deliverables:
        "Complete Figma design system, interactive screen prototypes, component library assets, and developer handover documentation.",
      startDate: "2026-09-01",
      completionDate: "2026-10-31",
      projectFee: "50000",
      currency: "INR",
      paymentStructure: "40% upfront, 30% milestone 1, 30% final delivery",
      paymentSchedule: "Payment due within 10 days of milestone approval",
      noticePeriod: "14 days written notice",
      earlyTerminationWork:
        "Freelancer shall deliver completed design assets produced up to termination upon pro-rata payment.",
      ipOwnership:
        "Client owns all final custom UI/UX design assets upon full project payment.",
      freelancerReusableMaterials:
        "Freelancer retains rights to pre-existing design templates and UI kits.",
      confidentialityRequired: "yes",
      jurisdiction: "Bengaluru, Karnataka, India",
    },
  },
  marketing: {
    label: "Digital Marketing",
    data: {
      freelancerName: "Vikram Malhotra",
      clientName: "Starlight Ventures India",
      clientAddress: "Mumbai, Maharashtra, India",
      servicesDescription:
        "Search engine optimization (SEO), social media marketing strategy, content creation, and monthly conversion analytics performance reporting.",
      deliverables:
        "Quarterly digital marketing strategy roadmap, monthly SEO audits, campaign ad copy, and analytics performance dashboards.",
      startDate: "2026-09-01",
      completionDate: "2026-12-31",
      projectFee: "60000",
      currency: "INR",
      paymentStructure: "Monthly retainer of ₹15,000 paid at start of each month",
      paymentSchedule: "Payment due on the 1st of each month",
      noticePeriod: "30 days written notice",
      earlyTerminationWork:
        "Services will cease at the end of the paid calendar month.",
      ipOwnership:
        "Client owns created marketing campaign materials and ad copy upon payment.",
      freelancerReusableMaterials:
        "Freelancer retains general marketing frameworks and analytical toolsets.",
      confidentialityRequired: "yes",
      jurisdiction: "Mumbai, Maharashtra, India",
    },
  },
};

function fieldClassName(hasError: boolean) {
  return `w-full rounded-xl border bg-surface px-4 py-3 text-base outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all ${
    hasError
      ? "border-red-400 bg-red-50/20 text-foreground"
      : "border-border hover:border-border-strong text-foreground"
  }`;
}

export function FreelanceAgreementForm() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [form, setForm] = useState<FreelanceAgreementInput>(defaultValues);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [demoBannerMessage, setDemoBannerMessage] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("web_dev");
  const [isPending, startTransition] = useTransition();

  const updateField = <K extends keyof FreelanceAgreementInput>(
    key: K,
    value: FreelanceAgreementInput[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setFieldErrors((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setErrorMessage(null);
  };

  const handleFillDemoData = (presetKey: string = selectedPreset) => {
    const preset = demoPresets[presetKey] || demoPresets.web_dev;
    setForm(preset.data);
    setSelectedPreset(presetKey);
    setFieldErrors({});
    setErrorMessage(null);
    setDemoBannerMessage(
      `✨ Demo data added (${preset.label}) — you can edit anything before generating.`,
    );
  };

  const validateStep = (step: number): boolean => {
    const errors: Record<string, string> = {};

    if (step === 1) {
      return true;
    }

    if (step === 2) {
      if (!form.freelancerName.trim()) {
        errors.freelancerName = "Freelancer name is required.";
      }
      if (!form.clientName.trim()) {
        errors.clientName = "Client or company name is required.";
      }
    }

    if (step === 3) {
      if (!form.servicesDescription.trim()) {
        errors.servicesDescription = "Services description is required.";
      }
      if (!form.deliverables.trim()) {
        errors.deliverables = "Deliverables description is required.";
      }
      if (!form.startDate.trim()) {
        errors.startDate = "Start date is required.";
      }
      if (!form.completionDate.trim()) {
        errors.completionDate = "Completion date is required.";
      }
      if (!form.projectFee.trim()) {
        errors.projectFee = "Project fee is required.";
      }
      if (!form.paymentStructure.trim()) {
        errors.paymentStructure = "Payment structure is required.";
      }
      if (!form.paymentSchedule.trim()) {
        errors.paymentSchedule = "Payment schedule is required.";
      }
    }

    if (step === 4) {
      const result = FreelanceAgreementInputSchema.safeParse(form);
      if (!result.success) {
        for (const issue of result.error.issues) {
          const key = issue.path[0];
          if (typeof key === "string" && !errors[key]) {
            errors[key] = issue.message;
          }
        }
      }
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleNextStep = () => {
    if (validateStep(currentStep)) {
      setCurrentStep((prev) => Math.min(prev + 1, 4));
    } else {
      setErrorMessage("Please fill out all required fields before proceeding.");
    }
  };

  const handlePrevStep = () => {
    setErrorMessage(null);
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setErrorMessage(null);

    const result = FreelanceAgreementInputSchema.safeParse(form);
    if (!result.success) {
      const errors: Record<string, string> = {};
      for (const issue of result.error.issues) {
        const key = issue.path[0];
        if (typeof key === "string" && !errors[key]) {
          errors[key] = issue.message;
        }
      }
      setFieldErrors(errors);
      setErrorMessage("Please fix the highlighted fields before generating.");
      return;
    }

    startTransition(async () => {
      const res = await generateFreelanceAgreement(form);
      if (res.error) {
        setErrorMessage(res.error);
        return;
      }
      if (res.documentId) {
        router.push(`/dashboard/create/${res.documentId}`);
      }
    });
  };

  return (
    <div className="space-y-7">
      {/* Wizard Header & Progress Bar */}
      {/* Wizard Header & Progress Bar */}
      <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-accent text-white font-bold text-xs shadow-xs">
              {currentStep}
            </span>
            <div>
              <span className="text-base font-bold text-foreground block">
                {currentStep === 1 && "Step 1: Agreement Type"}
                {currentStep === 2 && "Step 2: Agreement Parties"}
                {currentStep === 3 && "Step 3: Scope, Timing & Payment"}
                {currentStep === 4 && "Step 4: Review & Generate"}
              </span>
              <span className="text-xs text-secondary">
                {currentStep === 1 && "Select the contract scenario for your agreement"}
                {currentStep === 2 && "Enter contractor and client identification"}
                {currentStep === 3 && "Define project deliverables, fees, and schedule"}
                {currentStep === 4 && "Review legal terms and generate final draft"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-muted hidden lg:inline">Try Preset:</span>
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                type="button"
                onClick={() => handleFillDemoData("web_dev")}
                className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                  selectedPreset === "web_dev"
                    ? "bg-accent text-white shadow-xs"
                    : "bg-accent-soft text-accent border border-accent/20 hover:bg-accent/15"
                }`}
              >
                ⚡ Web Dev
              </button>
              <button
                type="button"
                onClick={() => handleFillDemoData("ui_design")}
                className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                  selectedPreset === "ui_design"
                    ? "bg-accent text-white shadow-xs"
                    : "bg-accent-soft text-accent border border-accent/20 hover:bg-accent/15"
                }`}
              >
                ⚡ UI/UX
              </button>
              <button
                type="button"
                onClick={() => handleFillDemoData("marketing")}
                className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                  selectedPreset === "marketing"
                    ? "bg-accent text-white shadow-xs"
                    : "bg-accent-soft text-accent border border-accent/20 hover:bg-accent/15"
                }`}
              >
                ⚡ Marketing
              </button>
            </div>
            <span className="text-xs font-mono font-semibold text-muted ml-2">
              {currentStep}/4
            </span>
          </div>
        </div>

        {/* Visual Progress Bar */}
        <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
          <div
            className="h-full bg-accent transition-all duration-300 ease-out"
            style={{ width: `${(currentStep / 4) * 100}%` }}
          />
        </div>
      </div>

      {/* Demo Banner Feedback */}
      {demoBannerMessage ? (
        <div className="rounded-xl border border-accent/20 bg-accent-soft p-3.5 text-xs font-semibold text-accent flex items-center justify-between shadow-xs animate-fadeIn">
          <span>{demoBannerMessage}</span>
          <button
            type="button"
            onClick={() => setDemoBannerMessage(null)}
            className="text-xs font-bold underline hover:no-underline ml-2"
          >
            Dismiss
          </button>
        </div>
      ) : null}


      {/* Form Steps */}
      <form onSubmit={handleSubmit} className="space-y-7">
        {/* STEP 1: What do you need? */}
        {currentStep === 1 ? (
          <div className="space-y-5">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                What document do you need?
              </h2>
              <p className="text-base text-muted">
                Select the agreement type to configure your guided questions.
              </p>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div className="rounded-2xl border-2 border-accent bg-accent/5 p-6 space-y-2 cursor-pointer shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-base font-bold text-foreground">
                    Freelance Service Agreement
                  </span>
                  <span className="rounded-full bg-accent/20 text-accent px-3 py-0.5 text-xs font-bold">
                    ✓ Active
                  </span>
                </div>
                <p className="text-sm text-muted leading-relaxed">
                  For independent contractors, freelancers, and clients defining project scope, deliverables, and payment.
                </p>
              </div>

              <div className="rounded-2xl border border-border/70 bg-surface/50 p-6 space-y-2 opacity-60 cursor-not-allowed select-none">
                <div className="flex items-center justify-between">
                  <span className="text-base font-bold text-foreground">
                    Non-Disclosure Agreement (NDA)
                  </span>
                  <span className="rounded bg-muted/15 text-muted px-2 py-0.5 text-[10px] font-semibold">
                    Coming Soon
                  </span>
                </div>
                <p className="text-sm text-muted leading-relaxed">
                  Protect confidential information and trade secrets between business partners.
                </p>
              </div>

              <div className="rounded-2xl border border-border/70 bg-surface/50 p-6 space-y-2 opacity-60 cursor-not-allowed select-none">
                <div className="flex items-center justify-between">
                  <span className="text-base font-bold text-foreground">
                    Master Services Agreement
                  </span>
                  <span className="rounded bg-muted/15 text-muted px-2 py-0.5 text-[10px] font-semibold">
                    Coming Soon
                  </span>
                </div>
                <p className="text-sm text-muted leading-relaxed">
                  Long-term commercial framework agreement for ongoing client engagements.
                </p>
              </div>

              <div className="rounded-2xl border border-border/70 bg-surface/50 p-6 space-y-2 opacity-60 cursor-not-allowed select-none">
                <div className="flex items-center justify-between">
                  <span className="text-base font-bold text-foreground">
                    Employment Agreement
                  </span>
                  <span className="rounded bg-muted/15 text-muted px-2 py-0.5 text-[10px] font-semibold">
                    Coming Soon
                  </span>
                </div>
                <p className="text-sm text-muted leading-relaxed">
                  Full-time or part-time employment contracts defining roles, salary, and benefits.
                </p>
              </div>
            </div>
          </div>
        ) : null}

        {/* STEP 2: Tell us about the agreement */}
        {currentStep === 2 ? (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                Tell us about the agreement parties
              </h2>
              <p className="text-base text-muted">
                Identify who is entering into this agreement.
              </p>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="freelancerName" className="mb-1.5 block text-base font-semibold text-foreground">
                  Your full name (Freelancer) *
                </label>
                <input
                  id="freelancerName"
                  value={form.freelancerName}
                  onChange={(e) => updateField("freelancerName", e.target.value)}
                  placeholder="e.g. Rahul Sharma"
                  className={fieldClassName(!!fieldErrors.freelancerName)}
                  disabled={isPending}
                />
                {fieldErrors.freelancerName ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.freelancerName}</p>
                ) : null}
              </div>

              <div>
                <label htmlFor="clientName" className="mb-1.5 block text-base font-semibold text-foreground">
                  Client or company name *
                </label>
                <input
                  id="clientName"
                  value={form.clientName}
                  onChange={(e) => updateField("clientName", e.target.value)}
                  placeholder="e.g. Acme Technologies Pvt. Ltd."
                  className={fieldClassName(!!fieldErrors.clientName)}
                  disabled={isPending}
                />
                {fieldErrors.clientName ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.clientName}</p>
                ) : null}
              </div>
            </div>

            <div>
              <label htmlFor="clientAddress" className="mb-1.5 block text-base font-semibold text-foreground">
                Client address (Optional)
              </label>
              <input
                id="clientAddress"
                value={form.clientAddress ?? ""}
                onChange={(e) => updateField("clientAddress", e.target.value)}
                placeholder="e.g. Ahmedabad, Gujarat, India"
                className={fieldClassName(!!fieldErrors.clientAddress)}
                disabled={isPending}
              />
            </div>
          </div>
        ) : null}

        {/* STEP 3: Tell us about the work & payment */}
        {currentStep === 3 ? (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                Tell us about the work & payment
              </h2>
              <p className="text-base text-muted">
                Specify services, project fee, timelines, and payment schedules.
              </p>
            </div>

            <div>
              <label htmlFor="servicesDescription" className="mb-1.5 block text-base font-semibold text-foreground">
                What services will you provide? *
              </label>
              <textarea
                id="servicesDescription"
                rows={3}
                value={form.servicesDescription}
                onChange={(e) => updateField("servicesDescription", e.target.value)}
                placeholder="Describe the services to be performed..."
                className={fieldClassName(!!fieldErrors.servicesDescription)}
                disabled={isPending}
              />
              {fieldErrors.servicesDescription ? (
                <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.servicesDescription}</p>
              ) : null}
            </div>

            <div>
              <label htmlFor="deliverables" className="mb-1.5 block text-base font-semibold text-foreground">
                What are the project deliverables? *
              </label>
              <textarea
                id="deliverables"
                rows={2}
                value={form.deliverables}
                onChange={(e) => updateField("deliverables", e.target.value)}
                placeholder="List key deliverables..."
                className={fieldClassName(!!fieldErrors.deliverables)}
                disabled={isPending}
              />
              {fieldErrors.deliverables ? (
                <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.deliverables}</p>
              ) : null}
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="startDate" className="mb-1.5 block text-base font-semibold text-foreground">
                  Start date *
                </label>
                <input
                  id="startDate"
                  type="date"
                  value={form.startDate}
                  onChange={(e) => updateField("startDate", e.target.value)}
                  className={fieldClassName(!!fieldErrors.startDate)}
                  disabled={isPending}
                />
                {fieldErrors.startDate ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.startDate}</p>
                ) : null}
              </div>

              <div>
                <label htmlFor="completionDate" className="mb-1.5 block text-base font-semibold text-foreground">
                  Expected completion date *
                </label>
                <input
                  id="completionDate"
                  type="date"
                  value={form.completionDate}
                  onChange={(e) => updateField("completionDate", e.target.value)}
                  className={fieldClassName(!!fieldErrors.completionDate)}
                  disabled={isPending}
                />
                {fieldErrors.completionDate ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.completionDate}</p>
                ) : null}
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="projectFee" className="mb-1.5 block text-base font-semibold text-foreground">
                  Total project fee *
                </label>
                <input
                  id="projectFee"
                  value={form.projectFee}
                  onChange={(e) => updateField("projectFee", e.target.value)}
                  placeholder="e.g. 75000"
                  className={fieldClassName(!!fieldErrors.projectFee)}
                  disabled={isPending}
                />
                {fieldErrors.projectFee ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.projectFee}</p>
                ) : null}
              </div>

              <div>
                <label htmlFor="currency" className="mb-1.5 block text-base font-semibold text-foreground">
                  Currency *
                </label>
                <input
                  id="currency"
                  value={form.currency}
                  onChange={(e) => updateField("currency", e.target.value)}
                  placeholder="INR or USD"
                  className={fieldClassName(!!fieldErrors.currency)}
                  disabled={isPending}
                />
                {fieldErrors.currency ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.currency}</p>
                ) : null}
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="paymentStructure" className="mb-1.5 block text-base font-semibold text-foreground">
                  Payment structure *
                </label>
                <input
                  id="paymentStructure"
                  value={form.paymentStructure}
                  onChange={(e) => updateField("paymentStructure", e.target.value)}
                  placeholder="e.g. 50% upfront, 50% on final delivery"
                  className={fieldClassName(!!fieldErrors.paymentStructure)}
                  disabled={isPending}
                />
                {fieldErrors.paymentStructure ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.paymentStructure}</p>
                ) : null}
              </div>

              <div>
                <label htmlFor="paymentSchedule" className="mb-1.5 block text-base font-semibold text-foreground">
                  Payment due timing *
                </label>
                <input
                  id="paymentSchedule"
                  value={form.paymentSchedule}
                  onChange={(e) => updateField("paymentSchedule", e.target.value)}
                  placeholder="e.g. Final payment due within 15 days of invoice"
                  className={fieldClassName(!!fieldErrors.paymentSchedule)}
                  disabled={isPending}
                />
                {fieldErrors.paymentSchedule ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.paymentSchedule}</p>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}

        {/* STEP 4: Review details & legal terms */}
        {currentStep === 4 ? (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                Review terms & generate agreement
              </h2>
              <p className="text-base text-muted">
                Configure IP rights, termination rules, and legal jurisdiction.
              </p>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="noticePeriod" className="mb-1.5 block text-base font-semibold text-foreground">
                  Termination notice period *
                </label>
                <input
                  id="noticePeriod"
                  value={form.noticePeriod}
                  onChange={(e) => updateField("noticePeriod", e.target.value)}
                  placeholder="e.g. 15 days written notice"
                  className={fieldClassName(!!fieldErrors.noticePeriod)}
                  disabled={isPending}
                />
                {fieldErrors.noticePeriod ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.noticePeriod}</p>
                ) : null}
              </div>

              <div>
                <label htmlFor="jurisdiction" className="mb-1.5 block text-base font-semibold text-foreground">
                  Governing jurisdiction *
                </label>
                <input
                  id="jurisdiction"
                  value={form.jurisdiction}
                  onChange={(e) => updateField("jurisdiction", e.target.value)}
                  placeholder="e.g. Ahmedabad, Gujarat, India"
                  className={fieldClassName(!!fieldErrors.jurisdiction)}
                  disabled={isPending}
                />
                {fieldErrors.jurisdiction ? (
                  <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.jurisdiction}</p>
                ) : null}
              </div>
            </div>

            <div>
              <label htmlFor="earlyTerminationWork" className="mb-1.5 block text-base font-semibold text-foreground">
                Early termination handling *
              </label>
              <textarea
                id="earlyTerminationWork"
                rows={2}
                value={form.earlyTerminationWork}
                onChange={(e) => updateField("earlyTerminationWork", e.target.value)}
                placeholder="What happens to completed work if ended early?"
                className={fieldClassName(!!fieldErrors.earlyTerminationWork)}
                disabled={isPending}
              />
              {fieldErrors.earlyTerminationWork ? (
                <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.earlyTerminationWork}</p>
              ) : null}
            </div>

            <div>
              <label htmlFor="ipOwnership" className="mb-1.5 block text-base font-semibold text-foreground">
                Intellectual property ownership *
              </label>
              <textarea
                id="ipOwnership"
                rows={2}
                value={form.ipOwnership}
                onChange={(e) => updateField("ipOwnership", e.target.value)}
                placeholder="Who owns the final work after payment?"
                className={fieldClassName(!!fieldErrors.ipOwnership)}
                disabled={isPending}
              />
              {fieldErrors.ipOwnership ? (
                <p className="mt-1 text-xs text-red-600 font-medium">{fieldErrors.ipOwnership}</p>
              ) : null}
            </div>

            <div>
              <label htmlFor="freelancerReusableMaterials" className="mb-1.5 block text-base font-semibold text-foreground">
                Reusable tools & materials *
              </label>
              <textarea
                id="freelancerReusableMaterials"
                rows={2}
                value={form.freelancerReusableMaterials}
                onChange={(e) =>
                  updateField("freelancerReusableMaterials", e.target.value)
                }
                placeholder="Rights to pre-existing code modules or tools..."
                className={fieldClassName(!!fieldErrors.freelancerReusableMaterials)}
                disabled={isPending}
              />
              {fieldErrors.freelancerReusableMaterials ? (
                <p className="mt-1 text-xs text-red-600 font-medium">
                  {fieldErrors.freelancerReusableMaterials}
                </p>
              ) : null}
            </div>

            <div>
              <label htmlFor="confidentialityRequired" className="mb-1.5 block text-base font-semibold text-foreground">
                Confidentiality obligations *
              </label>
              <select
                id="confidentialityRequired"
                value={form.confidentialityRequired}
                onChange={(e) =>
                  updateField(
                    "confidentialityRequired",
                    e.target.value as "yes" | "no",
                  )
                }
                className={fieldClassName(!!fieldErrors.confidentialityRequired)}
                disabled={isPending}
              >
                <option value="yes">Yes — Require mutual confidentiality</option>
                <option value="no">No — Standard public engagement</option>
              </select>
            </div>

            {/* Clean Details Summary Card */}
            <div className="rounded-2xl border border-border bg-surface-inset p-6 space-y-3 text-sm">
              <p className="font-bold text-foreground uppercase tracking-wider text-xs">
                Agreement Overview Summary
              </p>
              <div className="grid gap-3 sm:grid-cols-2 text-muted">
                <div>
                  Freelancer: <span className="font-semibold text-foreground">{form.freelancerName || "Not set"}</span>
                </div>
                <div>
                  Client: <span className="font-semibold text-foreground">{form.clientName || "Not set"}</span>
                </div>
                <div>
                  Fee: <span className="font-semibold text-foreground">{form.projectFee ? `₹${form.projectFee} ${form.currency}` : "Not set"}</span>
                </div>
                <div>
                  Notice Period: <span className="font-semibold text-foreground">{form.noticePeriod || "Not set"}</span>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {/* Error Alerts */}
        {errorMessage ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-600 dark:text-red-400 font-semibold">
            ⚠️ {errorMessage}
          </div>
        ) : null}

        {isPending ? (
          <div className="rounded-xl border border-accent/30 bg-accent/5 p-4 text-sm text-accent font-semibold">
            Creating your agreement... This may take up to a minute.
          </div>
        ) : null}

        {/* Step Navigation Controls */}
        <div className="flex items-center justify-between border-t border-border pt-6">
          {currentStep > 1 ? (
            <Button
              type="button"
              variant="outline"
              onClick={handlePrevStep}
              disabled={isPending}
              className="h-11 px-5 text-sm font-bold"
            >
              ← Back
            </Button>
          ) : (
            <div />
          )}

          {currentStep < 4 ? (
            <Button
              type="button"
              onClick={handleNextStep}
              disabled={isPending}
              className="h-11 px-6 text-sm font-bold shadow-xs"
            >
              Next Step →
            </Button>
          ) : (
            <Button
              type="submit"
              disabled={isPending}
              className="h-11 px-6 text-sm font-bold bg-accent text-white hover:bg-accent-hover shadow-xs"
            >
              {isPending ? "Creating agreement..." : "Generate Agreement →"}
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
