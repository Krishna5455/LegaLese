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
  currency: "USD",
  noticePeriod: "",
  earlyTerminationWork: "",
  ipOwnership: "",
  freelancerReusableMaterials: "",
  confidentialityRequired: "yes",
  jurisdiction: "",
};

function fieldClassName(hasError: boolean) {
  return `w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-accent/30 ${
    hasError ? "border-red-400" : "border-border"
  }`;
}

export function FreelanceAgreementForm() {
  const router = useRouter();
  const [form, setForm] = useState<FreelanceAgreementInput>(defaultValues);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
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

  const validateClient = (): boolean => {
    const result = FreelanceAgreementInputSchema.safeParse(form);
    if (result.success) {
      setFieldErrors({});
      return true;
    }

    const errors: Record<string, string> = {};
    for (const issue of result.error.issues) {
      const key = issue.path[0];
      if (typeof key === "string" && !errors[key]) {
        errors[key] = issue.message;
      }
    }
    setFieldErrors(errors);
    return false;
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setErrorMessage(null);

    if (!validateClient()) {
      setErrorMessage("Please fix the highlighted fields before continuing.");
      return;
    }

    startTransition(async () => {
      const result = await generateFreelanceAgreement(form);
      if (result.error) {
        setErrorMessage(result.error);
        return;
      }
      if (result.documentId) {
        router.push(`/dashboard/create/${result.documentId}`);
      }
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Parties</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="freelancerName" className="mb-1 block text-xs font-medium">
              Your full name (freelancer) *
            </label>
            <input
              id="freelancerName"
              value={form.freelancerName}
              onChange={(e) => updateField("freelancerName", e.target.value)}
              className={fieldClassName(!!fieldErrors.freelancerName)}
              disabled={isPending}
            />
            {fieldErrors.freelancerName ? (
              <p className="mt-1 text-xs text-red-600">{fieldErrors.freelancerName}</p>
            ) : null}
          </div>
          <div>
            <label htmlFor="clientName" className="mb-1 block text-xs font-medium">
              Client or company name *
            </label>
            <input
              id="clientName"
              value={form.clientName}
              onChange={(e) => updateField("clientName", e.target.value)}
              className={fieldClassName(!!fieldErrors.clientName)}
              disabled={isPending}
            />
            {fieldErrors.clientName ? (
              <p className="mt-1 text-xs text-red-600">{fieldErrors.clientName}</p>
            ) : null}
          </div>
        </div>
        <div>
          <label htmlFor="clientAddress" className="mb-1 block text-xs font-medium">
            Client address (optional)
          </label>
          <input
            id="clientAddress"
            value={form.clientAddress ?? ""}
            onChange={(e) => updateField("clientAddress", e.target.value)}
            className={fieldClassName(!!fieldErrors.clientAddress)}
            disabled={isPending}
          />
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Work</h2>
        <div>
          <label htmlFor="servicesDescription" className="mb-1 block text-xs font-medium">
            What services will you provide? *
          </label>
          <textarea
            id="servicesDescription"
            rows={3}
            value={form.servicesDescription}
            onChange={(e) => updateField("servicesDescription", e.target.value)}
            className={fieldClassName(!!fieldErrors.servicesDescription)}
            disabled={isPending}
          />
          {fieldErrors.servicesDescription ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.servicesDescription}</p>
          ) : null}
        </div>
        <div>
          <label htmlFor="deliverables" className="mb-1 block text-xs font-medium">
            What are the project deliverables? *
          </label>
          <textarea
            id="deliverables"
            rows={3}
            value={form.deliverables}
            onChange={(e) => updateField("deliverables", e.target.value)}
            className={fieldClassName(!!fieldErrors.deliverables)}
            disabled={isPending}
          />
          {fieldErrors.deliverables ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.deliverables}</p>
          ) : null}
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="startDate" className="mb-1 block text-xs font-medium">
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
              <p className="mt-1 text-xs text-red-600">{fieldErrors.startDate}</p>
            ) : null}
          </div>
          <div>
            <label htmlFor="completionDate" className="mb-1 block text-xs font-medium">
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
              <p className="mt-1 text-xs text-red-600">{fieldErrors.completionDate}</p>
            ) : null}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Payment</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="projectFee" className="mb-1 block text-xs font-medium">
              Total project fee *
            </label>
            <input
              id="projectFee"
              value={form.projectFee}
              onChange={(e) => updateField("projectFee", e.target.value)}
              placeholder="e.g. 5000"
              className={fieldClassName(!!fieldErrors.projectFee)}
              disabled={isPending}
            />
            {fieldErrors.projectFee ? (
              <p className="mt-1 text-xs text-red-600">{fieldErrors.projectFee}</p>
            ) : null}
          </div>
          <div>
            <label htmlFor="currency" className="mb-1 block text-xs font-medium">
              Currency *
            </label>
            <input
              id="currency"
              value={form.currency}
              onChange={(e) => updateField("currency", e.target.value)}
              placeholder="USD"
              className={fieldClassName(!!fieldErrors.currency)}
              disabled={isPending}
            />
            {fieldErrors.currency ? (
              <p className="mt-1 text-xs text-red-600">{fieldErrors.currency}</p>
            ) : null}
          </div>
        </div>
        <div>
          <label htmlFor="paymentStructure" className="mb-1 block text-xs font-medium">
            How will payment be structured? *
          </label>
          <input
            id="paymentStructure"
            value={form.paymentStructure}
            onChange={(e) => updateField("paymentStructure", e.target.value)}
            placeholder="e.g. 50% upfront, 50% on completion"
            className={fieldClassName(!!fieldErrors.paymentStructure)}
            disabled={isPending}
          />
          {fieldErrors.paymentStructure ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.paymentStructure}</p>
          ) : null}
        </div>
        <div>
          <label htmlFor="paymentSchedule" className="mb-1 block text-xs font-medium">
            When is payment due? *
          </label>
          <input
            id="paymentSchedule"
            value={form.paymentSchedule}
            onChange={(e) => updateField("paymentSchedule", e.target.value)}
            placeholder="e.g. Net 15 days after invoice"
            className={fieldClassName(!!fieldErrors.paymentSchedule)}
            disabled={isPending}
          />
          {fieldErrors.paymentSchedule ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.paymentSchedule}</p>
          ) : null}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Termination</h2>
        <div>
          <label htmlFor="noticePeriod" className="mb-1 block text-xs font-medium">
            How much notice is required to end the agreement? *
          </label>
          <input
            id="noticePeriod"
            value={form.noticePeriod}
            onChange={(e) => updateField("noticePeriod", e.target.value)}
            placeholder="e.g. 14 days written notice"
            className={fieldClassName(!!fieldErrors.noticePeriod)}
            disabled={isPending}
          />
          {fieldErrors.noticePeriod ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.noticePeriod}</p>
          ) : null}
        </div>
        <div>
          <label htmlFor="earlyTerminationWork" className="mb-1 block text-xs font-medium">
            What happens to completed work if the agreement ends early? *
          </label>
          <textarea
            id="earlyTerminationWork"
            rows={3}
            value={form.earlyTerminationWork}
            onChange={(e) => updateField("earlyTerminationWork", e.target.value)}
            className={fieldClassName(!!fieldErrors.earlyTerminationWork)}
            disabled={isPending}
          />
          {fieldErrors.earlyTerminationWork ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.earlyTerminationWork}</p>
          ) : null}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Intellectual property</h2>
        <div>
          <label htmlFor="ipOwnership" className="mb-1 block text-xs font-medium">
            Who owns the final work after payment is made? *
          </label>
          <textarea
            id="ipOwnership"
            rows={2}
            value={form.ipOwnership}
            onChange={(e) => updateField("ipOwnership", e.target.value)}
            className={fieldClassName(!!fieldErrors.ipOwnership)}
            disabled={isPending}
          />
          {fieldErrors.ipOwnership ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.ipOwnership}</p>
          ) : null}
        </div>
        <div>
          <label htmlFor="freelancerReusableMaterials" className="mb-1 block text-xs font-medium">
            Can you keep rights to reusable tools, templates, or general materials? *
          </label>
          <textarea
            id="freelancerReusableMaterials"
            rows={2}
            value={form.freelancerReusableMaterials}
            onChange={(e) =>
              updateField("freelancerReusableMaterials", e.target.value)
            }
            className={fieldClassName(!!fieldErrors.freelancerReusableMaterials)}
            disabled={isPending}
          />
          {fieldErrors.freelancerReusableMaterials ? (
            <p className="mt-1 text-xs text-red-600">
              {fieldErrors.freelancerReusableMaterials}
            </p>
          ) : null}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Confidentiality</h2>
        <div>
          <label htmlFor="confidentialityRequired" className="mb-1 block text-xs font-medium">
            Should both parties keep project information confidential? *
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
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold text-foreground">Disputes</h2>
        <div>
          <label htmlFor="jurisdiction" className="mb-1 block text-xs font-medium">
            Which state, city, or country should handle legal disputes related to this agreement? *
          </label>
          <input
            id="jurisdiction"
            value={form.jurisdiction}
            onChange={(e) => updateField("jurisdiction", e.target.value)}
            placeholder="e.g. California, USA"
            className={fieldClassName(!!fieldErrors.jurisdiction)}
            disabled={isPending}
          />
          {fieldErrors.jurisdiction ? (
            <p className="mt-1 text-xs text-red-600">{fieldErrors.jurisdiction}</p>
          ) : null}
        </div>
      </section>

      {errorMessage ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {errorMessage}
        </div>
      ) : null}

      {isPending ? (
        <div className="rounded-lg border border-accent/30 bg-accent/5 p-3 text-sm text-accent">
          Creating your agreement… This may take up to a minute.
        </div>
      ) : null}

      <Button type="submit" disabled={isPending} className="w-full sm:w-auto">
        {isPending ? "Creating your agreement…" : "Generate Agreement"}
      </Button>
    </form>
  );
}
