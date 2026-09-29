"use server";

import { revalidatePath } from "next/cache";
import { classifyUserIntent } from "@/lib/legalflow/intent-engine";
import { createClient } from "@/lib/supabase/server";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";
import {
  isDemoMode,
  DEMO_USER,
  getDemoFlow,
  saveDemoFlow,
  listDemoFlows,
  updateDemoFlowStep,
  DEMO_CALENDAR_EVENTS,
} from "@/lib/demo/demo-state";
import type {
  FlowStepStatus,
  LegalCalendarEvent,
  LegalFlowRow,
  LegalFlowStep,
} from "@/types/legalflow";

export type CreateFlowResult = {
  success?: boolean;
  error?: string;
  flowId?: string;
  flow?: LegalFlowRow;
};

export type GetFlowResult = {
  flow?: LegalFlowRow | null;
  error?: string;
};

export type ListFlowsResult = {
  flows?: LegalFlowRow[];
  error?: string;
};

/**
 * Creates a new LegalFlow from a user intent string.
 */
export async function createLegalFlowFromIntent(
  intentQuery: string,
): Promise<CreateFlowResult> {
  if (!intentQuery || intentQuery.trim().length === 0) {
    return { error: "Please enter what you are trying to accomplish." };
  }

  // In demo mode, provide instant synchronous in-memory flow
  if (isDemoMode()) {
    const isCanonicalFreelance =
      intentQuery.toLowerCase().includes("freelance") ||
      intentQuery.toLowerCase().includes("website");

    const flowId = isCanonicalFreelance
      ? "demo-freelance-flow"
      : `demo-flow-${Date.now()}`;

    const intent = classifyUserIntent(intentQuery);
    const demoFlow: LegalFlowRow = {
      id: flowId,
      user_id: DEMO_USER.id,
      title: intent.title,
      intent_query: intent.rawQuery,
      scenario_key: intent.scenarioKey,
      current_stage: isCanonicalFreelance ? "review" : "prepare",
      status: "active",
      steps: intent.recommendedSteps,
      trust_summary: intent.trustRecommendation,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    saveDemoFlow(demoFlow);

    return {
      success: true,
      flowId: demoFlow.id,
      flow: demoFlow,
    };
  }

  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { error: "You must be signed in to start a LegalFlow." };
  }

  const intent = classifyUserIntent(intentQuery);

  const { data: row, error: dbError } = await supabase
    .from("legal_flows")
    .insert({
      user_id: user.id,
      title: intent.title,
      intent_query: intent.rawQuery,
      scenario_key: intent.scenarioKey,
      current_stage: "prepare",
      status: "active",
      steps: intent.recommendedSteps,
      trust_summary: intent.trustRecommendation,
    })
    .select("*")
    .single();

  if (dbError) {
    console.error("[LegaLese/createLegalFlowFromIntent] DB error:", dbError);
    return {
      error: "Unable to start your LegalFlow journey right now. Please try again.",
    };
  }

  revalidatePath("/dashboard");
  revalidatePath("/dashboard/flow");

  return {
    success: true,
    flowId: (row as LegalFlowRow).id,
    flow: row as LegalFlowRow,
  };
}

/**
 * Retrieves a single LegalFlow by ID.
 */
export async function getLegalFlow(flowId: string): Promise<GetFlowResult> {
  if (!flowId) {
    return { error: "LegalFlow ID is required." };
  }

  if (isDemoMode() && (flowId.startsWith("demo-") || flowId === "demo-freelance-flow")) {
    const demoFlow = getDemoFlow(flowId);
    if (demoFlow) {
      return { flow: demoFlow };
    }
  }

  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { error: "You must be signed in to view this LegalFlow." };
  }

  if (isDemoMode() && user.id === DEMO_USER.id) {
    const demoFlow = getDemoFlow(flowId);
    if (demoFlow) {
      return { flow: demoFlow };
    }
  }

  const { data, error } = await supabase
    .from("legal_flows")
    .select("*")
    .eq("id", flowId)
    .eq("user_id", user.id)
    .single();

  if (error || !data) {
    if (isDemoMode()) {
      const demoFlow = getDemoFlow(flowId);
      if (demoFlow) return { flow: demoFlow };
    }
    return {
      error: "LegalFlow not found or you do not have permission to view it.",
    };
  }

  return { flow: data as LegalFlowRow };
}

/**
 * Lists all active and completed LegalFlows for the current user.
 */
export async function listLegalFlows(): Promise<ListFlowsResult> {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { error: "You must be signed in to view your LegalFlows." };
  }

  if (isDemoMode() && user.id === "00000000-0000-0000-0000-000000000000") {
    return { flows: listDemoFlows() };
  }

  const { data, error } = await supabase
    .from("legal_flows")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    if (isDemoMode()) {
      return { flows: listDemoFlows() };
    }
    console.error("[LegaLese/listLegalFlows] Error:", error);
    return { error: "Unable to load your LegalFlows." };
  }

  return { flows: (data as LegalFlowRow[]) ?? [] };
}

/**
 * Updates status of a specific step within a LegalFlow.
 */
export async function updateLegalFlowStep(
  flowId: string,
  stepId: string,
  newStatus: FlowStepStatus,
): Promise<{ success?: boolean; error?: string; flow?: LegalFlowRow }> {
  if (isDemoMode() && (flowId.startsWith("demo-") || !flowId)) {
    const updated = updateDemoFlowStep(flowId, stepId, newStatus);
    if (updated) {
      revalidatePath(`/dashboard/flow/${flowId}`);
      revalidatePath("/dashboard");
      return { success: true, flow: updated };
    }
  }

  const { flow, error } = await getLegalFlow(flowId);
  if (error || !flow) {
    if (isDemoMode()) {
      const updated = updateDemoFlowStep(flowId, stepId, newStatus);
      if (updated) return { success: true, flow: updated };
    }
    return { error: error ?? "LegalFlow not found." };
  }

  const updatedSteps = (flow.steps || []).map((step: LegalFlowStep) => {
    if (step.id === stepId) {
      return { ...step, status: newStatus };
    }
    return step;
  });

  // Calculate if all steps are completed
  const allDone = updatedSteps.every((s) => s.status === "completed");

  const supabase = await createClient();
  const { data: updated, error: updateErr } = await supabase
    .from("legal_flows")
    .update({
      steps: updatedSteps,
      status: allDone ? "completed" : flow.status,
      updated_at: new Date().toISOString(),
    })
    .eq("id", flowId)
    .eq("user_id", flow.user_id)
    .select("*")
    .single();

  if (updateErr) {
    if (isDemoMode()) {
      const demoUpdated = updateDemoFlowStep(flowId, stepId, newStatus);
      if (demoUpdated) return { success: true, flow: demoUpdated };
    }
    return { error: "Failed to update step progress." };
  }

  revalidatePath(`/dashboard/flow/${flowId}`);
  revalidatePath("/dashboard");

  return { success: true, flow: updated as LegalFlowRow };
}

/**
 * Links an uploaded or generated document to a LegalFlow.
 */
export async function linkDocumentToFlow(
  flowId: string,
  documentId: string,
  isGenerated = false,
): Promise<{ success?: boolean; error?: string }> {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { error: "Authentication required." };
  }

  const updatePayload: Record<string, unknown> = isGenerated
    ? { generated_doc_id: documentId }
    : { document_id: documentId };

  const { error } = await supabase
    .from("legal_flows")
    .update(updatePayload)
    .eq("id", flowId)
    .eq("user_id", user.id);

  if (error) {
    return { error: "Failed to link document to journey." };
  }

  revalidatePath(`/dashboard/flow/${flowId}`);
  return { success: true };
}

/**
 * Deletes a LegalFlow.
 */
export async function deleteLegalFlow(
  flowId: string,
): Promise<{ success?: boolean; error?: string }> {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { error: "Authentication required." };
  }

  const { error } = await supabase
    .from("legal_flows")
    .delete()
    .eq("id", flowId)
    .eq("user_id", user.id);

  if (error) {
    return { error: "Failed to delete LegalFlow." };
  }

  revalidatePath("/dashboard");
  return { success: true };
}

/**
 * Aggregates all extracted obligations and deadlines across documents for the Legal Calendar.
 */
export async function getCalendarObligations(): Promise<{
  events: LegalCalendarEvent[];
  error?: string;
}> {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    return { events: [], error: "Sign in required." };
  }

  if (isDemoMode() && user.id === "00000000-0000-0000-0000-000000000000") {
    return { events: DEMO_CALENDAR_EVENTS };
  }

  try {
    // 1. Fetch user's documents and legal flows
    const [{ data: userDocs }, { data: userFlows }] = await Promise.all([
      supabase
        .from("documents")
        .select("id, filename")
        .eq("user_id", user.id),
      supabase
        .from("legal_flows")
        .select("id, document_id, generated_doc_id, title")
        .eq("user_id", user.id),
    ]);

    const docIds = (userDocs || []).map((d) => d.id);
    const docMap = new Map((userDocs || []).map((d) => [d.id, d.filename]));
    const docFlowMap = new Map<string, string>();
    (userFlows || []).forEach((f) => {
      if (f.document_id) docFlowMap.set(f.document_id, f.id);
      if (f.generated_doc_id) docFlowMap.set(f.generated_doc_id, f.id);
    });

    let rawObligations: Array<{
      id: string;
      document_id: string;
      description: string;
      responsible_party: string | null;
      deadline: string | null;
    }> = [];

    if (docIds.length > 0) {
      const { data: obls } = await supabase
        .from("obligations")
        .select("id, document_id, description, responsible_party, deadline")
        .in("document_id", docIds);

      if (obls) {
        rawObligations = obls;
      }
    }

    if (rawObligations.length === 0 && isDemoMode()) {
      return { events: DEMO_CALENDAR_EVENTS };
    }

    // 2. Map obligations into structured LegalCalendarEvent items
    const events: LegalCalendarEvent[] = rawObligations.map((o) => {
      const flowId = docFlowMap.get(o.document_id);
      const desc = o.description.toLowerCase();
      let type: LegalCalendarEvent["type"] = "deadline";
      let amount: string | undefined;

      if (desc.includes("pay") || desc.includes("fee") || desc.includes("$") || desc.includes("₹") || desc.includes("invoice")) {
        type = "payment";
        const amtMatch = o.description.match(/(?:₹|\$|USD|INR|EUR)\s*[\d,]+(?:\.\d+)?/i);
        if (amtMatch) amount = amtMatch[0];
      } else if (desc.includes("deliver") || desc.includes("milestone") || desc.includes("submission")) {
        type = "delivery";
      } else if (desc.includes("renew") || desc.includes("expire") || desc.includes("term")) {
        type = "renewal";
      } else if (desc.includes("notice") || desc.includes("terminat")) {
        type = "notice";
      }

      return {
        id: o.id,
        flowId: flowId ?? undefined,
        documentId: o.document_id,
        title: o.description.length > 60 ? `${o.description.slice(0, 57)}...` : o.description,
        description: `Source: ${docMap.get(o.document_id) || "Document"}`,
        date: o.deadline || "Within 30 Days of Execution",
        responsibleParty: o.responsible_party || "Mutual",
        type,
        status: "pending",
        amount,
      };
    });

    return { events };
  } catch (err) {
    console.error("[LegaLese/getCalendarObligations] Error:", err);
    return { events: [], error: "Unable to aggregate calendar obligations." };
  }
}
