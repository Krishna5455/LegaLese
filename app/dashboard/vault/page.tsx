import { redirect } from "next/navigation";
import { DocumentVaultView, type VaultDocument } from "@/components/vault/DocumentVaultView";
import { createClient } from "@/lib/supabase/server";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";
import { isDemoMode, DEMO_VAULT_DOCUMENTS } from "@/lib/demo/demo-state";

export const dynamic = "force-dynamic";

export default async function VaultPage() {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  const isDemo = isDemoMode() && user.id === "00000000-0000-0000-0000-000000000000";

  const vaultDocs: VaultDocument[] = isDemo ? [...DEMO_VAULT_DOCUMENTS] : [];

  if (!isDemo) {
    // Fetch user documents, generated agreements, analyses, and flows concurrently
    const [
      { data: uploadedDocs },
      { data: generatedDocs },
      { data: analyses },
      { data: flows },
    ] = await Promise.all([
      supabase
        .from("documents")
        .select("id, filename, document_type, status, created_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      supabase
        .from("generated_documents")
        .select("id, title, document_type, status, created_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false }),
      supabase
        .from("analyses")
        .select("document_id, risk_score")
        .eq("user_id", user.id),
      supabase
        .from("legal_flows")
        .select("id, title, document_id, generated_doc_id")
        .eq("user_id", user.id),
    ]);

    const riskScoreMap = new Map<string, number>();
    (analyses || []).forEach((a) => {
      if (a.risk_score != null) riskScoreMap.set(a.document_id, a.risk_score);
    });

    const flowMap = new Map<string, { id: string; title: string }>();
    (flows || []).forEach((f) => {
      if (f.document_id) flowMap.set(f.document_id, { id: f.id, title: f.title });
      if (f.generated_doc_id) flowMap.set(f.generated_doc_id, { id: f.id, title: f.title });
    });

    // Add generated documents
    (generatedDocs || []).forEach((g) => {
      const linkedFlow = flowMap.get(g.id);
      vaultDocs.push({
        id: g.id,
        title: g.title,
        type: g.document_type || "Agreement",
        status: g.status || "Draft",
        isGenerated: true,
        createdAt: g.created_at,
        riskScore: 0,
        flowId: linkedFlow?.id,
        flowTitle: linkedFlow?.title,
        url: `/dashboard/create/${g.id}`,
      });
    });

    // Add uploaded contracts
    (uploadedDocs || []).forEach((u) => {
      const linkedFlow = flowMap.get(u.id);
      const risk = riskScoreMap.get(u.id) ?? null;
      vaultDocs.push({
        id: u.id,
        title: u.filename,
        type: u.document_type || "Contract",
        status: u.status || "Uploaded",
        isGenerated: false,
        createdAt: u.created_at,
        riskScore: risk,
        flowId: linkedFlow?.id,
        flowTitle: linkedFlow?.title,
        url: `/dashboard/documents/${u.id}`,
      });
    });

    if (vaultDocs.length === 0 && isDemoMode()) {
      vaultDocs.push(...DEMO_VAULT_DOCUMENTS);
    }

    // Sort unified documents by creation date descending
    vaultDocs.sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );
  }

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8">
      <DocumentVaultView documents={vaultDocs} />
    </main>
  );
}
