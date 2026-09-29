import { notFound, redirect } from "next/navigation";
import { getLegalFlow } from "@/lib/actions/legalflow";
import { LegalFlowWorkspace } from "@/components/legalflow/LegalFlowWorkspace";
import { createClient } from "@/lib/supabase/server";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";
import { isDemoMode, DEMO_CALENDAR_EVENTS } from "@/lib/demo/demo-state";
import type { LegalCalendarEvent } from "@/types/legalflow";

export const dynamic = "force-dynamic";

interface FlowPageProps {
  params: Promise<{ id: string }>;
}

export default async function FlowDetailPage({ params }: FlowPageProps) {
  const { id } = await params;
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  // 1. Fetch LegalFlow with verified user ownership
  const { flow, error } = await getLegalFlow(id);

  if (error || !flow) {
    notFound();
  }

  const isDemo = isDemoMode() && (id.startsWith("demo-") || user.id === "00000000-0000-0000-0000-000000000000");

  let connectedDoc: {
    id: string;
    title: string;
    type: string;
    status: string;
    isGenerated?: boolean;
    url: string;
  } | null = null;

  let obligations: LegalCalendarEvent[] = [];
  let findings: Array<{
    category: string;
    risk_level: string;
    explanation: string;
    questions?: string[];
  }> = [];

  if (isDemo) {
    connectedDoc = {
      id: "demo-doc-1",
      title: "Freelance Website Development Agreement",
      type: "freelance_service_agreement",
      status: "draft",
      isGenerated: true,
      url: `/dashboard/documents/demo-doc-1`,
    };
    obligations = DEMO_CALENDAR_EVENTS.slice(0, 3);
    findings = [
      {
        category: "Intellectual Property Transfer",
        risk_level: "low",
        explanation: "Deliverables and source code explicitly transfer to client upon full payment clearance.",
        questions: ["Are third-party open-source libraries exempted from exclusive assignment?"],
      },
      {
        category: "Payment Milestone Retainers",
        risk_level: "low",
        explanation: "50% upfront deposit required with final 50% payable upon milestone sign-off.",
        questions: ["Is there a defined timeline for client review and acceptance?"],
      },
    ];
  } else {
    let activeDocId: string | null = flow.document_id || null;

    // Resolve connected document
    if (flow.generated_doc_id) {
      const { data: genDoc } = await supabase
        .from("generated_documents")
        .select("id, title, document_type, status")
        .eq("id", flow.generated_doc_id)
        .eq("user_id", user.id)
        .single();

      if (genDoc) {
        connectedDoc = {
          id: genDoc.id,
          title: genDoc.title,
          type: genDoc.document_type,
          status: genDoc.status,
          isGenerated: true,
          url: `/dashboard/create/${genDoc.id}`,
        };
      }
    } else if (flow.document_id) {
      const { data: doc } = await supabase
        .from("documents")
        .select("id, filename, document_type, status")
        .eq("id", flow.document_id)
        .eq("user_id", user.id)
        .single();

      if (doc) {
        connectedDoc = {
          id: doc.id,
          title: doc.filename,
          type: doc.document_type || "Contract",
          status: doc.status || "Uploaded",
          isGenerated: false,
          url: `/dashboard/documents/${doc.id}`,
        };
        activeDocId = doc.id;
      }
    }

    // Concurrently fetch obligations and findings if active document exists
    if (activeDocId) {
      const [oblsResult, findingsResult] = await Promise.all([
        supabase
          .from("obligations")
          .select("id, description, responsible_party, deadline")
          .eq("document_id", activeDocId)
          .order("created_at", { ascending: true })
          .limit(4),
        supabase
          .from("findings")
          .select("category, risk_level, explanation, questions")
          .eq("document_id", activeDocId)
          .order("created_at", { ascending: true })
          .limit(6),
      ]);

      if (oblsResult.data && oblsResult.data.length > 0) {
        obligations = oblsResult.data.map((o) => {
          const desc = o.description.toLowerCase();
          let type: LegalCalendarEvent["type"] = "deadline";
          let amount: string | undefined;

          if (desc.includes("pay") || desc.includes("fee") || desc.includes("$") || desc.includes("₹")) {
            type = "payment";
            const amtMatch = o.description.match(/(?:₹|\$|USD|INR|EUR)\s*[\d,]+(?:\.\d+)?/i);
            if (amtMatch) amount = amtMatch[0];
          } else if (desc.includes("deliver") || desc.includes("milestone")) {
            type = "delivery";
          }

          return {
            id: o.id,
            documentId: activeDocId ?? undefined,
            title: o.description.length > 50 ? `${o.description.slice(0, 47)}...` : o.description,
            date: o.deadline || "Within 30 Days",
            responsibleParty: o.responsible_party || "Contractor",
            type,
            status: "pending",
            amount,
          };
        });
      }

      if (findingsResult.data) {
        findings = findingsResult.data as Array<{
          category: string;
          risk_level: string;
          explanation: string;
          questions?: string[];
        }>;
      }
    }
  }

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8">
      <LegalFlowWorkspace
        initialFlow={flow}
        connectedDoc={connectedDoc}
        obligations={obligations}
        findings={findings}
      />
    </main>
  );
}
