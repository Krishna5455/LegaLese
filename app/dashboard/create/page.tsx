import Link from "next/link";
import { redirect } from "next/navigation";

import { CreateDocumentSection } from "@/components/create/CreateDocumentSection";
import { DashboardNav } from "@/components/dashboard/DashboardNav";
import { listGeneratedDocuments } from "@/lib/actions/generated-documents";
import { DOCUMENT_TYPE_LIST } from "@/lib/generation/document-types";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function CreateDocumentPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { documents } = await listGeneratedDocuments();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <DashboardNav userEmail={user.email} active="create" />

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8 space-y-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Create Legal Document
          </h1>
          <p className="mt-1 text-xs text-muted">
            Answer a few guided questions to generate a custom, structured legal contract.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Main Form Section */}
          <div className="lg:col-span-2 space-y-8">
            <CreateDocumentSection types={DOCUMENT_TYPE_LIST} />
          </div>

          {/* Sidebar / Recent Drafts Section */}
          <div className="space-y-6">
            <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
              <h2 className="text-sm font-semibold text-foreground">
                Your Generated Drafts ({documents?.length ?? 0})
              </h2>

              {!documents || documents.length === 0 ? (
                <p className="text-xs text-muted">
                  No generated drafts yet. Fill out the form to create your first agreement.
                </p>
              ) : (
                <ul className="space-y-3">
                  {documents.map((doc) => (
                    <li key={doc.id}>
                      <Link
                        href={`/dashboard/create/${doc.id}`}
                        className="block rounded-lg border border-border/80 bg-background/50 p-3 hover:border-accent/40 transition-colors"
                      >
                        <p className="text-xs font-semibold text-foreground truncate">
                          {doc.title}
                        </p>
                        <div className="mt-1 flex items-center justify-between text-[10px] text-muted font-mono">
                          <span>
                            {new Intl.DateTimeFormat("en-US", {
                              dateStyle: "short",
                            }).format(new Date(doc.created_at))}
                          </span>
                          <span className="text-accent font-semibold">View Draft →</span>
                        </div>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="rounded-xl border border-border/60 bg-surface/50 p-4 text-xs text-muted space-y-2">
              <p className="font-semibold text-foreground">💡 How generation works</p>
              <p className="leading-relaxed">
                Your inputs are validated using schema controls before being passed to Gemini. 
                The AI structures the document into fixed section modules with stable IDs to prepare for future interactive review and explanation features.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
