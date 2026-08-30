import Link from "next/link";
import { redirect } from "next/navigation";

import { CreateDocumentSection } from "@/components/create/CreateDocumentSection";
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
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">
          Create Document
        </h1>
        <p className="mt-1 text-xs text-muted">
          Answer a few simple questions to create your customized legal agreement.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Main Form Section */}
        <div className="lg:col-span-2 space-y-8">
          <CreateDocumentSection types={DOCUMENT_TYPE_LIST} />
        </div>

        {/* Sidebar / Recent Drafts Section */}
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-surface p-6 space-y-4 shadow-sm">
            <h2 className="text-sm font-semibold text-foreground">
              Your Documents ({documents?.length ?? 0})
            </h2>

            {!documents || documents.length === 0 ? (
              <p className="text-xs text-muted">
                No agreements created yet. Complete the wizard to create your first document.
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
                        <span className="text-accent font-semibold">View →</span>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="rounded-xl border border-border/60 bg-surface/50 p-4 text-xs text-muted space-y-2">
            <p className="font-semibold text-foreground">💡 Guided Agreement Builder</p>
            <p className="leading-relaxed">
              Your responses are structured into fixed legal section modules, preparing your agreement for instant plain-language breakdown and clause-level review.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
