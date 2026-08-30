import Link from "next/link";
import { redirect } from "next/navigation";

import { GeneratedDocumentWorkspace } from "@/components/create/GeneratedDocumentWorkspace";
import { getGeneratedDocument } from "@/lib/actions/generated-documents";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

type PageProps = {
  params: Promise<{ id: string }>;
};

export default async function ViewGeneratedDocumentPage({ params }: PageProps) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { id } = await params;
  const { document: doc, error } = await getGeneratedDocument(id);

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8">
      {error || !doc ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-8 text-center space-y-4">
          <h1 className="text-xl font-bold text-red-600 dark:text-red-400">
            Document Not Found
          </h1>
          <p className="text-xs text-muted">
            {error ?? "The requested generated document could not be found or access was denied."}
          </p>
          <div>
            <Link
              href="/dashboard/create"
              className="inline-block rounded-lg bg-accent px-4 py-2 text-xs font-medium text-white hover:bg-accent/90 transition-colors"
            >
              Back to Create Document
            </Link>
          </div>
        </div>
      ) : (
        <GeneratedDocumentWorkspace document={doc} />
      )}
    </main>
  );
}
