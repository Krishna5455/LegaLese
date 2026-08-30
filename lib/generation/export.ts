import type { GeneratedDocumentContent } from "@/types/generation";

export function generatedDocumentToMarkdown(
  content: GeneratedDocumentContent,
  generatedAt?: string,
): string {
  const lines: string[] = [
    `# ${content.title}`,
    "",
    `**Document type:** Freelance Service Agreement`,
  ];

  if (generatedAt) {
    lines.push(
      `**Generated:** ${new Intl.DateTimeFormat("en-US", {
        dateStyle: "full",
        timeStyle: "short",
      }).format(new Date(generatedAt))}`,
    );
  }

  lines.push(
    "",
    "---",
    "",
    "## Parties",
    "",
    `- **Freelancer:** ${content.parties.freelancerName}`,
    `- **Client:** ${content.parties.clientName}`,
  );

  if (content.parties.clientAddress) {
    lines.push(`- **Client address:** ${content.parties.clientAddress}`);
  }

  lines.push("");

  const sortedSections = [...(content.sections ?? [])].sort((a, b) => a.order - b.order);

  for (const section of sortedSections) {
    const text = section.content || (section as unknown as { body?: string }).body || "";
    lines.push(`## ${section.title || "Section"}`, "", text, "", "---", "");
  }

  if (content.disclaimer) {
    lines.push("## Disclaimer", "", content.disclaimer, "");
  }

  return lines.join("\n");
}

export function generatedDocumentDownloadFilename(
  title: string,
  format: "pdf" | "docx" | "md" = "md",
  documentId?: string,
): string {
  const safeTitle = (title || "Freelance Service Agreement")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);

  const baseName = safeTitle || "Freelance-Service-Agreement";
  const shortId = documentId ? `-${documentId.slice(0, 8)}` : "";
  return `${baseName}${shortId}.${format}`;
}
