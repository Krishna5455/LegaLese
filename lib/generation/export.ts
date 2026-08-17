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

  const sortedSections = [...content.sections].sort((a, b) => a.order - b.order);

  for (const section of sortedSections) {
    lines.push(`## ${section.title}`, "", section.content, "", "---", "");
  }

  lines.push("## Disclaimer", "", content.disclaimer, "");

  return lines.join("\n");
}

export function generatedDocumentDownloadFilename(title: string): string {
  const safe = title
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .slice(0, 80);
  return `${safe || "Freelance_Service_Agreement"}.md`;
}
