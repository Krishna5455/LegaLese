import {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  Packer,
  PageNumber,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { GeneratedDocumentContent } from "@/types/generation";

export async function exportDocx(
  content: GeneratedDocumentContent,
  createdAt?: string,
): Promise<Buffer> {
  const children: (Paragraph | Table)[] = [];

  // Title
  children.push(
    new Paragraph({
      text: content.title.toUpperCase(),
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    }),
  );

  // Subtitle / Metadata
  const metadataText = createdAt
    ? `FREELANCE SERVICE AGREEMENT  •  Generated on ${new Intl.DateTimeFormat(
        "en-US",
        { dateStyle: "full" },
      ).format(new Date(createdAt))}`
    : "FREELANCE SERVICE AGREEMENT";

  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 360 },
      children: [
        new TextRun({
          text: metadataText,
          size: 18, // 9pt
          color: "6B7280",
          font: "Arial",
        }),
      ],
    }),
  );

  // Parties Heading
  children.push(
    new Paragraph({
      text: "PARTIES TO THE AGREEMENT",
      heading: HeadingLevel.HEADING_2,
      spacing: { before: 240, after: 120 },
    }),
  );

  // Parties Detail
  children.push(
    new Paragraph({
      spacing: { after: 60 },
      children: [
        new TextRun({ text: "Freelancer (Service Provider): ", bold: true, size: 20, font: "Arial" }),
        new TextRun({ text: content.parties.freelancerName, size: 20, font: "Arial" }),
      ],
    }),
    new Paragraph({
      spacing: { after: 60 },
      children: [
        new TextRun({ text: "Client: ", bold: true, size: 20, font: "Arial" }),
        new TextRun({ text: content.parties.clientName, size: 20, font: "Arial" }),
      ],
    }),
  );

  if (content.parties.clientAddress) {
    children.push(
      new Paragraph({
        spacing: { after: 60 },
        children: [
          new TextRun({ text: "Client Address: ", bold: true, size: 20, font: "Arial" }),
          new TextRun({ text: content.parties.clientAddress, size: 20, font: "Arial" }),
        ],
      }),
    );
  }

  children.push(
    new Paragraph({
      spacing: { after: 240 },
    }),
  );

  // Agreement Sections
  const sortedSections = [...(content.sections ?? [])].sort(
    (a, b) => a.order - b.order,
  );

  for (let index = 0; index < sortedSections.length; index++) {
    const section = sortedSections[index];

    // Section Title
    children.push(
      new Paragraph({
        text: `${index + 1}. ${section.title.toUpperCase()}`,
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 120 },
      }),
    );

    // Section Paragraphs
    const paragraphs = section.content.split(/\n+/);
    for (const paragraphText of paragraphs) {
      if (paragraphText.trim().length > 0) {
        children.push(
          new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { after: 120, line: 276 }, // 1.15 line spacing
            children: [
              new TextRun({
                text: paragraphText.trim(),
                size: 20, // 10pt
                font: "Arial",
                color: "1F2937",
              }),
            ],
          }),
        );
      }
    }
  }

  // Legal Disclaimer Box (Rendered as styled single-cell table)
  const disclaimerText =
    content.disclaimer ||
    "AI-generated draft. This document is provided for informational purposes and is not a substitute for professional legal advice.";

  const disclaimerTable = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 100, type: WidthType.PERCENTAGE },
            shading: { fill: "F9FAFB" },
            margins: { top: 180, bottom: 180, left: 240, right: 240 },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" },
              bottom: { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" },
              left: { style: BorderStyle.SINGLE, size: 12, color: "B45309" }, // Accent border left
              right: { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" },
            },
            children: [
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "LEGAL DISCLAIMER",
                    bold: true,
                    size: 17, // 8.5pt
                    color: "B45309",
                    font: "Arial",
                  }),
                ],
              }),
              new Paragraph({
                children: [
                  new TextRun({
                    text: disclaimerText,
                    italics: true,
                    size: 16, // 8pt
                    color: "4B5563",
                    font: "Arial",
                  }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });

  children.push(
    new Paragraph({ spacing: { before: 360 } }),
    disclaimerTable,
  );

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: 1440, // 1 inch
              bottom: 1440,
              left: 1440,
              right: 1440,
            },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.RIGHT,
                children: [
                  new TextRun({
                    text: "LegaLese | Legal Document Draft",
                    size: 16,
                    color: "9CA3AF",
                    font: "Arial",
                  }),
                ],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({
                    text: "Page ",
                    size: 16,
                    color: "6B7280",
                    font: "Arial",
                  }),
                  new TextRun({
                    children: [PageNumber.CURRENT],
                    size: 16,
                    color: "6B7280",
                    font: "Arial",
                  }),
                  new TextRun({
                    text: " of ",
                    size: 16,
                    color: "6B7280",
                    font: "Arial",
                  }),
                  new TextRun({
                    children: [PageNumber.TOTAL_PAGES],
                    size: 16,
                    color: "6B7280",
                    font: "Arial",
                  }),
                ],
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  return await Packer.toBuffer(doc);
}
