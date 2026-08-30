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

  const sanitizeText = (text?: string | null): string => {
    if (!text || typeof text !== "string") return "";
    return text
      .replace(/^#+\s*/gm, "")
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/\*(.*?)\*/g, "$1")
      .replace(/^-\s*/gm, "• ")
      .trim();
  };

  // --- DOCUMENT TITLE ---
  children.push(
    new Paragraph({
      text: (content.title || "FREELANCE SERVICE AGREEMENT").toUpperCase(),
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }),
  );

  // --- PREAMBLE ---
  children.push(
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { after: 180 },
      children: [
        new TextRun({
          text: 'This Freelance Service Agreement ("Agreement") is entered into by and between the following parties:',
          size: 20, // 10pt
          font: "Times New Roman",
          color: "111111",
        }),
      ],
    }),
  );

  // --- PARTIES TABLE (2 COLUMNS) ---
  const effectiveDateStr = createdAt
    ? new Intl.DateTimeFormat("en-US", { dateStyle: "long" }).format(
        new Date(createdAt),
      )
    : new Intl.DateTimeFormat("en-US", { dateStyle: "long" }).format(new Date());

  const partiesTable = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          // Client Column
          new TableCell({
            width: { size: 50, type: WidthType.PERCENTAGE },
            borders: {
              top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
            },
            children: [
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "CLIENT",
                    bold: true,
                    size: 20,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 40 },
                children: [
                  new TextRun({
                    text: `Name: ${content.parties.clientName}`,
                    size: 19,
                    font: "Times New Roman",
                  }),
                ],
              }),
              ...(content.parties.clientAddress
                ? [
                    new Paragraph({
                      spacing: { after: 40 },
                      children: [
                        new TextRun({
                          text: `Address: ${content.parties.clientAddress}`,
                          size: 19,
                          font: "Times New Roman",
                        }),
                      ],
                    }),
                  ]
                : []),
            ],
          }),
          // Freelancer Column
          new TableCell({
            width: { size: 50, type: WidthType.PERCENTAGE },
            borders: {
              top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
            },
            children: [
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "FREELANCER",
                    bold: true,
                    size: 20,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 40 },
                children: [
                  new TextRun({
                    text: `Name: ${content.parties.freelancerName}`,
                    size: 19,
                    font: "Times New Roman",
                  }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });

  children.push(partiesTable);

  children.push(
    new Paragraph({
      spacing: { before: 180, after: 240 },
      children: [
        new TextRun({
          text: `Effective Date: ${effectiveDateStr}`,
          italics: true,
          size: 19,
          font: "Times New Roman",
          color: "333333",
        }),
      ],
    }),
  );

  // --- NUMBERED SECTIONS ---
  const sortedSections = [...(content.sections ?? [])].sort(
    (a, b) => a.order - b.order,
  );

  for (let index = 0; index < sortedSections.length; index++) {
    const section = sortedSections[index];
    const sectionNum = index + 1;
    const cleanTitle = sanitizeText(section.title || `Section ${sectionNum}`).toUpperCase();
    const sectionBody = section.content || (section as unknown as { body?: string }).body || "";
    const cleanContent = sanitizeText(sectionBody);

    // Section Heading
    children.push(
      new Paragraph({
        text: `${index + 1}. ${cleanTitle}`,
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 240, after: 120 },
      }),
    );

    // Section Content Paragraphs
    const paragraphs = cleanContent.split(/\n+/);
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
                font: "Times New Roman",
                color: "111111",
              }),
            ],
          }),
        );
      }
    }
  }

  // --- SIGNATURES SECTION ---
  children.push(
    new Paragraph({
      spacing: { before: 360, after: 240 },
      children: [
        new TextRun({
          text: "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the Effective Date.",
          bold: true,
          size: 20,
          font: "Times New Roman",
        }),
      ],
    }),
  );

  const signatureTable = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          // CLIENT Signature Cell
          new TableCell({
            width: { size: 50, type: WidthType.PERCENTAGE },
            borders: {
              top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
            },
            children: [
              new Paragraph({
                spacing: { after: 80 },
                children: [
                  new TextRun({
                    text: "CLIENT:",
                    bold: true,
                    size: 19,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: `Name: ${content.parties.clientName}`,
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "Signature: __________________________",
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "Date: _____________________________",
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
            ],
          }),
          // FREELANCER Signature Cell
          new TableCell({
            width: { size: 50, type: WidthType.PERCENTAGE },
            borders: {
              top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
              right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
            },
            children: [
              new Paragraph({
                spacing: { after: 80 },
                children: [
                  new TextRun({
                    text: "FREELANCER:",
                    bold: true,
                    size: 19,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: `Name: ${content.parties.freelancerName}`,
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "Signature: __________________________",
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({
                    text: "Date: _____________________________",
                    size: 18,
                    font: "Times New Roman",
                  }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });

  children.push(signatureTable);

  // --- DISCLAIMER BOX ---
  const disclaimerText =
    content.disclaimer ||
    "This document is an AI-generated draft provided for informational purposes and is not a substitute for professional legal advice.";

  const disclaimerTable = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 100, type: WidthType.PERCENTAGE },
            shading: { fill: "F9FAFB" },
            margins: { top: 120, bottom: 120, left: 180, right: 180 },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
              bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
              left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
              right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
            },
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 40 },
                children: [
                  new TextRun({
                    text: "AI-GENERATED DOCUMENT DISCLAIMER",
                    bold: true,
                    size: 16,
                    color: "555555",
                    font: "Times New Roman",
                  }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({
                    text: disclaimerText,
                    italics: true,
                    size: 15,
                    color: "666666",
                    font: "Times New Roman",
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
                    text: "FREELANCE SERVICE AGREEMENT",
                    size: 15,
                    color: "777777",
                    font: "Times New Roman",
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
                    color: "555555",
                    font: "Times New Roman",
                  }),
                  new TextRun({
                    children: [PageNumber.CURRENT],
                    size: 16,
                    color: "555555",
                    font: "Times New Roman",
                  }),
                  new TextRun({
                    text: " of ",
                    size: 16,
                    color: "555555",
                    font: "Times New Roman",
                  }),
                  new TextRun({
                    children: [PageNumber.TOTAL_PAGES],
                    size: 16,
                    color: "555555",
                    font: "Times New Roman",
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
