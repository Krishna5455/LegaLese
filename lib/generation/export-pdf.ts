import PDFDocument from "pdfkit";
import type { GeneratedDocumentContent } from "@/types/generation";

export async function exportPdf(
  content: GeneratedDocumentContent,
  createdAt?: string,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({
        size: "LETTER",
        margin: 54, // 0.75 inch
        bufferPages: true,
        info: {
          Title: content.title || "Legal Agreement",
          Author: "LegaLese",
          Subject: "Legal Agreement Draft",
        },
      });

      const chunks: Buffer[] = [];
      doc.on("data", (chunk: Buffer) => chunks.push(chunk));
      doc.on("end", () => resolve(Buffer.concat(chunks)));
      doc.on("error", (err: Error) => reject(err));

      const fontTitle = "Helvetica-Bold";
      const fontHeader = "Helvetica-Bold";
      const fontBody = "Helvetica";
      const fontOblique = "Helvetica-Oblique";

      // --- DOCUMENT TITLE ---
      doc
        .font(fontTitle)
        .fontSize(18)
        .fillColor("#111827")
        .text(content.title.toUpperCase(), { align: "center" });

      doc.moveDown(0.5);

      // Subtitle / Document Type & Date
      doc
        .font(fontBody)
        .fontSize(9)
        .fillColor("#6B7280")
        .text("FREELANCE SERVICE AGREEMENT", { align: "center" });

      if (createdAt) {
        const formattedDate = new Intl.DateTimeFormat("en-US", {
          dateStyle: "full",
        }).format(new Date(createdAt));
        doc.text(`Generated on ${formattedDate}`, { align: "center" });
      }

      doc.moveDown(1);

      // Divider line
      doc
        .moveTo(54, doc.y)
        .lineTo(doc.page.width - 54, doc.y)
        .strokeColor("#E5E7EB")
        .lineWidth(1)
        .stroke();

      doc.moveDown(1.2);

      // --- PARTIES SECTION ---
      doc
        .font(fontHeader)
        .fontSize(12)
        .fillColor("#1F2937")
        .text("PARTIES TO THE AGREEMENT");

      doc.moveDown(0.4);

      doc
        .font(fontBody)
        .fontSize(10)
        .fillColor("#374151");

      doc.text(`• Freelancer (Service Provider): ${content.parties.freelancerName}`);
      doc.text(`• Client: ${content.parties.clientName}`);
      if (content.parties.clientAddress) {
        doc.text(`• Client Address: ${content.parties.clientAddress}`);
      }

      doc.moveDown(1.2);

      // --- AGREEMENT SECTIONS ---
      const sortedSections = [...(content.sections ?? [])].sort(
        (a, b) => a.order - b.order,
      );

      for (let index = 0; index < sortedSections.length; index++) {
        const section = sortedSections[index];

        // Section Title
        doc
          .font(fontHeader)
          .fontSize(11)
          .fillColor("#111827")
          .text(`${index + 1}. ${section.title.toUpperCase()}`);

        doc.moveDown(0.4);

        // Section Content
        doc
          .font(fontBody)
          .fontSize(9.5)
          .fillColor("#374151")
          .text(section.content, {
            align: "justify",
            lineGap: 3,
          });

        doc.moveDown(1);
      }

      // --- LEGAL DISCLAIMER ---
      doc.moveDown(1);

      const disclaimerText =
        content.disclaimer ||
        "AI-generated draft. This document is provided for informational purposes and is not a substitute for professional legal advice.";

      // Measure disclaimer text height
      doc.fontSize(8.5);
      const disclaimerHeight = doc.heightOfString(disclaimerText, {
        width: doc.page.width - 138,
      });

      const boxY = doc.y;
      // Check if disclaimer fits on current page, add page if necessary
      if (boxY + disclaimerHeight + 40 > doc.page.height - 54) {
        doc.addPage();
      }

      const finalBoxY = doc.y;
      const boxWidth = doc.page.width - 108;
      const boxHeight = disclaimerHeight + 24;

      // Draw light background box
      doc
        .rect(54, finalBoxY, boxWidth, boxHeight)
        .fillColor("#F9FAFB")
        .strokeColor("#D1D5DB")
        .fillAndStroke();

      doc
        .font(fontHeader)
        .fontSize(8.5)
        .fillColor("#B45309")
        .text("LEGAL DISCLAIMER", 66, finalBoxY + 8, { width: boxWidth - 24 });

      doc
        .font(fontOblique)
        .fontSize(8)
        .fillColor("#4B5563")
        .text(disclaimerText, 66, finalBoxY + 20, {
          width: boxWidth - 24,
          lineGap: 2,
        });

      // --- PAGE NUMBERS & HEADERS ---
      const range = doc.bufferedPageRange();
      for (let i = range.start; i < range.start + range.count; i++) {
        doc.switchToPage(i);

        // Top Header
        doc
          .font(fontBody)
          .fontSize(7.5)
          .fillColor("#9CA3AF")
          .text("LegaLese | Legal Document Draft", 54, 24, {
            width: doc.page.width - 108,
            align: "right",
          });

        // Bottom Footer (Page X of Y)
        doc
          .font(fontBody)
          .fontSize(8)
          .fillColor("#6B7280")
          .text(
            `Page ${i + 1} of ${range.count}`,
            54,
            doc.page.height - 36,
            {
              width: doc.page.width - 108,
              align: "center",
            },
          );
      }

      doc.end();
    } catch (err) {
      reject(err);
    }
  });
}
