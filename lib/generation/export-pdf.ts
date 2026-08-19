import PDFDocument from "pdfkit/js/pdfkit.standalone.js";
import type { GeneratedDocumentContent } from "@/types/generation";

export async function exportPdf(
  content: GeneratedDocumentContent,
  createdAt?: string,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({
        size: "A4", // 595.28 x 841.89 pt
        margin: 54, // 0.75 in
        bufferPages: true,
        info: {
          Title: content.title || "Freelance Service Agreement",
          Author: "LegaLese",
          Subject: "Legal Agreement Draft",
        },
      });

      const chunks: Buffer[] = [];
      doc.on("data", (chunk: Buffer) => chunks.push(chunk));
      doc.on("end", () => resolve(Buffer.concat(chunks)));
      doc.on("error", (err: Error) => reject(err));

      const fontBold = "Times-Bold";
      const fontRegular = "Times-Roman";
      const fontItalic = "Times-Italic";

      const pageWidth = doc.page.width;
      const contentWidth = pageWidth - 108; // 487.28 pt

      // Helper to strip markdown symbols and raw section IDs
      const sanitizeText = (text: string): string => {
        return text
          .replace(/^#+\s*/gm, "")
          .replace(/\*\*(.*?)\*\*/g, "$1")
          .replace(/\*(.*?)\*/g, "$1")
          .replace(/^-\s*/gm, "• ")
          .trim();
      };

      // --- DOCUMENT HEADER / TITLE ---
      const documentTitle = (content.title || "FREELANCE SERVICE AGREEMENT")
        .toUpperCase()
        .replace(/[^A-Z0-9\s.,-]/g, "");

      doc
        .font(fontBold)
        .fontSize(16)
        .fillColor("#000000")
        .text(documentTitle, { align: "center" });

      doc.moveDown(0.8);

      // --- PREAMBLE & PARTIES SECTION ---
      doc
        .font(fontRegular)
        .fontSize(10)
        .fillColor("#111111")
        .text(
          'This Freelance Service Agreement ("Agreement") is entered into by and between the following parties:',
          { align: "left", lineGap: 3 },
        );

      doc.moveDown(0.8);

      // Client & Freelancer Details Box
      const effectiveDateStr = createdAt
        ? new Intl.DateTimeFormat("en-US", { dateStyle: "long" }).format(
            new Date(createdAt),
          )
        : new Intl.DateTimeFormat("en-US", { dateStyle: "long" }).format(new Date());

      const partiesY = doc.y;
      const halfWidth = (contentWidth - 20) / 2;

      // CLIENT Box (Left column)
      doc
        .font(fontBold)
        .fontSize(10)
        .fillColor("#000000")
        .text("CLIENT", 54, partiesY);

      doc
        .font(fontRegular)
        .fontSize(9.5)
        .fillColor("#222222")
        .text(`Name: ${content.parties.clientName}`, 54, doc.y + 2, {
          width: halfWidth,
        });

      if (content.parties.clientAddress) {
        doc.text(`Address: ${content.parties.clientAddress}`, 54, doc.y + 2, {
          width: halfWidth,
        });
      }

      const clientColumnBottom = doc.y;

      // FREELANCER Box (Right column)
      const rightColX = 54 + halfWidth + 20;
      doc
        .font(fontBold)
        .fontSize(10)
        .fillColor("#000000")
        .text("FREELANCER", rightColX, partiesY);

      doc
        .font(fontRegular)
        .fontSize(9.5)
        .fillColor("#222222")
        .text(
          `Name: ${content.parties.freelancerName}`,
          rightColX,
          partiesY + 14,
          { width: halfWidth },
        );

      const freelancerColumnBottom = doc.y;
      const partiesBottomY = Math.max(clientColumnBottom, freelancerColumnBottom);

      doc.y = partiesBottomY + 10;
      doc
        .font(fontItalic)
        .fontSize(9.5)
        .fillColor("#333333")
        .text(`Effective Date: ${effectiveDateStr}`, 54, doc.y);

      doc.moveDown(1);

      // Divider Line
      doc
        .moveTo(54, doc.y)
        .lineTo(pageWidth - 54, doc.y)
        .strokeColor("#999999")
        .lineWidth(0.75)
        .stroke();

      doc.moveDown(1.2);

      // --- NUMBERED SECTIONS ---
      const sortedSections = [...(content.sections ?? [])].sort(
        (a, b) => a.order - b.order,
      );

      for (let index = 0; index < sortedSections.length; index++) {
        const section = sortedSections[index];
        const sectionNum = index + 1;
        const cleanTitle = sanitizeText(section.title).toUpperCase();
        const cleanContent = sanitizeText(section.content);

        // Heading
        doc
          .font(fontBold)
          .fontSize(10.5)
          .fillColor("#000000")
          .text(`${sectionNum}. ${cleanTitle}`);

        doc.moveDown(0.4);

        // Body
        doc
          .font(fontRegular)
          .fontSize(9.5)
          .fillColor("#111111")
          .text(cleanContent, {
            align: "justify",
            lineGap: 3.5,
            paragraphGap: 6,
          });

        doc.moveDown(1.2);
      }

      // --- SIGNATURES SECTION ---
      if (doc.y + 140 > doc.page.height - 70) {
        doc.addPage();
      }

      doc
        .font(fontBold)
        .fontSize(10)
        .fillColor("#000000")
        .text("IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the Effective Date.", {
          align: "left",
          lineGap: 4,
        });

      doc.moveDown(1.5);

      const sigY = doc.y;

      // CLIENT Signature Block
      doc
        .font(fontBold)
        .fontSize(9.5)
        .fillColor("#000000")
        .text("CLIENT:", 54, sigY);

      doc
        .font(fontRegular)
        .fontSize(9)
        .fillColor("#222222")
        .text(`Name: ${content.parties.clientName}`, 54, sigY + 16)
        .text("Signature: __________________________", 54, sigY + 34)
        .text("Date: _____________________________", 54, sigY + 52);

      // FREELANCER Signature Block
      doc
        .font(fontBold)
        .fontSize(9.5)
        .fillColor("#000000")
        .text("FREELANCER:", rightColX, sigY);

      doc
        .font(fontRegular)
        .fontSize(9)
        .fillColor("#222222")
        .text(`Name: ${content.parties.freelancerName}`, rightColX, sigY + 16)
        .text("Signature: __________________________", rightColX, sigY + 34)
        .text("Date: _____________________________", rightColX, sigY + 52);

      doc.y = sigY + 80;

      // --- LEGAL DISCLAIMER FOOTER BLOCK ---
      if (doc.y + 60 > doc.page.height - 60) {
        doc.addPage();
      }

      const disclaimerText =
        content.disclaimer ||
        "This document is an AI-generated draft provided for informational purposes and is not a substitute for professional legal advice.";

      doc
        .moveTo(54, doc.y)
        .lineTo(pageWidth - 54, doc.y)
        .strokeColor("#CCCCCC")
        .lineWidth(0.5)
        .stroke();

      doc.moveDown(0.6);

      doc
        .font(fontBold)
        .fontSize(8)
        .fillColor("#555555")
        .text("AI-GENERATED DOCUMENT DISCLAIMER", 54, doc.y, { align: "center" });

      doc.moveDown(0.3);

      doc
        .font(fontItalic)
        .fontSize(7.5)
        .fillColor("#666666")
        .text(disclaimerText, 54, doc.y, {
          align: "center",
          width: contentWidth,
          lineGap: 2,
        });

      // --- PAGE NUMBERS & HEADER ---
      const range = doc.bufferedPageRange();
      for (let i = range.start; i < range.start + range.count; i++) {
        doc.switchToPage(i);

        // Header (Top right)
        doc
          .font(fontRegular)
          .fontSize(7.5)
          .fillColor("#777777")
          .text("FREELANCE SERVICE AGREEMENT", 54, 24, {
            width: contentWidth,
            align: "right",
            lineBreak: false,
          });

        // Footer (Page X of Y)
        doc
          .font(fontRegular)
          .fontSize(8)
          .fillColor("#555555")
          .text(
            `Page ${i + 1} of ${range.count}`,
            54,
            doc.page.height - 36,
            {
              width: contentWidth,
              align: "center",
              lineBreak: false,
            },
          );
      }

      doc.end();
    } catch (err) {
      reject(err);
    }
  });
}
