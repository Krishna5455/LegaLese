# LegaLese — AI Legal Document Generator (CREATE Flow)

This document describes the design, user flow, data schemas, AI prompt architecture, database persistence, and export options for the **AI Legal Document Generator** feature in LegaLese.

---

## Overview

The **CREATE** flow enables users to generate customized legal agreements (starting with a **Freelance Service Agreement**) through guided, non-legalese questions. The generated agreement is structured into modular sections with **stable snake_case section IDs** to support planned future features:

1. **AI Explanation of Created Agreements**: Explaining each generated clause in plain English.
2. **AI Draft Review**: Providing redline suggestions or risk flags on user drafts.
3. **Interactive Clause Customization**: Parameter-driven live tweaking of specific sections.

---

## Required User Flow

```
┌──────────────────────────────┐
│  /dashboard/create           │
│  Select Document Type        │
│  (Freelance Agreement)       │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Answer Guided Questions     │
│  (Parties, Work, Payment,    │
│   Termination, IP, Disputes) │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Validate Inputs             │
│  (Client + Server Zod)       │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Generate via Gemini API     │
│  (Server-only, JSON schema)  │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Save Draft to Database      │
│  (`generated_documents` DB)  │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  /dashboard/create/[id]      │
│  Display Agreement & ID Tags │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Download / Copy Markdown    │
│  (.md export or clipboard)   │
└──────────────────────────────┘
```

---

## Technical Architecture

### 1. Client & Page Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/dashboard/create` | `CreateDocumentPage` | Type selector, guided form, recent draft history |
| `/dashboard/create/[id]` | `ViewGeneratedDocumentPage` | Full workspace for viewing and downloading a generated draft |

### 2. Components (`components/create/`)

- `DocumentTypeSelector.tsx`: Renders selectable card list of available document types.
- `FreelanceAgreementForm.tsx`: Guided inputs across 6 logical sections (Parties, Work, Payment, Termination, IP, Disputes).
- `GeneratedDocumentWorkspace.tsx`: Displays draft header, agreed parties, saved badge, section breakdown with visible section IDs, legal disclaimer, copy, and markdown download buttons.

### 3. Server Actions (`lib/actions/generated-documents.ts`)

- `generateFreelanceAgreement(input)`: Validates form input, calls Gemini, inserts draft row into Postgres, returns `documentId`.
- `getGeneratedDocument(documentId)`: Fetches user's generated draft by ID with user authorization.
- `downloadGeneratedDocument(documentId)`: Formats generated content as a clean Markdown document for download.
- `listGeneratedDocuments()`: Fetches all generated document drafts owned by current user.

---

## AI Generation Pipeline (`lib/ai/`)

### Server-Side Security
All AI generation calls run strictly server-side using `GEMINI_API_KEY` loaded via `lib/ai/config.ts`. The API key is never exposed to the client.

### Prompt System (`lib/ai/generation-prompt.ts`)
- **System Instruction**: Enforces legal drafting assistant role, zero hallucination of facts, plain language, disclaimer inclusion, and JSON-only formatting.
- **User Prompt**: Passes all structured form fields (freelancer name, client name, services, deliverables, dates, payment terms, notice period, IP ownership, confidentiality, jurisdiction).

### Stable Section IDs
Gemini is instructed to return sections with stable snake_case IDs in exact order:
1. `parties`
2. `services_and_deliverables`
3. `payment`
4. `term_and_schedule`
5. `termination`
6. `intellectual_property`
7. `confidentiality`
8. `dispute_resolution`
9. `miscellaneous`

### Output Validation Schema (`lib/ai/generation-schema.ts`)
```ts
export const GeneratedSectionSchema = z.object({
  id: z.string().min(1).max(100),
  title: z.string().min(1).max(200),
  content: z.string().min(1).max(15000),
  order: z.number().int().min(0).max(100),
});

export const GeneratedDocumentContentSchema = z.object({
  title: z.string().min(1).max(300),
  documentType: z.literal("freelance_service_agreement"),
  parties: GeneratedDocumentPartiesSchema,
  sections: z.array(GeneratedSectionSchema).min(3).max(20),
  disclaimer: z.string().min(1).max(3000),
});
```

---

## Database Schema (`supabase/migrations/002_create_generated_documents.sql`)

```sql
CREATE TABLE IF NOT EXISTS generated_documents (
  id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  document_type     text        NOT NULL,
  title             text        NOT NULL,
  input_data        jsonb       NOT NULL,
  generated_content jsonb       NOT NULL,
  model             text        NULL,
  status            text        NOT NULL DEFAULT 'draft',
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now()
);
```

**Security Policies (RLS):**
Users can SELECT, INSERT, UPDATE, and DELETE only rows where `auth.uid() = user_id`.

---

## Export Capabilities (`lib/generation/export.ts`)

- **Markdown Export**: Formats agreement title, generated timestamp, parties summary, section headings (`## Title`), section contents, and disclaimer into a clean `.md` file.
- **Copy to Clipboard**: Copies formatted markdown text with instant visual feedback ("✓ Copied to Clipboard").
- **File Download**: Generates downloadable browser blob named `{Safe_Title}.md`.
