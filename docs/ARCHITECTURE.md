# LegaLese — System Architecture

This document describes the **current** LegaLese architecture as implemented in the repository. It reflects the existing codebase, not the planned SIH 2026 end state.

---

## Overview

LegaLese is a **full-stack Next.js application** using the App Router. There is no separate backend microservice. Business logic lives in **server actions** and **server components**, with selective **client components** for interactive UI (upload, analysis triggers, filters).

```
┌──────────────────────────────────────────────────────────────┐
│                         FRONTEND                              │
│  Next.js App Router + React 19 + Tailwind CSS 4              │
│  Server Components (data fetch) + Client Components (forms)   │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      APPLICATION LAYER                        │
│  middleware.ts          Session refresh, route guards           │
│  lib/actions/*          Server actions (auth, docs, AI, reports)│
│  lib/documents/*        Validation, extraction, processing    │
│  lib/ai/*               Gemini client, prompts, Zod schema    │
└────────────┬───────────────────────────────┬───────────────────┘
             │                               │
┌────────────▼────────────┐    ┌─────────────▼──────────────────┐
│   Supabase (BaaS)       │    │   Google Gemini API            │
│   Auth + Postgres +     │    │   Server-only API key          │
│   Private Storage       │    │   JSON structured responses    │
└─────────────────────────┘    └────────────────────────────────┘
```

---

## Frontend

### Routing

| Route | Type | Purpose |
|-------|------|---------|
| `/` | Static | Public landing page |
| `/login` | Dynamic | Sign in |
| `/signup` | Static | Sign up |
| `/dashboard` | Dynamic | Upload, document list, inline analysis |
| `/dashboard/documents/[id]` | Dynamic | Full analysis workspace |
| `/auth/callback` | Route handler | Supabase auth code exchange |

### Rendering strategy
- **Server Components** fetch documents, analyses, and user session on dashboard pages.
- **Client Components** handle uploads, delete confirmations, analyze/process buttons, search/filter, report download.
- **`dynamic = "force-dynamic"`** on dashboard routes due to cookie-based auth.

### UI organization
```
components/
  auth/           AuthForm, SignOutButton
  dashboard/      ContractUpload, DocumentList, DocumentCard,
                  AnalysisPanel, ContractDetailWorkspace,
                  DetailHeader, FindingCard, SearchFilterBar
  [landing]       Header, Hero, Footer, AnalyzeSection, Button
```

### Styling
- Tailwind CSS 4 with CSS variables in `app/globals.css` (slate/navy legal-tech palette).
- No component library (no shadcn, MUI, etc.).

---

## Backend

LegaLese does not expose a REST API for core features. The backend is **Next.js server actions**:

| Module | File | Responsibilities |
|--------|------|------------------|
| Auth | `lib/actions/auth.ts` | signUp, signIn, signOut |
| Documents | `lib/actions/documents.ts` | uploadDocument, processDocument, deleteDocument |
| Analyses | `lib/actions/analyses.ts` | analyzeDocument, getAnalysis |
| Reports | `lib/actions/reports.ts` | generateReport, downloadReport, getReport |

**Single route handler:** `app/auth/callback/route.ts` for OAuth/email confirmation.

### Middleware
`middleware.ts` → `lib/supabase/middleware.ts`:
- Refreshes Supabase session cookies on each matched request.
- Redirects unauthenticated users away from `/dashboard/*`.
- Redirects authenticated users away from `/login` and `/signup`.

### Request size
`next.config.ts` sets `serverActions.bodySizeLimit: "55mb"` to support 50 MB contract uploads.

---

## Database

**Provider:** Supabase Postgres  
**Access:** `@supabase/ssr` server client with user JWT (RLS enforced)

### Entity relationship (simplified)

```
auth.users
    │
    ├── documents (user_id)
    │       ├── analyses (document_id, user_id)
    │       ├── clauses (document_id)
    │       ├── findings (document_id, clause_id?)
    │       ├── key_terms (document_id, source_clause_id?)
    │       ├── obligations (document_id, source_clause_id?)
    │       └── reports (document_id, user_id)
    │
    └── [Storage objects referenced by storage_path / file_path]
```

### Key tables

| Table | Primary use |
|-------|-------------|
| `documents` | Uploaded contract metadata and processing status |
| `analyses` | AI run metadata, summary, risk_score, full JSON result |
| `clauses` | Verbatim extracted clauses with section/page |
| `findings` | Risk items with optional clause FK and questions JSONB |
| `key_terms` | Defined terms linked to source clauses |
| `obligations` | Duties/deadlines linked to source clauses |
| `reports` | Pointer to generated Markdown report in storage |

Schema reference (do not execute): `supabase/migrations/001_create_analyses_schema.sql`

---

## AI / LLM Layer

**Provider:** Google Gemini via `@google/generative-ai`  
**Entry point:** `lib/ai/gemini.ts` (marked `server-only`)

### Flow
1. Load `ProcessedDocument` JSON from storage.
2. Truncate text if over `MAX_ANALYSIS_CHARS`.
3. Send system instruction + contract text to Gemini with `responseMimeType: application/json`.
4. Parse and validate response with Zod (`lib/ai/schema.ts`).
5. Persist structured rows to Postgres.
6. Compute `risk_score` deterministically (`lib/ai/scorer.ts`).

### Prompt design (`lib/ai/prompt.ts`)
- Role: contract **understanding** assistant, not legal advice.
- Output: single JSON object with `summary`, `clauses`, `findings`, `keyTerms`, `obligations`.
- Findings link to clauses via zero-based `clauseIndex`.
- AI does **not** compute overall risk score.

### Configuration
| Env var | Purpose |
|---------|---------|
| `GEMINI_API_KEY` | Server-only API authentication |
| `GEMINI_MODEL` | Model selection (default: `gemini-3.6-flash`) |
| `MAX_ANALYSIS_CHARS` | Truncation limit for LLM input |

### Not present
- No embedding/vector search.
- No multi-step agent workflows.
- No document **generation** prompts.
- No fine-tuned or RAG-augmented models.

---

## Document Processing

### Supported formats
| Type | Library | Page awareness |
|------|---------|----------------|
| PDF | unpdf | Yes (per-page sections) |
| DOCX | mammoth | No (paragraph sections) |
| TXT | native Buffer | No (paragraph sections) |

### Pipeline (`lib/documents/processor.ts`)
1. Detect document type from `document_type` or extension.
2. Extract text via format-specific extractor.
3. Clean/normalize (`lib/documents/cleaner.ts`).
4. Build `ProcessedDocument` with sections, fullText, wordCount.
5. Store as `{storage_path}.extracted.json` in private bucket.

### Validation (`lib/documents/validation.ts`)
- Max size: 50 MB.
- Allowed extensions: `.pdf`, `.docx`, `.txt`.
- Client-side pre-validation in `ContractUpload.tsx`; server re-validates in `uploadDocument`.

### Status lifecycle
```
uploaded → processing → complete
                    ↘ failed (retry available)
```

---

## Authentication

**Provider:** Supabase Auth (email/password)

### Client architecture
| File | Context |
|------|---------|
| `lib/supabase/client.ts` | Browser client (public env vars) |
| `lib/supabase/server.ts` | Server client (cookie read/write) |
| `lib/supabase/middleware.ts` | Edge session refresh |

### Session model
- HTTP-only cookies managed by `@supabase/ssr`.
- `supabase.auth.getUser()` used in server actions (not session object alone).
- No custom JWT handling in application code.

### Protected resources
- All `/dashboard/*` routes (middleware + page-level redirect).
- All server actions verify `getUser()` before mutations/queries.

---

## Storage

**Bucket:** `contracts` (private, not public)

### Object types and paths
| Artifact | Path pattern | Content-Type |
|----------|--------------|--------------|
| Original contract | `{user_id}/{uuid}.{ext}` | pdf/docx/txt |
| Extraction JSON | `{user_id}/{uuid}.{ext}.extracted.json` | application/json |
| Analysis report | `{user_id}/reports/{document_id}_report.md` | text/plain |

### Access pattern
- **Upload:** server action with authenticated Supabase client.
- **Download:** server-side `storage.download()` — never public URLs.
- **Delete:** removes original, extraction artifact, and report path on document delete.

### Security
- User-scoped paths prefixed with `user_id`.
- RLS/storage policies assumed on Supabase side.
- No service-role key in application code.

---

## External APIs

| API | Direction | Used for |
|-----|-----------|----------|
| Supabase Auth | Outbound | User authentication |
| Supabase Postgres | Outbound | All structured data |
| Supabase Storage | Outbound | Contract files and artifacts |
| Google Gemini | Outbound | Contract analysis JSON generation |

**No other external integrations** (no Stripe, email provider, lawyer marketplace, vector DB, OCR service, etc.).

---

## Data Flow — Upload to Analysis (Primary User Journey)

```
1. User → ContractUpload (client)
2. Client validates file type/size
3. uploadDocument (server action)
      → Storage: contracts/{userId}/{uuid}.pdf
      → DB: documents row (status: uploaded)
      → processDocumentInternal (automatic)
            → status: processing
            → extract text
            → Storage: ...extracted.json
            → status: complete
4. User → Analyze button on DocumentCard
5. analyzeDocument (server action)
      → Download extracted.json
      → Gemini API → validated JSON
      → Insert clauses, findings, key_terms, obligations, analyses
6. UI → AnalysisPanel (inline) or /dashboard/documents/[id] (full workspace)
7. Optional → generateReport → Markdown in storage + reports row
```

---

## Deployment (Current State)

- Standard Next.js build: `npm run build` / `npm run start`.
- Environment variables must be set on host (see `.env.example`).
- No infrastructure-as-code or CI pipeline in repository.
- Telemetry: Next.js anonymous telemetry (opt-out available).

---

## Extension Points for SIH Prototype

These are natural seams for adding features **without rewriting** the existing app:

1. **New routes under `app/dashboard/`** for CREATE and EXPERT flows.
2. **New server actions** in `lib/actions/` for generation, escalation, retrieval.
3. **Prompt modules** in `lib/ai/` for generation vs analysis vs review.
4. **New tables** (e.g. `lawyer_requests`, `generated_documents`, `experience_cases`) alongside existing schema.
5. **Reuse** upload/storage/validation for any new document types.
6. **Extend** `ContractDetailWorkspace` for side-by-side traceability viewer.

---

## Security Summary

| Control | Implementation |
|---------|----------------|
| Auth required for dashboard | Middleware + server action checks |
| RLS | Supabase policies (assumed on remote) |
| Private files | No public bucket; server-side download only |
| AI key isolation | `server-only` + no `NEXT_PUBLIC_` prefix |
| Input validation | Client + server file validation; Zod for AI output |
| Path scoping | Storage paths include `user_id` |

**Pending verification:** Multi-user cross-tenant access denial should be tested manually against live Supabase policies.
