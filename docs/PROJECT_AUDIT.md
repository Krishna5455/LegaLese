# LegaLese — Project Technical Audit

**Audit date:** August 2026  
**Purpose:** Baseline assessment of the existing LegaLese codebase before evolving it into the SIH 2026 prototype.  
**Scope:** Read-only inspection of the repository. No application code was modified during this audit.

---

## Executive Summary

LegaLese is a **Next.js 16 App Router** application with **Supabase Auth**, **private contract storage**, **document text extraction**, and **Gemini-powered contract analysis**. The project has progressed through five internal phases (foundation → auth → upload → extraction → AI analysis) and is **strong on the ANALYZE path** for uploaded contracts.

The **CREATE path** (AI document generation, draft review, customization) and **INNOVATION/EXPERT layers** (real-world experience, similar-case retrieval, lawyer escalation) are **not implemented**. The landing page remains a marketing shell and does not yet expose the full product surface.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (React 19)                       │
│  Landing │ Login/Signup │ Dashboard │ Document Detail Workspace │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    Server Actions + Server Components
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  Supabase Auth          Supabase Postgres         Supabase Storage
  (session/cookies)      (RLS-enforced tables)     (private `contracts`)
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                    Document Processing Pipeline
                    (PDF/DOCX/TXT → JSON artifact)
                                │
                         Gemini API (server-only)
                    (structured JSON analysis output)
```

**Pattern:** Monolithic Next.js full-stack app. No separate backend service. Security-sensitive operations run in **server actions** using the Supabase **anon key + user session** (not service role).

---

## Current Tech Stack

| Layer | Technology | Version (approx.) |
|-------|------------|-------------------|
| Framework | Next.js (App Router) | 16.3.x |
| UI | React | 19.2.x |
| Language | TypeScript | 5.8.x |
| Styling | Tailwind CSS | 4.3.x |
| Linting | ESLint + eslint-config-next | 9.x / 16.x |
| Auth & DB | Supabase (`@supabase/ssr`, `@supabase/supabase-js`) | 0.12.x / 2.11x |
| AI | Google Gemini (`@google/generative-ai`) | 0.24.x |
| PDF extraction | unpdf | 1.8.x |
| DOCX extraction | mammoth | 1.12.x |
| Validation | Zod | 4.4.x |

---

## Project Structure

```
app/
  page.tsx                    # Public landing page
  layout.tsx                  # Root layout (Geist fonts, metadata)
  login/ signup/              # Auth pages
  auth/callback/route.ts      # Supabase OAuth/email callback
  dashboard/
    page.tsx                  # Main authenticated workspace
    documents/[id]/page.tsx   # Per-document analysis detail view

components/
  auth/                       # AuthForm, SignOutButton
  dashboard/                  # Upload, list, analysis UI
  Header, Hero, Footer, ...   # Landing page components

lib/
  actions/                    # auth, documents, analyses, reports
  ai/                         # gemini, prompt, schema, scorer
  documents/                  # processor, validation, extractors
  supabase/                   # client, server, middleware helpers
  env.ts                      # Public env validation

types/
  database.ts                 # Document row type
  analysis.ts                 # Analysis/clause/finding types
  processing.ts               # Extraction pipeline types

supabase/migrations/          # Reference-only schema documentation
middleware.ts                 # Session refresh + route protection
```

---

## Existing Functionality

### Public / Marketing
- Landing page with brand, tagline, product explanation, and CTA anchor.
- Sign in / Sign up links in header.

### Authentication (Phase 2 — complete)
- Email/password sign up, sign in, sign out.
- Cookie-based session persistence via `@supabase/ssr`.
- Middleware protects `/dashboard/*`; redirects unauthenticated users to `/login`.
- Auth callback route for email confirmation flows.

### Document Upload & Management (Phase 3 — complete)
- Drag-and-drop / file picker upload on dashboard.
- Client + server validation: PDF, DOCX, TXT; max 50 MB.
- Private storage path: `{user_id}/{uuid}{extension}`.
- Database row in `documents` with metadata and `status = "uploaded"`.
- Storage cleanup on failed DB insert; delete removes storage artifacts + DB row.
- Document list with filename, type, size, date, status badges.

### Document Processing (Phase 4 — complete)
- Automatic extraction triggered after upload.
- Manual Process / Retry for `uploaded` or `failed` documents.
- Extractors: PDF (unpdf, page-aware), DOCX (mammoth), TXT.
- Text cleaning/normalization pipeline.
- Extracted JSON artifact stored at `{storage_path}.extracted.json`.
- Status lifecycle: `uploaded` → `processing` → `complete` | `failed`.
- Scanned/password-protected PDF detection with user-facing errors.

### AI Contract Analysis (Phase 5 — complete)
- Gemini analysis via server-only `GEMINI_API_KEY`.
- Structured JSON output validated with Zod.
- Persists to: `analyses`, `clauses`, `findings`, `key_terms`, `obligations`.
- Deterministic risk score (0–3) computed in app code, not by AI.
- Inline quick-view analysis panel on dashboard.
- Full analysis workspace at `/dashboard/documents/[id]`.
- Search/filter by risk level and text across findings/clauses/terms/obligations.

### Report Export (partial product feature)
- Generate Markdown analysis report → private storage + `reports` table.
- Download report via authenticated server action (no public URLs).
- Copy aggregated review questions to clipboard.

---

## Current AI Workflow

1. User uploads contract → stored in private `contracts` bucket.
2. Text extraction runs → `ProcessedDocument` JSON artifact saved.
3. User clicks **Analyze** (or analysis auto-eligible when status is `complete`).
4. Server loads extracted JSON from storage.
5. Text truncated if over `MAX_ANALYSIS_CHARS` (default 200,000).
6. Gemini receives:
   - **System instruction** (`lib/ai/prompt.ts`): contract understanding assistant, no legal advice, JSON-only output schema.
   - **User message**: document metadata + full contract text.
7. Response parsed as JSON and validated against `AIAnalysisOutputSchema`.
8. Application inserts clauses first, maps `clauseIndex` → DB IDs, then inserts findings/key_terms/obligations with FK links.
9. Overall `risk_score` computed from highest finding severity.
10. UI renders summary, findings (with linked clause quotes + page numbers), clauses, terms, obligations.

**Prompt focus:** Analyze existing contract text only. Extract verbatim clauses, link findings to clauses, provide plain-English explanations and lawyer questions. Does **not** generate new legal documents.

---

## Database Structure

### Tables (remote Supabase — reference in `supabase/migrations/001_create_analyses_schema.sql`)

| Table | Purpose |
|-------|---------|
| `documents` | Uploaded file metadata, storage path, status |
| `analyses` | Top-level AI result, summary, risk_score, model, JSONB `result` |
| `clauses` | Extracted clause text, section, clause_number, page_number |
| `findings` | Risk findings linked to clauses; questions as JSONB |
| `key_terms` | Defined terms with optional source clause FK |
| `obligations` | Duties/deadlines with optional source clause FK |
| `reports` | Generated Markdown report file path |

**Security:** RLS and user-ownership policies assumed configured on remote Supabase. Application code uses authenticated user JWT only — no service-role key in codebase.

### Document row fields (in use)
`id`, `user_id`, `filename`, `document_type`, `mime_type`, `size_bytes`, `storage_path`, `status`, `created_at`, `updated_at`

### Document status values (observed in code)
`uploaded`, `processing`, `complete`, `failed`

---

## Document Processing Pipeline

```
Upload (FormData)
  → validateDocumentFile() [client + server]
  → storage.upload(userId/uuid.ext)
  → documents.insert(status: uploaded)
  → processDocumentInternal()
       → status: processing
       → processDocumentBuffer()
            → PDF | DOCX | TXT extractor
            → cleanDocumentText()
       → storage.upload(userId/uuid.ext.extracted.json)
       → status: complete | failed
```

---

## Existing Integrations

| Integration | Usage | Exposure |
|-------------|-------|----------|
| Supabase Auth | Sign up/in/out, session | Public URL + anon key (client-safe) |
| Supabase Postgres | All app data | Server + RLS-scoped client |
| Supabase Storage | Contracts, extraction JSON, reports | Private bucket, authenticated download |
| Google Gemini | Contract analysis | **Server-only** `GEMINI_API_KEY` |

### Environment Variables

| Variable | Required | Notes |
|----------|----------|-------|
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Browser-safe |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Browser-safe |
| `NEXT_PUBLIC_SITE_URL` | Optional | Auth redirect URLs |
| `GEMINI_API_KEY` | Yes (for analysis) | Server-only |
| `GEMINI_MODEL` | Optional | Default `gemini-3.6-flash` |
| `MAX_ANALYSIS_CHARS` | Optional | Default 200000 |

---

## Deployment Configuration

- **No** `vercel.json`, Docker, or CI/CD configuration found in repository.
- **No** dedicated API route layer beyond `app/auth/callback/route.ts`.
- Production deployment likely manual (Vercel or similar) with env vars configured in host dashboard.
- `next.config.ts` sets `serverActions.bodySizeLimit: "55mb"` for large uploads.

---

## Bugs, Gaps, and Incomplete Functionality

| Issue | Severity | Notes |
|-------|----------|-------|
| No original contract download/preview in UI | Medium | Storage download exists server-side for processing; users cannot view source PDF in app |
| No PDF page viewer / highlight traceability | Medium | Page numbers stored but no visual jump-to-page in original document |
| Analysis partial-write on failure | Medium | If clause insert succeeds but findings insert fails, orphaned clause rows may remain (no transaction rollback) |
| Single analysis per document | Low | Re-analysis not supported; existing analysis returned |
| Scanned PDFs rejected | Expected | By design; OCR not implemented |
| `GEMINI_MODEL` default may be invalid | Medium | Default `gemini-3.6-flash` depends on Google API availability; misconfiguration causes runtime failures |
| Next.js 16 middleware deprecation warning | Low | Build warns to migrate `middleware.ts` → `proxy` |
| Landing CTA still points to `#analyze` placeholder | Low | Marketing page not wired to dashboard upload flow |
| No automated tests | High | No unit, integration, or E2E test suite |
| Cross-user access not verified in repo | High | Security relies on Supabase RLS; manual multi-user testing pending |
| Delete may leave analysis child rows | Low | Depends on DB `ON DELETE CASCADE`; app deletes `documents` row and storage files explicitly |
| No lawyer escalation workflow | N/A | Out of scope for current build |
| No document generation | N/A | CREATE features not started |

---

## Technical Risks

1. **AI reliability:** Gemini output quality and JSON conformance vary; 30s timeout may fail on long contracts.
2. **Cost/latency:** Full-document analysis sent to LLM; truncation may miss tail clauses.
3. **No observability:** Console logging only; no structured logging, error tracking, or analytics.
4. **Monolithic server actions:** Harder to scale extraction/analysis independently for SIH demo load.
5. **RLS as sole authorization:** Correct pattern, but untested cross-tenant scenarios in application QA.
6. **Secret management:** Gemini key must stay server-only; any future client-side AI would be a regression.

---

## Recommendations (for SIH evolution — not implemented in this audit)

1. **Preserve** existing upload → extract → analyze pipeline; it is the strongest reusable asset.
2. **Add** CREATE module as a separate route/workflow without rewriting dashboard analysis.
3. **Implement** original-document viewer (authenticated signed URL or server proxy) for traceability demo.
4. **Add** lawyer escalation as a new table + UI action (request queue, not email-only).
5. **Introduce** experience/similarity layer as retrieval over stored analyses or a curated corpus (RAG).
6. **Add** E2E tests for auth, upload, analysis, and cross-user denial.
7. **Validate** `GEMINI_MODEL` against available models and document in `.env.example`.
8. **Wire** landing page CTA to `/signup` or `/dashboard` for demo flow.
9. **Consider** background job queue for extraction/analysis if demo documents are large.
10. **Do not** introduce service-role key for user operations.

---

## Audit Limitations

- Database RLS policies were not inspected live (only referenced in README and migration docs).
- Supabase Storage bucket policies were not verified against remote project.
- Gemini analysis was not executed during this audit (requires live API key and sample document).
- Cross-user security was assessed from code patterns only; manual verification remains pending.
