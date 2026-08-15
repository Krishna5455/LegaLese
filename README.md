# LegaLese

**Understand before you sign.**

LegaLese is a legal technology product that helps people review contracts in plain language. This repository contains the foundation for the web application.

## Tech stack

- [Next.js](https://nextjs.org/) — React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) — Static typing
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first styling
- [ESLint](https://eslint.org/) — Linting via `eslint-config-next`
- [unpdf](https://github.com/unjs/unpdf) & [mammoth](https://github.com/mwilliamson/mammoth.js) — Document parsing
- [Zod](https://zod.dev/) — Runtime schema validation
- [@google/generative-ai](https://www.npmjs.com/package/@google/generative-ai) — Gemini AI integration

## Prerequisites

- [Node.js](https://nodejs.org/) 20.x or later
- npm 10.x or later
- A Supabase project with the LegaLese schema and private `contracts` Storage bucket provisioned
- A Google Gemini API key (obtain from [Google AI Studio](https://aistudio.google.com/app/apikey))

## Environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env.local
```

Required variables:

| Variable | Description |
| -------- | ----------- |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL (browser-safe) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon/public key (browser-safe) |
| `NEXT_PUBLIC_SITE_URL` | Optional site URL for auth redirects (defaults to `http://localhost:3000`) |
| `GEMINI_API_KEY` | **Server-only.** Gemini API key. Never use `NEXT_PUBLIC_` prefix. |
| `GEMINI_MODEL` | Optional. Gemini model name (default: `gemini-2.0-flash`) |
| `MAX_ANALYSIS_CHARS` | Optional. Maximum document characters sent to AI (default: `200000`) |

> **Security note**: Never commit `.env.local`. Never expose `GEMINI_API_KEY` to client code. The AI API key is used only in server actions.

## Database Setup

### Phase 2 — `documents` table
Apply your existing Supabase schema for the `documents` table with RLS policies.

### Phase 5 — Analysis tables
Apply the migration at `supabase/migrations/001_create_analyses_schema.sql` in your Supabase SQL editor:

```sql
-- Run in Supabase Dashboard → SQL Editor
-- File: supabase/migrations/001_create_analyses_schema.sql
```

This creates: `analyses`, `findings`, `key_terms`, `obligations`, `questions` with RLS and indexes.

## Storage Configuration

The private `contracts` Storage bucket stores raw contract files and extraction artifacts.
Ensure `allowed_mime_types` includes `application/json`:

```sql
UPDATE storage.buckets
SET allowed_mime_types = ARRAY[
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'application/json'
]
WHERE id = 'contracts';
```

## Getting started

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available scripts

| Command         | Description                          |
| --------------- | ------------------------------------ |
| `npm run dev`   | Start the development server         |
| `npm run build` | Create a production build            |
| `npm run start` | Serve the production build           |
| `npm run lint`  | Run ESLint                           |

## Project structure

```
app/           Next.js App Router pages, layouts, and auth routes
components/    Reusable UI and auth components
lib/
  actions/     Server actions (auth, documents, analyses)
  ai/          Gemini AI client, prompt builder, Zod schema, risk scorer
  documents/   Document extraction pipeline (PDF/DOCX/TXT)
  supabase/    Supabase client factories
supabase/
  migrations/  SQL migration files for the LegaLese schema
types/         Shared TypeScript types (database, processing, analysis)
middleware.ts  Session refresh and route protection
```

## Status

- **Phase 1**: Next.js foundation and landing page ✅
- **Phase 2**: Supabase authentication and protected dashboard ✅
- **Phase 3**: Secure contract upload and document management ✅
- **Phase 4**: Document processing pipeline & structured text extraction ✅
- **Phase 5**: AI contract analysis with Gemini — evidence-backed findings, key terms, obligations, questions ✅

## Legal disclaimer

LegaLese is a contract **understanding** tool. It is not a law firm and does not provide legal advice. Always consult a qualified legal professional for significant decisions.
