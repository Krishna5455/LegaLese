# LegaLese

**Understand before you sign.**

LegaLese is a legal technology product that helps people review contracts in plain language. This repository contains the foundation for the web application.

## Tech stack

- [Next.js](https://nextjs.org/) — React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) — Static typing
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first styling
- [ESLint](https://eslint.org/) — Linting via `eslint-config-next`
- [unpdf](https://github.com/unjs/unpdf) & [mammoth](https://github.com/mwilliamson/mammoth.js) — Document parsing

## Prerequisites

- [Node.js](https://nodejs.org/) 20.x or later
- npm 10.x or later
- A Supabase project with the LegaLese schema and private `contracts` Storage bucket provisioned

## Environment variables

Copy the example file and fill in your Supabase project values:

```bash
cp .env.example .env.local
```

Required variables:

| Variable | Description |
| -------- | ----------- |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL (browser-safe) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon/public key (browser-safe) |
| `NEXT_PUBLIC_SITE_URL` | Optional site URL for auth redirects (defaults to `http://localhost:3000`) |

Never commit `.env.local`. Never expose the Supabase service-role key in client code.

## Storage Configuration

The private `contracts` Storage bucket stores raw contract files (`.pdf`, `.docx`, `.txt`) and the server-generated structured extraction artifacts (`<storage_path>.extracted.json`).

To ensure the private `contracts` bucket accepts both raw documents and extracted JSON artifacts, ensure `allowed_mime_types` includes `application/json`:

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
lib/           Supabase clients, document extractors/processors, and server actions
types/         Shared database and processing types
middleware.ts  Session refresh and route protection
```

## Status

- **Phase 1**: Next.js foundation and landing page (COMPLETE)
- **Phase 2**: Supabase authentication and protected dashboard (COMPLETE)
- **Phase 3**: Secure contract upload and document management (COMPLETE)
- **Phase 4**: Document processing pipeline & structured text extraction (COMPLETE)
