# LegaLese

**Understand before you sign.**

LegaLese is a legal technology product that helps people review contracts in plain language. This repository contains the foundation for the web application.

## Tech stack

- [Next.js](https://nextjs.org/) — React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) — Static typing
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first styling
- [ESLint](https://eslint.org/) — Linting via `eslint-config-next`

## Prerequisites

- [Node.js](https://nodejs.org/) 20.x or later
- npm 10.x or later
- A Supabase project with the LegaLese schema already provisioned

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
lib/           Supabase clients, env helpers, and server actions
types/         Shared database types
middleware.ts  Session refresh and route protection
```

## Authentication

Phase 2 adds Supabase Auth with:

- `/login` — sign in
- `/signup` — create account
- `/dashboard` — protected user dashboard
- `/auth/callback` — OAuth/email confirmation callback

Unauthenticated users are redirected from `/dashboard` to `/login`.

## Status

Phase 2: Supabase authentication and protected dashboard. Document upload, storage integration, and AI analysis are not included yet.
