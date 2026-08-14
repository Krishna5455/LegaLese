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
app/           Next.js App Router pages and layouts
components/    Reusable UI components
public/        Static assets
```

## Status

This is Phase 1: project foundation and landing page only. Authentication, Supabase, document processing, and AI integrations are not included yet.
