# LegaLese — Project Cleanup Report

This report is the result of a comprehensive audit across all directories (`app/`, `components/`, `lib/`, `types/`, `hooks/`, `public/`, configuration, and dependencies) of the LegaLese codebase.

---

## 1. Files Definitely Safe to Remove

The following files were confirmed to have **zero active imports, zero references, and zero production usage** across the entire application:

| File Path | Size | Category | Reason for Removal |
|---|---|---|---|
| `app/design-lab/page.tsx` | 64.6 KB | Scratch / Test Route | Abandoned UI playground. Not linked in navbar, footer, or any application route. Adds 64KB+ to static build. |
| `components/AnalyzeSection.tsx` | 3.0 KB | Obsolete Landing Section | Old landing section superseded by `components/landing/DocumentIntelligence.tsx`. Zero imports. |
| `components/Hero.tsx` | 4.2 KB | Obsolete Hero | Old landing hero superseded by `components/landing/LandingHero.tsx`. Zero imports. |
| `components/ui/DocumentPreview.tsx` | 4.1 KB | Abandoned UI Widget | Old modal/preview card. Zero imports in production code. |
| `components/ui/StatusBadge.tsx` | 2.3 KB | Abandoned UI Widget | Only imported by unused `DocumentPreview.tsx`. Dedicated badges are implemented in `DocumentCard`, `DetailHeader`, and `DocumentReviewView`. |
| `components/ui/Badge.tsx` | 1.1 KB | Unused Primitive | Generic UI badge primitive with zero imports in any component or page. |
| `components/ui/Card.tsx` | 2.3 KB | Unused Primitive | Generic UI card primitive with zero imports in any component or page. |
| `components/ui/PageHeader.tsx` | 1.4 KB | Unused Primitive | Standalone page header with zero imports in any page. |
| `components/ui/SectionHeader.tsx` | 1.2 KB | Unused Primitive | Standalone section header with zero imports in any page. |
| `components/landing/AnalyzePipeline.tsx` | 6.6 KB | Abandoned Section | Experimental landing pipeline variant. Zero imports in `app/page.tsx` or any route. |
| `components/landing/AudienceTrust.tsx` | 5.3 KB | Abandoned Section | Unused landing section variant. Zero imports. |
| `components/landing/FeatureGrid.tsx` | 4.0 KB | Abandoned Section | Unused feature grid variant. Zero imports. |
| `components/landing/ProblemEditorial.tsx` | 7.8 KB | Abandoned Section | Replaced by `ProblemSection.tsx`. Zero imports. |
| `components/landing/ProductExperience.tsx` | 12.3 KB | Abandoned Section | Replaced by `ProductShowcase.tsx` and `UnderstandSection.tsx`. Zero imports. |
| `components/landing/ReviewSection.tsx` | 11.7 KB | Abandoned Section | Experimental landing section variant. Zero imports. |
| `components/landing/ValueStrip.tsx` | 1.3 KB | Abandoned Section | Unused landing banner variant. Zero imports. |

**Total Safe Recoverable Source Footprint:** ~133 KB of dead TypeScript/JSX code across 16 files.

---

## 2. Files Duplicated / Obsolete

1. **`components/Hero.tsx`** vs. **`components/landing/LandingHero.tsx`**:
   `Hero.tsx` is an early prototype with static buttons. `LandingHero.tsx` is the actual production hero with split-text motion, magnet CTA, and legal authority badges. `Hero.tsx` is obsolete.
2. **`components/AnalyzeSection.tsx`** vs. **`components/landing/DocumentIntelligence.tsx`**:
   `AnalyzeSection.tsx` is an early prototype with basic cards. `DocumentIntelligence.tsx` is the active rich component.
3. **`components/landing/ProblemEditorial.tsx`** vs. **`components/landing/ProblemSection.tsx`**:
   Both contain identical theme tokens and similar structure; `ProblemSection.tsx` is the one wired into `app/page.tsx`.
4. **`middleware.ts`**:
   Next.js 16 Turbopack emits a deprecation warning: `"The middleware file convention is deprecated. Please use proxy instead."` Next.js 16 standardizes on `proxy.ts`.

---

## 3. Dependencies Audit

Inspected all packages listed in `package.json`:

```json
{
  "dependencies": {
    "@google/generative-ai": "^0.24.1",    // USED: lib/ai/gemini.ts
    "@supabase/ssr": "^0.12.4",             // USED: lib/supabase/server.ts, middleware.ts, client.ts
    "@supabase/supabase-js": "^2.112.3",    // USED: Supabase database & storage calls
    "@types/pdfkit": "^0.17.6",             // USED: TypeScript types for pdfkit
    "docx": "^9.7.1",                       // USED: lib/generation/export-docx.ts
    "lucide-react": "^1.37.0",              // USED: UI icons across all pages
    "mammoth": "^1.12.1",                   // USED: lib/documents/extractors/docx.ts
    "next": "^16.3.1",                      // REQUIRED: Framework
    "pdfkit": "^0.19.1",                    // USED: lib/generation/export-pdf.ts
    "react": "^19.2.8",                     // REQUIRED: Core library
    "react-dom": "^19.2.8",                 // REQUIRED: Core library
    "unpdf": "^1.8.1",                      // USED: lib/documents/extractors/pdf.ts
    "zod": "^4.4.3"                         // USED: Form validation & AI schema structured outputs
  }
}
```

**Verdict:** There are **zero unused dependencies**. Every single package in `dependencies` is actively imported and utilized by core application logic (PDF/DOCX extraction & export, AI generation, Supabase Auth/DB). No bloated or unneeded packages were added.

---

## 4. Components Convertible from Client → Server Components

Audit of components marked `"use client"` that do not require client-side state, browser APIs, or event handlers:

| Component | Current State | Reason It Can Be Server Component |
|---|---|---|
| `components/landing/HumanStorySection.tsx` | `"use client"` | Pure presentational JSX with static content, Next.js `<Image>`, and `<Link>`. The only animated child is `<ScrollReveal>`, which is already a client component and can wrap Server Component children. |
| `components/landing/EcosystemRoadmap.tsx` | `"use client"` | Pure presentational JSX mapping over static `roadmapItems`. No hooks, no state, no click handlers. |
| `components/landing/FinalCta.tsx` | `"use client"` | Pure layout container wrapping `<Magnet>` and `<ScrollReveal>`. No local state or browser APIs. |

*Note:* `components/dashboard/DocumentList.tsx` is already a Server Component.

---

## 5. Performance Bottlenecks Discovered

### A. N+5 Massive Supabase Query Explosion on `/dashboard`
- **Location:** `app/dashboard/page.tsx` (lines 48–81)
- **Issue:** On every single visit to `/dashboard`, the server executes:
  1. `supabase.auth.getUser()`
  2. `supabase.from("documents").select("*")`
  3. `supabase.from("analyses").select("*").in("document_id", documentIds)...`
  4. `supabase.from("clauses").select("*").in("document_id", documentIds)...`
  5. `supabase.from("findings").select("*").in("document_id", documentIds)...`
  6. `supabase.from("key_terms").select("*").in("document_id", documentIds)...`
  7. `supabase.from("obligations").select("*").in("document_id", documentIds)...`
- **Impact:** For a user with 5+ documents, the server downloads hundreds or thousands of rows of clauses, findings, and key terms across the network and iterates through all of them in JavaScript memory, merely to count risk scores and supply an inline quick-view panel that is closed by default.
- **Remedy:** On `/dashboard`, only query the high-level `analyses` summary fields (`id, document_id, risk_score, summary, created_at`). Deep contract details belong in the dedicated `/dashboard/documents/[id]` workspace.

### B. Lack of Layout Sharing Across `/dashboard` Routes
- **Location:** `app/dashboard/page.tsx`, `app/dashboard/create/page.tsx`, `app/dashboard/create/[id]/page.tsx`, `app/dashboard/documents/[id]/page.tsx`
- **Issue:** There is no `app/dashboard/layout.tsx`. Each route independently renders `<DashboardNav>` and calls `supabase.auth.getUser()`.
- **Impact:** Navigating between dashboard pages destroys and remounts the navigation bar, causing layout shifts, re-running redundant authentication checks, and losing navigation state.
- **Remedy:** Introduce `app/dashboard/layout.tsx` to host `<DashboardNav>` with shared user session state.

### C. Zero Loading Skeletons (`loading.tsx`) Leading to Frozen UI Perception
- **Location:** All routes under `app/dashboard/`
- **Issue:** When a user clicks a link to `/dashboard`, `/dashboard/create`, `/dashboard/create/[id]`, or `/dashboard/documents/[id]`, Next.js waits for the full server render before transitioning. Without a `loading.tsx`, the browser appears unresponsive.
- **Remedy:** Add lightweight `loading.tsx` skeletons for `/dashboard` and subroutes so transitions are visually instant.

### D. Eager Loading of Heavy Interactive Panels
- **Location:** `DocumentCard.tsx` (eagerly imports `AnalysisPanel.tsx`), `GeneratedDocumentWorkspace.tsx` (eagerly imports `DocumentExplanationView.tsx` and `DocumentReviewView.tsx`).
- **Impact:** Initial page bundles load extensive client JavaScript for secondary views that are not immediately visible.
- **Remedy:** Use `next/dynamic` to dynamically import `AnalysisPanel`, `DocumentExplanationView`, and `DocumentReviewView`.

### E. Polling Timer Multiplication
- **Location:** `components/dashboard/DocumentCard.tsx` (line 110)
- **Issue:** Every card with `status === "processing"` runs its own `setInterval(() => router.refresh(), 3000)`. If multiple documents are processing, multiple parallel refreshes flood the server every 3 seconds.
- **Remedy:** Consolidate polling or guard it to ensure a single controlled refresh.

---

## 6. Navigation Bottlenecks Discovered

1. **Anchor Hrefs vs. Page Navigation**:
   In `DashboardNav.tsx`, "My Documents" links to `/dashboard#documents` and "Analyze Contract" links to `/dashboard#upload`. When already on `/dashboard`, clicking them should smoothly scroll to the section without unneeded server queries.
2. **Missing Active Link State in Header & DashboardNav**:
   Navbar links sometimes did not highlight accurately or relied on manual props rather than path matching.
3. **No Route Transition Indicator**:
   Next.js client-side navigations had no micro-progress feedback on slower network connections.

---

## 7. Potential Future Cleanup Items

1. **Consolidate Modals in `Header.tsx`**:
   `components/Header.tsx` implements its "Community Soon" and "Legal Expert Soon" dialogs inline. If more modals are added in the future, consider using an accessible dialog primitive.
2. **Standardize Toast Notifications**:
   If user-facing feedback toasts are needed across create/upload flows, integrate a lightweight toast container rather than component-level alerts.
3. **Migrate `middleware.ts` to `proxy.ts`**:
   Follow Next.js 16 conventions to eliminate the canary deprecation notice.
