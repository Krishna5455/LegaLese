import { ButtonLink } from "@/components/Button";

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-border py-24 sm:py-32">
      {/* Background glow gradient */}
      <div className="absolute left-1/2 top-0 -z-10 h-[300px] w-[600px] -translate-x-1/2 bg-accent/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="mx-auto max-w-5xl px-6 text-center">
        <div className="mx-auto mb-6 flex w-fit items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3.5 py-1 text-xs font-semibold text-accent">
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
          COMMERCIAL LEGAL-TECH AI PLATFORM
        </div>

        <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight text-foreground sm:text-6xl sm:leading-tight">
          Understand legal contracts <br />
          <span className="text-accent">before you sign.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base text-muted sm:text-lg leading-relaxed">
          LegaLese transforms complex legal agreements into clear, evidence-backed insights. Spot high-risk clauses, obligations, and defined terms instantly—with zero legal jargon.
        </p>

        <div className="mt-10 flex items-center justify-center gap-4">
          <ButtonLink href="/signup" variant="primary" className="px-6 py-3 text-sm">
            Start Free Contract Review
          </ButtonLink>
          <ButtonLink href="/login" variant="outline" className="px-6 py-3 text-sm">
            Sign In to Dashboard
          </ButtonLink>
        </div>

        {/* Product Trust Badges */}
        <div className="mt-16 flex flex-wrap items-center justify-center gap-8 border-t border-border/60 pt-8 text-xs font-medium text-subtle">
          <div className="flex items-center gap-2">
            <svg className="h-4 w-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Supabase RLS Isolated
          </div>
          <div className="flex items-center gap-2">
            <svg className="h-4 w-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Private Storage Buckets
          </div>
          <div className="flex items-center gap-2">
            <svg className="h-4 w-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Gemini 3.6 Flash Engine
          </div>
          <div className="flex items-center gap-2">
            <svg className="h-4 w-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Zero Model Training on Data
          </div>
        </div>
      </div>
    </section>
  );
}
