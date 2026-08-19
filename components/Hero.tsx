import { ButtonLink } from "@/components/Button";

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-border py-24 sm:py-32 bg-surface">
      {/* Background glow gradient */}
      <div className="absolute left-1/2 top-0 -z-10 h-[350px] w-[700px] -translate-x-1/2 bg-accent/10 blur-[130px] rounded-full pointer-events-none" />

      <div className="mx-auto max-w-5xl px-6 text-center">
        <div className="mx-auto mb-6 flex w-fit items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-4 py-1.5 text-xs font-bold text-accent shadow-2xs">
          <span className="h-2 w-2 rounded-full bg-accent animate-pulse" />
          COMMERCIAL LEGAL-TECH AI PLATFORM
        </div>

        <h1 className="mx-auto max-w-4xl text-5xl font-extrabold tracking-tight text-foreground sm:text-6xl sm:leading-tight">
          Understand legal contracts <br />
          <span className="text-accent">before you sign.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg text-muted sm:text-xl leading-relaxed font-normal">
          LegaLese transforms complex legal agreements into clear, actionable insights. Spot high-risk clauses, key obligations, and payment terms instantly—with zero legal jargon.
        </p>

        <div className="mt-10 flex items-center justify-center gap-4">
          <ButtonLink href="/signup" variant="primary" className="h-12 px-7 py-3 text-base font-bold shadow-xs hover:-translate-y-0.5 transition-transform">
            Start Free Contract Review
          </ButtonLink>
          <ButtonLink href="/login" variant="outline" className="h-12 px-7 py-3 text-base font-semibold hover:-translate-y-0.5 transition-transform">
            Sign In to Dashboard
          </ButtonLink>
        </div>
      </div>
    </section>
  );
}
