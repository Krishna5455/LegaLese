import { ButtonLink } from "@/components/Button";

export function Hero() {
  return (
    <section className="mx-auto flex max-w-5xl flex-1 flex-col justify-center px-6 py-20 sm:py-28">
      <div className="max-w-2xl">
        <p className="mb-4 text-sm font-medium uppercase tracking-widest text-accent">
          Legal technology
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">
          LegaLese
        </h1>
        <p className="mt-4 text-xl text-muted sm:text-2xl">
          Understand before you sign.
        </p>
        <p className="mt-6 text-base leading-relaxed text-muted sm:text-lg">
          LegaLese turns dense contract language into clear, actionable
          summaries. Upload an agreement, review key clauses, and spot risks
          before you commit—without needing a law degree.
        </p>
        <div className="mt-10">
          <ButtonLink href="#analyze">Analyze a Contract</ButtonLink>
        </div>
      </div>
    </section>
  );
}
