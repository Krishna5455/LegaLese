import { Header } from "@/components/Header";
import { LandingHero } from "@/components/landing/LandingHero";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { DocumentIntelligence } from "@/components/landing/DocumentIntelligence";
import { TrustSection } from "@/components/landing/TrustSection";
import { TrackingSection } from "@/components/landing/TrackingSection";
import { SecuritySection } from "@/components/landing/SecuritySection";
import { FinalCta } from "@/components/landing/FinalCta";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#F9F9F8] flex flex-col font-sans selection:bg-[#059669]/15 selection:text-[#059669]">
      {/* NAVBAR */}
      <Header />

      <main className="flex-1">
        {/* SECTION 1 — HERO & CANONICAL JOURNEY */}
        <LandingHero />

        {/* SECTION 2 — THE LEGAL PROBLEM */}
        <ProblemSection />

        {/* SECTION 3 — AI INTELLIGENCE & REVIEW */}
        <DocumentIntelligence />

        {/* SECTION 4 — TRUST LAYER & ATTORNEY HANDOFF */}
        <TrustSection />

        {/* SECTION 5 — OBLIGATIONS & REPOSITORY */}
        <TrackingSection />

        {/* SECTION 6 — SECURITY & PRIVACY */}
        <SecuritySection />

        {/* SECTION 7 — CLOSING CTA */}
        <FinalCta />
      </main>

      {/* FOOTER */}
      <Footer />
    </div>
  );
}
