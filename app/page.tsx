import { Header } from "@/components/Header";
import { LandingHero } from "@/components/landing/LandingHero";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { UnderstandSection } from "@/components/landing/UnderstandSection";
import { CreateDocShowcase } from "@/components/landing/CreateDocShowcase";
import { DocumentIntelligence } from "@/components/landing/DocumentIntelligence";
import { HumanStorySection } from "@/components/landing/HumanStorySection";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { ProductShowcase } from "@/components/landing/ProductShowcase";
import { EcosystemRoadmap } from "@/components/landing/EcosystemRoadmap";
import { FinalCta } from "@/components/landing/FinalCta";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#F7F7F5] flex flex-col font-sans selection:bg-[#059669]/15 selection:text-[#059669]">
      {/* NAVBAR */}
      <Header />

      <main className="flex-1">
        {/* SECTION 1 — HERO */}
        <LandingHero />

        {/* SECTION 2 — PROBLEM */}
        <ProblemSection />

        {/* SECTION 3 — UNDERSTAND */}
        <UnderstandSection />

        {/* SECTION 4 — CREATE */}
        <CreateDocShowcase />

        {/* SECTION 5 — ANALYZE */}
        <DocumentIntelligence />

        {/* SECTION 6 — HUMAN STORY */}
        <HumanStorySection />

        {/* SECTION 7 — WORKFLOW */}
        <HowItWorks />

        {/* SECTION 8 — PRODUCT SHOWCASE */}
        <ProductShowcase />

        {/* SECTION 9 — FUTURE */}
        <EcosystemRoadmap />

        {/* SECTION 10 — CTA */}
        <FinalCta />
      </main>

      {/* FOOTER */}
      <Footer />
    </div>
  );
}
