import { AnalyzeSection } from "@/components/AnalyzeSection";
import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { Hero } from "@/components/Hero";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex flex-1 flex-col">
        <Hero />
        <AnalyzeSection />
      </main>
      <Footer />
    </div>
  );
}
