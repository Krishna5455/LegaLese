"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { Menu, X, ArrowRight, Check, Users, Scale } from "lucide-react";

export function Header() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeModal, setActiveModal] = useState<"community" | "expert" | null>(null);
  const [emailSubmitted, setEmailSubmitted] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 15);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-200 h-16 flex items-center ${
          scrolled
            ? "bg-[#F7F7F5]/90 backdrop-blur-md border-b border-[#E7E5E2] shadow-[0_1px_3px_rgba(0,0,0,0.03)]"
            : "bg-[#F7F7F5]/70 backdrop-blur-xs border-b border-transparent"
        }`}
      >
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 sm:px-8">
          {/* Brand Logo */}
          <Link href="/#hero" className="flex items-center gap-2.5 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/25 text-[#059669] font-bold text-base shadow-2xs group-hover:bg-[#059669] group-hover:text-white transition-all">
              §
            </div>
            <span className="text-[17px] font-bold tracking-tight text-[#171717] flex items-center">
              <span>Lega</span>
              <span className="text-[#059669]">Lese</span>
            </span>
          </Link>

          {/* Center Desktop Navigation Links */}
          <nav
            aria-label="Main navigation"
            className="hidden md:flex items-center gap-1 text-[13px] font-medium rounded-full bg-white border border-[#E7E5E2] px-3.5 py-1 shadow-2xs"
          >
            <Link
              href="/#hero"
              className="px-2.5 py-1 text-[#171717] hover:text-[#059669] transition-colors rounded-full"
            >
              Home
            </Link>
            <Link
              href="/#create"
              className="px-2.5 py-1 text-[#5F6368] hover:text-[#171717] transition-colors rounded-full"
            >
              Create
            </Link>
            <Link
              href="/dashboard"
              className="px-2.5 py-1 text-[#5F6368] hover:text-[#171717] transition-colors rounded-full"
            >
              Documents
            </Link>
            <Link
              href="/#analyze"
              className="px-2.5 py-1 text-[#5F6368] hover:text-[#171717] transition-colors rounded-full"
            >
              Analyze
            </Link>

            <span className="h-3 w-px bg-[#E7E5E2] mx-1" />

            <button
              type="button"
              suppressHydrationWarning
              onClick={() => {
                setActiveModal("community");
                setEmailSubmitted(false);
              }}
              className="px-2 py-0.5 text-[12px] text-[#5F6368] hover:text-[#171717] flex items-center gap-1.5 transition-colors"
            >
              Community{" "}
              <span className="text-[9px] uppercase tracking-wider font-semibold bg-[#F0EFEA] border border-[#E7E5E2] px-1.5 py-0.2 rounded text-[#8A8F98]">
                Soon
              </span>
            </button>
            <button
              type="button"
              suppressHydrationWarning
              onClick={() => {
                setActiveModal("expert");
                setEmailSubmitted(false);
              }}
              className="px-2 py-0.5 text-[12px] text-[#5F6368] hover:text-[#171717] flex items-center gap-1.5 transition-colors"
            >
              Legal Expert{" "}
              <span className="text-[9px] uppercase tracking-wider font-semibold bg-[#F0EFEA] border border-[#E7E5E2] px-1.5 py-0.2 rounded text-[#8A8F98]">
                Soon
              </span>
            </button>
          </nav>

          {/* Right CTA Links */}
          <div className="hidden sm:flex items-center gap-3">
            <Link
              href="/login"
              className="text-[13px] font-medium text-[#5F6368] hover:text-[#171717] transition-colors px-3 py-1.5"
            >
              Sign in
            </Link>
            <Link
              href="/dashboard/create"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#171717] px-3.5 py-1.5 text-[13px] font-medium text-white hover:bg-[#262626] transition-all shadow-xs active:scale-98"
            >
              <span>Create a document</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#059669]" />
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-[#5F6368] hover:text-[#171717] rounded-lg hover:bg-black/5"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 border-b border-[#E7E5E2] bg-[#F7F7F5] px-6 py-5 space-y-4 shadow-lg animate-in fade-in slide-in-from-top-2">
            <nav className="flex flex-col space-y-2 text-sm">
              <Link
                href="/#hero"
                onClick={() => setMobileMenuOpen(false)}
                className="text-[#171717] font-medium hover:text-[#059669] py-1"
              >
                Home
              </Link>
              <Link
                href="/#create"
                onClick={() => setMobileMenuOpen(false)}
                className="text-[#5F6368] hover:text-[#171717] py-1"
              >
                Create Agreement
              </Link>
              <Link
                href="/dashboard"
                onClick={() => setMobileMenuOpen(false)}
                className="text-[#5F6368] hover:text-[#171717] py-1"
              >
                Documents Workspace
              </Link>
              <Link
                href="/#analyze"
                onClick={() => setMobileMenuOpen(false)}
                className="text-[#5F6368] hover:text-[#171717] py-1"
              >
                Analyze Contract
              </Link>
              <button
                type="button"
                onClick={() => {
                  setMobileMenuOpen(false);
                  setActiveModal("community");
                  setEmailSubmitted(false);
                }}
                className="text-left text-[#5F6368] hover:text-[#171717] py-1 flex items-center justify-between"
              >
                <span>Community Library</span>
                <span className="text-[10px] bg-[#E7E5E2] px-2 py-0.5 rounded">Soon</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setMobileMenuOpen(false);
                  setActiveModal("expert");
                  setEmailSubmitted(false);
                }}
                className="text-left text-[#5F6368] hover:text-[#171717] py-1 flex items-center justify-between"
              >
                <span>Legal Expert Network</span>
                <span className="text-[10px] bg-[#E7E5E2] px-2 py-0.5 rounded">Soon</span>
              </button>
            </nav>
            <div className="pt-3 border-t border-[#E7E5E2] flex flex-col gap-2.5">
              <Link
                href="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="text-center py-2 text-sm font-medium text-[#5F6368] hover:text-[#171717]"
              >
                Sign in
              </Link>
              <Link
                href="/dashboard/create"
                onClick={() => setMobileMenuOpen(false)}
                className="text-center py-2 text-sm font-medium bg-[#171717] text-white rounded-lg hover:bg-[#262626]"
              >
                Create a document
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Interactive Coming Soon Modals */}
      {activeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="relative w-full max-w-md rounded-2xl bg-white p-6 sm:p-8 border border-[#E7E5E2] shadow-2xl space-y-5">
            <button
              type="button"
              onClick={() => setActiveModal(null)}
              className="absolute top-4 right-4 p-1.5 rounded-lg text-[#8A8F98] hover:text-[#171717] hover:bg-[#F7F7F5] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-[#059669]/10 text-[#059669] flex items-center justify-center">
                {activeModal === "community" ? (
                  <Users className="w-5 h-5" />
                ) : (
                  <Scale className="w-5 h-5" />
                )}
              </div>
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#059669] font-bold">
                  Roadmap Milestone
                </span>
                <h3 className="text-lg font-bold text-[#171717]">
                  {activeModal === "community"
                    ? "Community Agreement Library"
                    : "1-Click Legal Expert Review"}
                </h3>
              </div>
            </div>

            <p className="text-xs sm:text-sm text-[#5F6368] leading-relaxed">
              {activeModal === "community"
                ? "Access hundreds of peer-vetted contracts for freelance retainers, agency SOWs, software development, and design services rated for mutual fairness."
                : "Submit your analyzed contract directly to a bar-certified corporate attorney for human certification with a guaranteed 24-hour turnaround time."}
            </p>

            {emailSubmitted ? (
              <div className="p-3.5 rounded-xl bg-[#F0FDF4] border border-[#BBF7D0] flex items-center gap-2 text-xs font-semibold text-[#166534]">
                <Check className="w-4 h-4 text-[#059669]" />
                <span>You are on the priority waitlist! We will notify you at launch.</span>
              </div>
            ) : (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  setEmailSubmitted(true);
                }}
                className="space-y-3"
              >
                <div className="flex items-center gap-2">
                  <input
                    type="email"
                    required
                    suppressHydrationWarning
                    placeholder="Enter your work email"
                    className="flex-1 rounded-lg border border-[#E7E5E2] bg-[#F7F7F5] px-3.5 py-2 text-xs text-[#171717] placeholder:text-[#8A8F98] focus:outline-none focus:ring-1 focus:ring-[#059669]"
                  />
                  <button
                    type="submit"
                    suppressHydrationWarning
                    className="rounded-lg bg-[#171717] px-4 py-2 text-xs font-semibold text-white hover:bg-[#262626] transition-colors shrink-0 cursor-pointer"
                  >
                    Join Waitlist
                  </button>
                </div>
                <p className="text-[11px] text-[#8A8F98]">
                  Early access beta launching in Q4 2026. Zero spam.
                </p>
              </form>
            )}
          </div>
        </div>
      )}
    </>
  );
}
