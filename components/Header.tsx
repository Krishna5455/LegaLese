"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, ArrowRight } from "lucide-react";

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-[#F9F9F8]/90 backdrop-blur-md border-b border-[#E5E5E3] flex items-center transition-all">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Brand Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/25 text-[#059669] font-bold text-xs group-hover:bg-[#059669] group-hover:text-white transition-all">
            §
          </div>
          <span className="text-base font-bold tracking-tight text-[#111827] flex items-center">
            <span>Lega</span>
            <span className="text-[#059669]">Lese</span>
          </span>
        </Link>

        {/* Center Desktop Navigation Links */}
        <nav
          aria-label="Main navigation"
          className="hidden md:flex items-center gap-1 text-xs font-medium text-[#4B5563]"
        >
          <Link
            href="/#journey"
            className="px-3 py-1.5 hover:text-[#111827] transition-colors rounded-lg"
          >
            The Legal Journey
          </Link>
          <Link
            href="/#intelligence"
            className="px-3 py-1.5 hover:text-[#111827] transition-colors rounded-lg"
          >
            AI & Trust Layer
          </Link>
          <Link
            href="/#tracking"
            className="px-3 py-1.5 hover:text-[#111827] transition-colors rounded-lg"
          >
            Obligations & Repository
          </Link>
        </nav>

        {/* Right CTA Links */}
        <div className="hidden sm:flex items-center gap-3">
          <Link
            href="/login"
            className="text-xs font-medium text-[#4B5563] hover:text-[#111827] transition-colors px-3 py-1.5"
          >
            Sign in
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#111827] px-4 py-2 text-xs font-semibold text-white hover:bg-[#1F2937] transition-all shadow-xs active:scale-98"
          >
            <span>Start a LegalFlow</span>
            <ArrowRight className="w-3.5 h-3.5 text-[#10B981]" />
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-1.5 text-[#4B5563] hover:text-[#111827] rounded-lg hover:bg-black/5"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-16 left-0 right-0 border-b border-[#E5E5E3] bg-[#F9F9F8] px-6 py-4 space-y-3 shadow-md">
          <nav className="flex flex-col space-y-2 text-xs">
            <Link
              href="/#journey"
              onClick={() => setMobileMenuOpen(false)}
              className="text-[#111827] font-medium py-1"
            >
              The Legal Journey
            </Link>
            <Link
              href="/#intelligence"
              onClick={() => setMobileMenuOpen(false)}
              className="text-[#4B5563] hover:text-[#111827] py-1"
            >
              AI & Trust Layer
            </Link>
            <Link
              href="/#tracking"
              onClick={() => setMobileMenuOpen(false)}
              className="text-[#4B5563] hover:text-[#111827] py-1"
            >
              Obligations & Repository
            </Link>
          </nav>
          <div className="pt-3 border-t border-[#E5E5E3] flex flex-col gap-2">
            <Link
              href="/login"
              onClick={() => setMobileMenuOpen(false)}
              className="text-center py-2 text-xs font-medium text-[#4B5563] hover:text-[#111827]"
            >
              Sign in
            </Link>
            <Link
              href="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="text-center py-2 text-xs font-semibold bg-[#111827] text-white rounded-lg hover:bg-[#1F2937]"
            >
              Start a LegalFlow →
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
