"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { signOut } from "@/lib/actions/auth";
import {
  LayoutDashboard,
  Calendar,
  FolderLock,
  LogOut,
  ChevronDown,
  Menu,
  X,
  Sparkles,
  HelpCircle,
} from "lucide-react";

type DashboardNavProps = {
  userEmail?: string | null;
  active?: "dashboard" | "documents" | "calendar" | "flow";
};

export function DashboardNav({ userEmail, active: explicitActive }: DashboardNavProps) {
  const pathname = usePathname();
  const [profileOpen, setProfileOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  // Compute active tab from pathname if not explicitly provided
  let active = explicitActive;
  if (!active) {
    if (pathname.startsWith("/dashboard/vault") || pathname.startsWith("/dashboard/documents")) {
      active = "documents";
    } else if (pathname.startsWith("/dashboard/calendar")) {
      active = "calendar";
    } else if (pathname.startsWith("/dashboard/flow")) {
      active = "flow";
    } else {
      active = "dashboard";
    }
  }

  const initial = userEmail ? userEmail[0].toUpperCase() : "U";

  return (
    <>
      <header className="sticky top-0 z-40 h-16 bg-[#FAFAF9]/90 backdrop-blur-md border-b border-[#E5E5E3] flex items-center transition-all">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 sm:px-6">
          {/* Left: Brand Logo & Workspace context */}
          <div className="flex items-center gap-6">
            <Link href="/dashboard" className="flex items-center gap-2.5 group shrink-0">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/25 text-[#059669] font-bold text-xs group-hover:bg-[#059669] group-hover:text-white transition-colors">
                §
              </div>
              <span className="text-base font-bold tracking-tight text-[#111827] flex items-center">
                <span>Lega</span>
                <span className="text-[#059669]">Lese</span>
              </span>
            </Link>

            {/* Primary Navigation Items */}
            <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-[#4B5563]">
              <Link
                href="/dashboard"
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  active === "dashboard" || active === "flow"
                    ? "bg-white text-[#111827] font-semibold border border-[#E5E5E3] shadow-2xs"
                    : "hover:text-[#111827] hover:bg-black/5"
                }`}
              >
                <LayoutDashboard className="w-3.5 h-3.5" />
                <span>Dashboard</span>
              </Link>

              <Link
                href="/dashboard/vault"
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  active === "documents"
                    ? "bg-white text-[#111827] font-semibold border border-[#E5E5E3] shadow-2xs"
                    : "hover:text-[#111827] hover:bg-black/5"
                }`}
              >
                <FolderLock className="w-3.5 h-3.5" />
                <span>Documents</span>
              </Link>

              <Link
                href="/dashboard/calendar"
                className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                  active === "calendar"
                    ? "bg-white text-[#111827] font-semibold border border-[#E5E5E3] shadow-2xs"
                    : "hover:text-[#111827] hover:bg-black/5"
                }`}
              >
                <Calendar className="w-3.5 h-3.5" />
                <span>Calendar</span>
              </Link>
            </nav>
          </div>

          {/* Right: Actions, Help & Profile */}
          <div className="flex items-center gap-3">
            {/* Quick Demo Canonical Badge */}
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#ECFDF5] border border-[#A7F3D0] text-[11px] font-medium text-[#065F46]">
              <Sparkles className="w-3 h-3 text-[#059669]" />
              <span>LegalFlow Platform</span>
            </div>

            {/* Help Button */}
            <button
              type="button"
              onClick={() => setHelpOpen(true)}
              className="p-1.5 rounded-lg text-[#6B7280] hover:text-[#111827] hover:bg-black/5 transition-colors cursor-pointer"
              aria-label="Product guidance"
            >
              <HelpCircle className="w-4 h-4" />
            </button>

            {/* Profile Menu */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setProfileOpen((prev) => !prev)}
                className="flex items-center gap-1.5 p-1 rounded-full hover:bg-black/5 transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#059669]/20"
                aria-label="User profile menu"
              >
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#111827] text-white font-semibold text-xs">
                  {initial}
                </div>
                <ChevronDown className="w-3 h-3 text-[#6B7280]" />
              </button>

              {profileOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setProfileOpen(false)}
                  />
                  <div className="absolute right-0 top-10 z-50 w-56 rounded-xl bg-white border border-[#E5E5E3] shadow-lg p-2 space-y-1">
                    <div className="px-3 py-2 border-b border-[#E5E5E3] mb-1">
                      <p className="text-[10px] font-mono font-semibold text-[#6B7280] uppercase tracking-wider">
                        Workspace Session
                      </p>
                      <p className="text-xs text-[#111827] font-medium truncate mt-0.5">
                        {userEmail || "demo@legalese.app"}
                      </p>
                    </div>

                    <Link
                      href="/dashboard"
                      onClick={() => setProfileOpen(false)}
                      className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-[#F9F9F8] transition-colors flex items-center gap-2"
                    >
                      <LayoutDashboard className="w-3.5 h-3.5 text-[#6B7280]" />
                      <span>Command Center</span>
                    </Link>

                    <Link
                      href="/dashboard/vault"
                      onClick={() => setProfileOpen(false)}
                      className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-[#F9F9F8] transition-colors flex items-center gap-2"
                    >
                      <FolderLock className="w-3.5 h-3.5 text-[#6B7280]" />
                      <span>Documents Repository</span>
                    </Link>

                    <Link
                      href="/dashboard/calendar"
                      onClick={() => setProfileOpen(false)}
                      className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-[#F9F9F8] transition-colors flex items-center gap-2"
                    >
                      <Calendar className="w-3.5 h-3.5 text-[#6B7280]" />
                      <span>Legal Calendar</span>
                    </Link>

                    <div className="border-t border-[#E5E5E3] my-1" />

                    <form action={signOut}>
                      <button
                        type="submit"
                        className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#DC2626] hover:bg-[#FEF2F2] transition-colors flex items-center gap-2 cursor-pointer"
                      >
                        <LogOut className="w-3.5 h-3.5" />
                        <span>Sign Out</span>
                      </button>
                    </form>
                  </div>
                </>
              )}
            </div>

            {/* Mobile Menu Toggle */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              className="md:hidden p-1.5 rounded-lg text-[#4B5563] hover:text-[#111827] hover:bg-black/5 transition-colors focus:outline-none"
              aria-label="Toggle mobile menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 border-b border-[#E5E5E3] bg-[#FAFAF9] px-4 py-3 space-y-1 shadow-sm">
            <Link
              href="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="block rounded-lg px-3 py-2 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-black/5"
            >
              Dashboard
            </Link>
            <Link
              href="/dashboard/vault"
              onClick={() => setMobileMenuOpen(false)}
              className="block rounded-lg px-3 py-2 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-black/5"
            >
              Documents
            </Link>
            <Link
              href="/dashboard/calendar"
              onClick={() => setMobileMenuOpen(false)}
              className="block rounded-lg px-3 py-2 text-xs font-medium text-[#4B5563] hover:text-[#111827] hover:bg-black/5"
            >
              Calendar
            </Link>
          </div>
        )}
      </header>

      {/* Help & Guidance Modal */}
      {helpOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
          <div className="relative w-full max-w-md rounded-2xl bg-white border border-[#E5E5E3] p-6 space-y-4 shadow-xl">
            <button
              type="button"
              onClick={() => setHelpOpen(false)}
              className="absolute top-4 right-4 p-1 rounded-lg text-[#6B7280] hover:text-[#111827] hover:bg-[#F3F4F6] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-[#059669]/10 text-[#059669]">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-[#111827]">How LegalFlow Works</h3>
                <p className="text-xs text-[#6B7280]">From Intent to Contract Fulfillment</p>
              </div>
            </div>

            <div className="space-y-3 text-xs text-[#4B5563] leading-relaxed pt-2">
              <div className="flex gap-2">
                <span className="font-bold text-[#111827]">1. Intent:</span>
                <span>Type what you want to achieve (e.g. &quot;I want to hire a freelancer for my website&quot;).</span>
              </div>
              <div className="flex gap-2">
                <span className="font-bold text-[#111827]">2. Understand & Prepare:</span>
                <span>LegaLese structures the requirements and drafts the agreement.</span>
              </div>
              <div className="flex gap-2">
                <span className="font-bold text-[#111827]">3. Review & Trust:</span>
                <span>AI risk detection identifies liability traps and recommends professional review when warranted.</span>
              </div>
              <div className="flex gap-2">
                <span className="font-bold text-[#111827]">4. Track & Fulfill:</span>
                <span>Extracted payment milestones and deliverable deadlines are tracked in your Legal Calendar.</span>
              </div>
            </div>

            <div className="pt-2 border-t border-[#E5E5E3] flex justify-end">
              <button
                type="button"
                onClick={() => setHelpOpen(false)}
                className="px-4 py-2 rounded-lg bg-[#111827] text-white text-xs font-semibold hover:bg-[#1F2937] transition-colors"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
