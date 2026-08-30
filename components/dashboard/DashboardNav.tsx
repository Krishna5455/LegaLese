"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { signOut } from "@/lib/actions/auth";
import {
  LayoutDashboard,
  FilePlus,
  FileText,
  Search,
  LogOut,
  ChevronDown,
  Menu,
  X,
} from "lucide-react";

type DashboardNavProps = {
  userEmail?: string | null;
  active?: "dashboard" | "create" | "documents" | "analyze";
};

export function DashboardNav({ userEmail, active: explicitActive }: DashboardNavProps) {
  const pathname = usePathname();
  const [profileOpen, setProfileOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Compute active tab from pathname if not explicitly provided
  let active = explicitActive;
  if (!active) {
    if (pathname.startsWith("/dashboard/create")) {
      active = "create";
    } else if (pathname.startsWith("/dashboard/documents")) {
      active = "documents";
    } else {
      active = "dashboard";
    }
  }

  const initial = userEmail ? userEmail[0].toUpperCase() : "U";

  const handleSmoothScroll = (targetId: string) => {
    if (pathname === "/dashboard") {
      const el = document.getElementById(targetId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
      }
    }
  };

  return (
    <header className="sticky top-0 z-50 h-16 border-b border-[#E7E5E2] bg-[#F7F7F5]/90 backdrop-blur-md flex items-center transition-all">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Left: Brand Logo */}
        <Link href="/dashboard" className="flex items-center gap-2.5 group shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/25 text-[#059669] font-bold text-xs group-hover:bg-[#059669] group-hover:text-white transition-colors">
            §
          </div>
          <span className="text-base font-bold tracking-tight text-[#171717] flex items-center">
            <span>Lega</span>
            <span className="text-[#059669]">Lese</span>
          </span>
        </Link>

        {/* Center: Main Navigation (Desktop) */}
        <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-[#5F6368]">
          <Link
            href="/dashboard"
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              active === "dashboard"
                ? "bg-white text-[#171717] font-semibold border border-[#E7E5E2] shadow-2xs"
                : "hover:text-[#171717] hover:bg-black/5"
            }`}
          >
            <LayoutDashboard className="w-3.5 h-3.5" />
            <span>Home</span>
          </Link>

          <Link
            href="/dashboard/create"
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              active === "create"
                ? "bg-white text-[#171717] font-semibold border border-[#E7E5E2] shadow-2xs"
                : "hover:text-[#171717] hover:bg-black/5"
            }`}
          >
            <FilePlus className="w-3.5 h-3.5" />
            <span>Create Document</span>
          </Link>

          <Link
            href="/dashboard#documents"
            onClick={() => handleSmoothScroll("documents")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              active === "documents"
                ? "bg-white text-[#171717] font-semibold border border-[#E7E5E2] shadow-2xs"
                : "hover:text-[#171717] hover:bg-black/5"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>My Documents</span>
          </Link>

          <Link
            href="/dashboard#upload"
            onClick={() => handleSmoothScroll("upload")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              active === "analyze"
                ? "bg-white text-[#171717] font-semibold border border-[#E7E5E2] shadow-2xs"
                : "hover:text-[#171717] hover:bg-black/5"
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Analyze Contract</span>
          </Link>

          {/* Understated Coming Soon Features */}
          <span className="flex items-center gap-1.5 px-2.5 py-1.5 text-[#8A8F98] select-none cursor-not-allowed text-xs">
            <span>Community</span>
            <span className="rounded bg-[#F0EFEA] text-[#8A8F98] border border-[#E7E5E2] px-1.5 py-0.2 text-[9px] font-mono font-medium">
              Soon
            </span>
          </span>

          <span className="flex items-center gap-1.5 px-2.5 py-1.5 text-[#8A8F98] select-none cursor-not-allowed text-xs hidden lg:flex">
            <span>Legal Expert</span>
            <span className="rounded bg-[#F0EFEA] text-[#8A8F98] border border-[#E7E5E2] px-1.5 py-0.2 text-[9px] font-mono font-medium">
              Soon
            </span>
          </span>
        </nav>

        {/* Right: Profile Avatar & Mobile Toggle */}
        <div className="flex items-center gap-3">
          {/* Avatar Button */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setProfileOpen((prev) => !prev)}
              className="flex items-center gap-1.5 p-1 rounded-full hover:bg-black/5 transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#059669]/20"
              aria-label="User profile menu"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#171717] text-white font-semibold text-xs border border-[#171717]">
                {initial}
              </div>
              <ChevronDown className="w-3.5 h-3.5 text-[#8A8F98]" />
            </button>

            {/* Profile Dropdown Menu */}
            {profileOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setProfileOpen(false)}
                />
                <div className="absolute right-0 top-11 z-50 w-56 rounded-xl border border-[#E7E5E2] bg-white p-2 shadow-lg space-y-1 animate-in fade-in slide-in-from-top-2">
                  <div className="px-3 py-2 border-b border-[#E7E5E2] mb-1">
                    <p className="text-[10px] font-mono font-semibold text-[#8A8F98] uppercase tracking-wider">
                      Account
                    </p>
                    <p className="text-xs text-[#171717] font-medium truncate mt-0.5">
                      {userEmail || "user@legalese.com"}
                    </p>
                  </div>

                  <Link
                    href="/dashboard"
                    onClick={() => setProfileOpen(false)}
                    className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-[#F7F7F5] transition-colors flex items-center gap-2"
                  >
                    <LayoutDashboard className="w-3.5 h-3.5 text-[#8A8F98]" />
                    <span>Workspace Dashboard</span>
                  </Link>

                  <Link
                    href="/dashboard/create"
                    onClick={() => setProfileOpen(false)}
                    className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-[#F7F7F5] transition-colors flex items-center gap-2"
                  >
                    <FilePlus className="w-3.5 h-3.5 text-[#8A8F98]" />
                    <span>Create Document</span>
                  </Link>

                  <div className="border-t border-[#E7E5E2] my-1" />

                  <form action={signOut}>
                    <button
                      type="submit"
                      className="w-full text-left rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#B91C1C] hover:bg-[#FEF2F2] transition-colors flex items-center gap-2 cursor-pointer"
                    >
                      <LogOut className="w-3.5 h-3.5" />
                      <span>Sign Out</span>
                    </button>
                  </form>
                </div>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            className="md:hidden p-1.5 rounded-lg text-[#5F6368] hover:text-[#171717] hover:bg-black/5 transition-colors focus:outline-none"
            aria-label="Toggle mobile menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-16 left-0 right-0 border-b border-[#E7E5E2] bg-[#F7F7F5] px-4 py-3 space-y-1 shadow-sm">
          <Link
            href="/dashboard"
            onClick={() => setMobileMenuOpen(false)}
            className="block rounded-lg px-3 py-2 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-black/5"
          >
            Home
          </Link>
          <Link
            href="/dashboard/create"
            onClick={() => setMobileMenuOpen(false)}
            className="block rounded-lg px-3 py-2 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-black/5"
          >
            Create Document
          </Link>
          <Link
            href="/dashboard#documents"
            onClick={() => {
              setMobileMenuOpen(false);
              handleSmoothScroll("documents");
            }}
            className="block rounded-lg px-3 py-2 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-black/5"
          >
            My Documents
          </Link>
          <Link
            href="/dashboard#upload"
            onClick={() => {
              setMobileMenuOpen(false);
              handleSmoothScroll("upload");
            }}
            className="block rounded-lg px-3 py-2 text-xs font-medium text-[#5F6368] hover:text-[#171717] hover:bg-black/5"
          >
            Analyze Contract
          </Link>
        </div>
      )}
    </header>
  );
}
