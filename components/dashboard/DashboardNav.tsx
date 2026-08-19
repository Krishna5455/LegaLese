"use client";

import Link from "next/link";
import { useState } from "react";
import { signOut } from "@/lib/actions/auth";

type DashboardNavProps = {
  userEmail?: string | null;
  active?: "dashboard" | "create";
};

export function DashboardNav({ userEmail, active }: DashboardNavProps) {
  const [profileOpen, setProfileOpen] = useState(false);

  const initial = userEmail ? userEmail[0].toUpperCase() : "U";

  return (
    <header className="sticky top-0 z-50 h-[68px] border-b border-[#EDEDF5] bg-white flex items-center shadow-2xs">
      <div className="mx-auto flex w-full max-w-[1320px] items-center justify-between px-6 sm:px-8">
        {/* LEFT: Dual-Tone Logo (28px) */}
        <Link href="/dashboard" className="flex items-center gap-2.5 group shrink-0">
          <div className="flex h-8.5 w-8.5 items-center justify-center rounded-xl bg-[#F1F0FF] border border-[#E0E7FF] text-[#6366F1] font-bold text-sm shadow-2xs group-hover:bg-indigo-100 transition-colors">
            §
          </div>
          <span className="text-2xl sm:text-[28px] font-bold tracking-tight">
            <span className="text-[#111827]">Lega</span>
            <span className="text-[#6366F1]">Lese</span>
          </span>
        </Link>

        {/* CENTER / MAIN NAV: 15px Navigation Text & Spacious 28-36px Gaps */}
        <nav className="flex items-center gap-7 lg:gap-8 xl:gap-9 text-[15px] font-medium">
          <Link
            href="/dashboard"
            className={
              active === "dashboard"
                ? "bg-[#F1F0FF] text-[#6366F1] font-semibold rounded-[12px] px-3.5 py-2 transition-all"
                : "text-[#57534E] hover:text-[#6366F1] hover:bg-[#F5F3FF] rounded-[12px] px-3.5 py-2 transition-all duration-200 ease-out"
            }
          >
            Home
          </Link>

          <Link
            href="/dashboard/create"
            className={
              active === "create"
                ? "bg-[#F1F0FF] text-[#6366F1] font-semibold rounded-[12px] px-3.5 py-2 transition-all"
                : "text-[#57534E] hover:text-[#6366F1] hover:bg-[#F5F3FF] rounded-[12px] px-3.5 py-2 transition-all duration-200 ease-out"
            }
          >
            Create Document
          </Link>

          <Link
            href="/dashboard"
            className="text-[#57534E] hover:text-[#6366F1] hover:bg-[#F5F3FF] rounded-[12px] px-3.5 py-2 transition-all duration-200 ease-out hidden md:inline-block"
          >
            My Documents
          </Link>

          <Link
            href="/dashboard#upload"
            className="text-[#57534E] hover:text-[#6366F1] hover:bg-[#F5F3FF] rounded-[12px] px-3.5 py-2 transition-all duration-200 ease-out hidden md:inline-block"
          >
            Analyze Contract
          </Link>

          {/* Roadmap Features with 11px Badges */}
          <span className="text-[#57534E] flex items-center gap-1.5 cursor-not-allowed select-none px-2 py-1 text-[15px] font-medium hidden lg:inline-flex">
            <span>Community</span>
            <span className="rounded-md bg-[#EEF2FF] text-[#6366F1] border border-[#E0E7FF] px-2 py-0.5 text-[11px] font-semibold">
              Coming Soon
            </span>
          </span>

          <span className="text-[#57534E] flex items-center gap-1.5 cursor-not-allowed select-none px-2 py-1 text-[15px] font-medium hidden xl:inline-flex">
            <span>Legal Expert</span>
            <span className="rounded-md bg-[#EEF2FF] text-[#6366F1] border border-[#E0E7FF] px-2 py-0.5 text-[11px] font-semibold">
              Coming Soon
            </span>
          </span>
        </nav>

        {/* RIGHT: Avatar Profile Button & Popover Dropdown */}
        <div className="relative shrink-0">
          <button
            type="button"
            onClick={() => setProfileOpen((prev) => !prev)}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-[#F1F0FF] text-[#6366F1] font-bold text-sm border border-[#E0E7FF] hover:bg-indigo-100 transition-colors shadow-2xs cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent/30"
            title="User Profile"
          >
            {initial}
          </button>

          {/* Profile Dropdown Popover */}
          {profileOpen ? (
            <>
              {/* Backdrop dismiss overlay */}
              <div
                className="fixed inset-0 z-40"
                onClick={() => setProfileOpen(false)}
              />

              <div className="absolute right-0 top-11 z-50 w-60 rounded-2xl border border-border bg-white p-3 shadow-lg space-y-2 animate-fadeIn">
                <div className="px-2.5 py-1.5 space-y-0.5">
                  <p className="text-xs font-bold text-[#111827]">
                    Signed in as
                  </p>
                  <p className="text-xs text-muted font-mono truncate">
                    {userEmail || "user@legalese.com"}
                  </p>
                </div>

                <div className="border-t border-border/60 my-1" />

                <form action={signOut}>
                  <button
                    type="submit"
                    className="w-full text-left rounded-xl px-2.5 py-2 text-xs font-semibold text-red-600 hover:bg-red-50 transition-colors flex items-center gap-1.5"
                  >
                    <span>🚪</span>
                    <span>Sign Out</span>
                  </button>
                </form>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </header>
  );
}
