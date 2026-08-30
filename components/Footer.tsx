import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-[#E7E5E2] bg-white py-12 px-6 sm:px-8 text-xs text-[#5F6368]">
      <div className="mx-auto max-w-7xl space-y-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6">
          {/* Brand Mark */}
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded bg-[#059669]/10 text-[#059669] font-bold text-xs border border-[#059669]/20">
              §
            </span>
            <span className="text-sm font-bold text-[#171717]">LegaLese</span>
            <span className="text-[#8A8F98]">· Legal-tech contract management</span>
          </div>

          {/* Nav Links */}
          <div className="flex flex-wrap items-center gap-6 text-[13px]">
            <Link href="/" className="hover:text-[#171717] transition-colors">
              Home
            </Link>
            <Link href="/dashboard/create" className="hover:text-[#171717] transition-colors">
              Create Document
            </Link>
            <Link href="/dashboard" className="hover:text-[#171717] transition-colors">
              Documents
            </Link>
            <Link href="/dashboard#upload" className="hover:text-[#171717] transition-colors">
              Analyze Contract
            </Link>
            <Link href="/login" className="hover:text-[#171717] transition-colors">
              Sign In
            </Link>
          </div>
        </div>

        {/* Disclaimer & Copyright */}
        <div className="pt-6 border-t border-[#E7E5E2] flex flex-col md:flex-row md:items-center md:justify-between gap-4 text-[#8A8F98]">
          <p className="max-w-2xl text-[11px] leading-relaxed">
            Disclaimer: LegaLese provides document generation tools and automated contract auditing for informational purposes. LegaLese is not a law firm and does not provide formal legal representation or attorney-client privilege.
          </p>
          <p className="text-[11px] font-mono shrink-0">
            © {new Date().getFullYear()} LegaLese Inc. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
