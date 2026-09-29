import { Calendar, FolderLock, CheckCircle2, Clock, DollarSign, Bell } from "lucide-react";

export function TrackingSection() {
  return (
    <section id="tracking" className="py-20 bg-[#F9F9F8] border-b border-[#E5E5E3]">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          {/* Left Column: Calendar Card Demo */}
          <div className="lg:col-span-6 space-y-4">
            <div className="p-6 rounded-2xl bg-white border border-[#E5E5E3] shadow-sm space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-[#E5E5E3]">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-[#059669]" />
                  <span className="text-xs font-bold text-[#111827]">Legal Obligation Timeline</span>
                </div>
                <span className="text-[11px] font-mono text-[#6B7280]">3 Active Deadlines</span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div className="p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700">
                      <DollarSign className="w-3.5 h-3.5" />
                    </div>
                    <div>
                      <p className="font-semibold text-[#111827]">50% Advance Retainer</p>
                      <p className="text-[11px] text-[#6B7280]">Due on signing ($2,500 USD)</p>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono font-bold text-[#059669] bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    Day 0
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-1.5 rounded-lg bg-blue-50 text-blue-700">
                      <Clock className="w-3.5 h-3.5" />
                    </div>
                    <div>
                      <p className="font-semibold text-[#111827]">UI Prototype Milestone</p>
                      <p className="text-[11px] text-[#6B7280]">Deliverable acceptance window</p>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    Day 14
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#F9F9F8] border border-[#E5E5E3] flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-1.5 rounded-lg bg-purple-50 text-purple-700">
                      <Bell className="w-3.5 h-3.5" />
                    </div>
                    <div>
                      <p className="font-semibold text-[#111827]">90-Day Warranty Expiry</p>
                      <p className="text-[11px] text-[#6B7280]">Bug fix support period</p>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded border border-purple-200">
                    Day 90
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Narrative */}
          <div className="lg:col-span-6 space-y-5">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E5E5E3] text-xs font-semibold text-[#111827]">
              <FolderLock className="w-3.5 h-3.5 text-[#059669]" />
              <span>Tracking & Repository</span>
            </div>

            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827] leading-snug">
              Never miss a contract deadline or renewal window.
            </h2>

            <p className="text-sm text-[#4B5563] leading-relaxed">
              LegaLese extracts contractual milestones directly into your Legal Calendar and archives all executed versions in your unified Document Repository.
            </p>

            <ul className="space-y-2.5 text-xs text-[#374151]">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>Automatic milestone & payment due date extraction</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>Unified repository for generated agreements and audited uploads</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#059669]" />
                <span>Bidirectional linkage from any document back to its active LegalFlow</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
