import { Calendar, Clock, DollarSign } from "lucide-react";
import type { LegalCalendarEvent } from "@/types/legalflow";

interface UpcomingObligationsCardProps {
  obligations?: LegalCalendarEvent[];
}

export function UpcomingObligationsCard({
  obligations = [],
}: UpcomingObligationsCardProps) {
  return (
    <div className="rounded-2xl border border-[#E7E5E2] bg-white p-5 sm:p-6 space-y-4 shadow-xs">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#059669]/10 border border-[#059669]/20 text-[#059669]">
            <Calendar className="w-4 h-4" />
          </div>
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#8A8F98]">
              Obligations &amp; Deadlines
            </p>
            <h3 className="text-sm font-bold text-[#171717]">Upcoming Dates</h3>
          </div>
        </div>

        {obligations.length > 0 && (
          <span className="rounded-full bg-[#059669]/10 border border-[#059669]/20 px-2 py-0.5 text-[10px] font-mono font-semibold text-[#059669]">
            {obligations.length} Active
          </span>
        )}
      </div>

      {obligations.length === 0 ? (
        <div className="rounded-xl border border-dashed border-[#E7E5E2] bg-[#F7F7F5]/50 p-4 text-center">
          <p className="text-xs text-[#5F6368]">No obligations detected yet.</p>
          <p className="text-[11px] text-[#8A8F98] mt-0.5">
            Key milestones and payment dates will appear here once extracted.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {obligations.map((obl, idx) => {
            const isPayment = obl.type === "payment";

            return (
              <div
                key={obl.id || idx}
                className="flex items-start justify-between gap-2.5 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5] p-3 text-xs"
              >
                <div className="space-y-0.5 min-w-0">
                  <div className="flex items-center gap-1.5 font-bold text-[#171717]">
                    {isPayment ? (
                      <DollarSign className="w-3.5 h-3.5 text-[#059669] shrink-0" />
                    ) : (
                      <Clock className="w-3.5 h-3.5 text-[#8A8F98] shrink-0" />
                    )}
                    <span className="truncate">{obl.title}</span>
                  </div>
                  {obl.responsibleParty && (
                    <p className="text-[11px] text-[#8A8F98]">
                      Party: <span className="text-[#5F6368]">{obl.responsibleParty}</span>
                    </p>
                  )}
                </div>

                <div className="text-right shrink-0">
                  <span className="rounded bg-white border border-[#E7E5E2] px-2 py-0.5 text-[10px] font-mono font-medium text-[#171717]">
                    {obl.date}
                  </span>
                  {obl.amount && (
                    <p className="text-[11px] font-bold text-[#059669] mt-0.5">
                      {obl.amount}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
