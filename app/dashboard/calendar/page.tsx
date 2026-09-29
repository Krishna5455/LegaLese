import { redirect } from "next/navigation";
import { getCalendarObligations } from "@/lib/actions/legalflow";
import { LegalCalendarView } from "@/components/calendar/LegalCalendarView";
import { createClient } from "@/lib/supabase/server";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";

export const dynamic = "force-dynamic";

export default async function CalendarPage() {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  const { events } = await getCalendarObligations();

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8">
      <LegalCalendarView initialEvents={events || []} />
    </main>
  );
}
