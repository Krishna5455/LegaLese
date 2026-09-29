import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { DashboardNav } from "@/components/dashboard/DashboardNav";
import { getAuthenticatedUser } from "@/lib/supabase/auth-helper";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const user = await getAuthenticatedUser(supabase);

  if (!user) {
    redirect("/login");
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F7F5] text-[#171717] font-sans selection:bg-[#059669]/15 selection:text-[#059669]">
      <DashboardNav userEmail={user.email} />
      <div className="flex-1 flex flex-col">{children}</div>
    </div>
  );
}
