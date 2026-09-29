import { redirect } from "next/navigation";
import { isDemoMode } from "@/lib/demo/demo-state";

export default function FlowRedirectPage() {
  if (isDemoMode()) {
    redirect("/dashboard/flow/demo-freelance-flow");
  }
  redirect("/dashboard");
}
