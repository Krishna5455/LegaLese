import { type SupabaseClient, type User } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { isDemoMode, DEMO_USER } from "@/lib/demo/demo-state";

/**
 * Fast & safe user resolver that checks Supabase authentication.
 * Skips unnecessary remote network calls when no session cookie exists.
 * Falls back to DEMO_USER only when NEXT_PUBLIC_DEMO_MODE=true is configured.
 */
export async function getAuthenticatedUser(
  supabase: SupabaseClient,
): Promise<User | null> {
  const isDemo = isDemoMode();

  try {
    const cookieStore = await cookies();
    const allCookies = cookieStore.getAll();
    const hasAuthCookie = allCookies.some(
      (c) => c.name.includes("auth-token") || c.name.startsWith("sb-"),
    );

    // Fast-path: if no auth cookie is present in demo mode, avoid remote network latency
    if (isDemo && !hasAuthCookie) {
      return DEMO_USER as unknown as User;
    }

    // Fast-path: if no auth cookie in standard mode, user is definitely unauthenticated
    if (!hasAuthCookie) {
      return null;
    }

    const {
      data: { user },
      error,
    } = await supabase.auth.getUser();

    if (!error && user) {
      return user;
    }
  } catch (err) {
    if (!isDemo) {
      console.error("[LegaLese/getAuthenticatedUser] Auth check error:", err);
    }
  }

  if (isDemo) {
    return DEMO_USER as unknown as User;
  }

  return null;
}
