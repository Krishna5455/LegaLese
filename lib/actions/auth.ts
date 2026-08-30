"use server";

import { redirect } from "next/navigation";

import { createClient } from "@/lib/supabase/server";

export type AuthActionState = {
  error?: string;
  message?: string;
};

export async function signUp(
  _prevState: AuthActionState,
  formData: FormData,
): Promise<AuthActionState> {
  const email = String(formData.get("email") ?? "").trim();
  const password = String(formData.get("password") ?? "");

  if (!email || !password) {
    return { error: "Email and password are required." };
  }

  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
  const supabase = await createClient();

  // 1. Attempt signup, preserving emailRedirectTo so email verification links work when sent
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${siteUrl}/auth/callback`,
    },
  });

  // If user already exists, directly sign them in with their credentials
  if (error && error.message.toLowerCase().includes("already registered")) {
    const { error: signInErr } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (!signInErr) {
      redirect("/dashboard");
    }
    return { error: signInErr.message };
  }

  if (error) {
    return { error: error.message };
  }

  // 2. If a session is already established (e.g. Confirm email is turned off in Supabase), redirect directly
  if (data?.session) {
    redirect("/dashboard");
  }

  // 3. Attempt immediate password login so the user is directly signed in without waiting
  const { data: signInData, error: signInError } =
    await supabase.auth.signInWithPassword({
      email,
      password,
    });

  if (!signInError && signInData?.session) {
    redirect("/dashboard");
  }

  // 4. If Supabase has email confirmation enabled, inform the user with verification instructions
  return {
    message: `Account created! We've sent a verification link to ${email}. Click the link in your email to sign in directly. (To skip this email step permanently, turn off 'Confirm email' in your Supabase Auth settings).`,
  };
}

export async function signIn(
  _prevState: AuthActionState,
  formData: FormData,
): Promise<AuthActionState> {
  const email = String(formData.get("email") ?? "").trim();
  const password = String(formData.get("password") ?? "");

  if (!email || !password) {
    return { error: "Email and password are required." };
  }

  const supabase = await createClient();
  const { error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    if (error.message.toLowerCase().includes("email not confirmed")) {
      return {
        error:
          "Your email address has not been confirmed yet. Please check your inbox for the verification link, or turn off 'Confirm email' in your Supabase Dashboard settings to sign in without email verification.",
      };
    }
    return { error: error.message };
  }

  const next = String(formData.get("next") ?? "/dashboard");
  redirect(next.startsWith("/") ? next : "/dashboard");
}

export async function signOut() {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect("/login");
}
