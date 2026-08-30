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

  const supabase = await createClient();

  // 1. Attempt to sign up directly without requiring email verification
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
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

  // 2. If a session is already returned, redirect directly to dashboard
  if (data?.session) {
    redirect("/dashboard");
  }

  // 3. Attempt immediate sign-in with password so user is directly logged in
  const { data: signInData, error: signInError } =
    await supabase.auth.signInWithPassword({
      email,
      password,
    });

  if (!signInError && signInData?.session) {
    redirect("/dashboard");
  }

  if (signInError) {
    if (signInError.message.toLowerCase().includes("email not confirmed")) {
      return {
        error:
          "To sign in directly without email verification, turn off 'Confirm email' in your Supabase Dashboard (Authentication -> Providers -> Email -> Confirm email).",
      };
    }
    return { error: signInError.message };
  }

  redirect("/dashboard");
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
          "Email verification is currently required by your Supabase project settings. Turn off 'Confirm email' in Supabase Dashboard (Authentication -> Providers -> Email) to sign in directly.",
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
