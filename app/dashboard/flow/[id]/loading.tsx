import { Loader2 } from "lucide-react";

export default function Loading() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-12 flex flex-col items-center justify-center space-y-4">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#059669]/10 text-[#059669]">
        <Loader2 className="w-6 h-6 animate-spin" />
      </div>
      <p className="text-xs font-semibold uppercase tracking-wider text-[#8A8F98]">
        Loading LegalFlow Journey...
      </p>
    </main>
  );
}
