export default function ViewGeneratedDocumentLoading() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8 space-y-8 animate-pulse">
      {/* Back button skeleton */}
      <div className="h-4 w-32 rounded bg-[#E7E5E2]" />

      {/* Header Banner Skeleton */}
      <div className="rounded-xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-6 shadow-xs">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-2">
            <div className="h-5 w-40 rounded-full bg-[#E7E5E2]" />
            <div className="h-8 w-72 rounded bg-[#E7E5E2]" />
            <div className="h-4 w-48 rounded bg-[#E7E5E2]/60" />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="h-9 w-28 rounded-lg bg-[#E7E5E2]" />
            <div className="h-9 w-20 rounded-lg bg-[#E7E5E2]/70" />
            <div className="h-9 w-24 rounded-lg bg-[#E7E5E2]/70" />
          </div>
        </div>

        {/* Tab Switcher Skeleton */}
        <div className="flex items-center gap-2 p-1 bg-[#F7F7F5] rounded-lg border border-[#E7E5E2]">
          <div className="h-8 w-32 rounded bg-white shadow-xs" />
          <div className="h-8 w-44 rounded bg-[#E7E5E2]/50" />
          <div className="h-8 w-40 rounded bg-[#E7E5E2]/50" />
        </div>
      </div>

      {/* Content Preview Skeleton */}
      <div className="rounded-xl border border-[#E7E5E2] bg-white p-6 sm:p-8 space-y-6 shadow-xs">
        <div className="space-y-3">
          <div className="h-6 w-56 rounded bg-[#E7E5E2]" />
          <div className="h-4 w-full rounded bg-[#E7E5E2]/60" />
          <div className="h-4 w-5/6 rounded bg-[#E7E5E2]/60" />
        </div>
        <div className="space-y-3 pt-4 border-t border-[#E7E5E2]">
          <div className="h-6 w-48 rounded bg-[#E7E5E2]" />
          <div className="h-4 w-full rounded bg-[#E7E5E2]/60" />
          <div className="h-4 w-4/5 rounded bg-[#E7E5E2]/60" />
        </div>
      </div>
    </main>
  );
}
