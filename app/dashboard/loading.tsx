export default function DashboardLoading() {
  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8 space-y-8 animate-pulse">
      {/* Header Skeleton */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-[#E7E5E2] pb-6">
        <div className="space-y-2">
          <div className="h-7 w-48 rounded-lg bg-[#E7E5E2]" />
          <div className="h-4 w-72 rounded-lg bg-[#E7E5E2]/60" />
        </div>
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-32 rounded-lg bg-[#E7E5E2]" />
          <div className="h-9 w-32 rounded-lg bg-[#E7E5E2]/70" />
        </div>
      </div>

      {/* 2 Spotlight Cards Skeleton */}
      <div className="grid gap-5 md:grid-cols-2">
        <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 space-y-4 shadow-xs">
          <div className="flex items-start justify-between">
            <div className="h-10 w-10 rounded-xl bg-[#E7E5E2]" />
            <div className="h-3.5 w-24 rounded bg-[#E7E5E2]" />
          </div>
          <div className="space-y-2">
            <div className="h-5 w-40 rounded bg-[#E7E5E2]" />
            <div className="h-3.5 w-full rounded bg-[#E7E5E2]/70" />
            <div className="h-3.5 w-3/4 rounded bg-[#E7E5E2]/70" />
          </div>
          <div className="h-4 w-28 rounded bg-[#E7E5E2]" />
        </div>

        <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 space-y-4 shadow-xs">
          <div className="flex items-start justify-between">
            <div className="h-10 w-10 rounded-xl bg-[#E7E5E2]" />
            <div className="h-3.5 w-24 rounded bg-[#E7E5E2]" />
          </div>
          <div className="space-y-2">
            <div className="h-5 w-44 rounded bg-[#E7E5E2]" />
            <div className="h-3.5 w-full rounded bg-[#E7E5E2]/70" />
            <div className="h-3.5 w-3/4 rounded bg-[#E7E5E2]/70" />
          </div>
          <div className="h-4 w-28 rounded bg-[#E7E5E2]" />
        </div>
      </div>

      {/* 3 Metric Cards Skeleton */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-xl border border-[#E7E5E2] bg-white p-5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="h-3 w-20 rounded bg-[#E7E5E2]" />
              <div className="h-4 w-4 rounded bg-[#E7E5E2]" />
            </div>
            <div className="h-7 w-12 rounded bg-[#E7E5E2]" />
          </div>
        ))}
      </div>

      {/* Upload Box Skeleton */}
      <div className="space-y-3">
        <div className="space-y-1">
          <div className="h-5 w-48 rounded bg-[#E7E5E2]" />
          <div className="h-3.5 w-72 rounded bg-[#E7E5E2]/60" />
        </div>
        <div className="h-36 rounded-xl border border-dashed border-[#E7E5E2] bg-white" />
      </div>

      {/* Recent Documents Table Skeleton */}
      <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-5 shadow-sm">
        <div className="flex items-center justify-between border-b border-[#E7E5E2] pb-4">
          <div className="space-y-1">
            <div className="h-5 w-36 rounded bg-[#E7E5E2]" />
            <div className="h-3.5 w-60 rounded bg-[#E7E5E2]/60" />
          </div>
          <div className="h-5 w-16 rounded bg-[#E7E5E2]" />
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5]/50" />
          ))}
        </div>
      </div>
    </main>
  );
}
