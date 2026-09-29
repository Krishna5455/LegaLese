export default function DocumentDetailLoading() {
  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8 space-y-6 animate-pulse">
      {/* Back button skeleton */}
      <div className="h-4 w-36 rounded bg-[#E7E5E2]" />

      {/* Detail Header Skeleton */}
      <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 sm:p-7 space-y-4 shadow-xs">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2.5">
            <div className="flex items-center gap-2">
              <div className="h-5 w-20 rounded bg-[#E7E5E2]" />
              <div className="h-5 w-24 rounded-full bg-[#E7E5E2]" />
              <div className="h-5 w-36 rounded bg-[#E7E5E2]/70" />
            </div>
            <div className="h-8 w-72 rounded bg-[#E7E5E2]" />
            <div className="h-4 w-52 rounded bg-[#E7E5E2]/60" />
          </div>

          <div className="flex items-center gap-2.5">
            <div className="h-9 w-32 rounded-lg bg-[#E7E5E2]" />
            <div className="h-9 w-36 rounded-lg bg-[#E7E5E2]/80" />
          </div>
        </div>
      </div>

      {/* Control Bar Skeleton */}
      <div className="rounded-xl border border-[#E7E5E2] bg-white p-3.5 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-2">
          <div className="h-7 w-24 rounded-lg bg-[#E7E5E2]" />
          <div className="h-7 w-28 rounded-lg bg-[#E7E5E2]/70 hidden md:block" />
        </div>
        <div className="h-6 w-32 rounded-full bg-[#E7E5E2]" />
      </div>

      {/* Two-Column Workspace Skeleton */}
      <div className="grid gap-6 lg:grid-cols-12 items-start">
        {/* Left Pane Skeleton: Contract Document */}
        <div className="lg:col-span-5 rounded-xl border border-[#E7E5E2] bg-white p-4 space-y-4 shadow-xs">
          <div className="h-5 w-40 rounded bg-[#E7E5E2]" />
          <div className="h-8 w-full rounded-lg bg-[#F7F7F5]" />
          <div className="space-y-3 pt-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="p-3 rounded-lg border border-[#E7E5E2]/70 space-y-2">
                <div className="h-3 w-32 rounded bg-[#E7E5E2]" />
                <div className="h-3 w-full rounded bg-[#E7E5E2]/50" />
                <div className="h-3 w-4/5 rounded bg-[#E7E5E2]/50" />
              </div>
            ))}
          </div>
        </div>

        {/* Right Pane Skeleton: Analysis */}
        <div className="lg:col-span-7 space-y-5">
          <div className="rounded-xl border border-[#E7E5E2] bg-white p-1 flex items-center gap-2 shadow-xs">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-8 w-20 rounded-lg bg-[#F7F7F5]" />
            ))}
          </div>

          <div className="rounded-2xl border border-[#E7E5E2] bg-white p-6 space-y-4 shadow-xs">
            <div className="h-5 w-48 rounded bg-[#E7E5E2]" />
            <div className="h-3.5 w-full rounded bg-[#E7E5E2]/60" />
            <div className="h-3.5 w-5/6 rounded bg-[#E7E5E2]/60" />
            <div className="h-3 w-full rounded-full bg-[#E7E5E2]/40 mt-3" />
            <div className="grid grid-cols-3 gap-3 pt-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 rounded-xl bg-[#F7F7F5]" />
              ))}
            </div>
          </div>

          <div className="space-y-3">
            {[1, 2].map((i) => (
              <div key={i} className="rounded-xl border border-[#E7E5E2] bg-white p-5 space-y-3 shadow-xs">
                <div className="h-4 w-36 rounded bg-[#E7E5E2]" />
                <div className="h-3 w-full rounded bg-[#E7E5E2]/60" />
                <div className="h-14 rounded-lg bg-[#F7F7F5]" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
