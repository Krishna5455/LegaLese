export default function DocumentDetailLoading() {
  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 py-8 space-y-6 animate-pulse">
      {/* Back button skeleton */}
      <div className="h-4 w-36 rounded bg-[#E7E5E2]" />

      {/* Detail Header Skeleton */}
      <div className="space-y-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="h-5 w-20 rounded bg-[#E7E5E2]" />
              <div className="h-5 w-24 rounded-full bg-[#E7E5E2]" />
              <div className="h-5 w-36 rounded bg-[#E7E5E2]/70" />
            </div>
            <div className="h-8 w-64 rounded bg-[#E7E5E2]" />
            <div className="h-4 w-48 rounded bg-[#E7E5E2]/60" />
          </div>

          <div className="flex items-center gap-2.5">
            <div className="h-9 w-32 rounded-lg bg-[#E7E5E2]" />
            <div className="h-9 w-36 rounded-lg bg-[#E7E5E2]/80" />
          </div>
        </div>
      </div>

      {/* Workspace Tabs & Grid Skeleton */}
      <div className="rounded-xl border border-[#E7E5E2] bg-white p-6 space-y-6 shadow-xs">
        <div className="flex items-center gap-3 border-b border-[#E7E5E2] pb-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-8 w-24 rounded-lg bg-[#F7F7F5]" />
          ))}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-32 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5]/50 p-4 space-y-3">
              <div className="h-4 w-40 rounded bg-[#E7E5E2]" />
              <div className="h-3 w-full rounded bg-[#E7E5E2]/60" />
              <div className="h-3 w-3/4 rounded bg-[#E7E5E2]/60" />
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
