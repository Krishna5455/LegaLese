export default function CreateLoading() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 sm:px-6 py-8 space-y-8 animate-pulse">
      <div className="space-y-2">
        <div className="h-7 w-48 rounded-lg bg-[#E7E5E2]" />
        <div className="h-4 w-72 rounded-lg bg-[#E7E5E2]/60" />
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Form Skeleton */}
        <div className="lg:col-span-2 space-y-6 rounded-xl border border-[#E7E5E2] bg-white p-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-20 rounded-xl border border-[#E7E5E2] bg-[#F7F7F5]" />
            ))}
          </div>
          <div className="border-t border-[#E7E5E2] pt-6 space-y-4">
            <div className="h-5 w-40 rounded bg-[#E7E5E2]" />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="h-10 rounded-lg bg-[#F7F7F5]" />
              <div className="h-10 rounded-lg bg-[#F7F7F5]" />
            </div>
            <div className="h-24 rounded-lg bg-[#F7F7F5]" />
          </div>
        </div>

        {/* Sidebar Skeleton */}
        <div className="space-y-6">
          <div className="rounded-xl border border-[#E7E5E2] bg-white p-6 space-y-4 shadow-sm">
            <div className="h-4 w-32 rounded bg-[#E7E5E2]" />
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-14 rounded-lg bg-[#F7F7F5]" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
