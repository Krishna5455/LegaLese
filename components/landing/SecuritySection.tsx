import { Lock, ShieldCheck, Database, Key } from "lucide-react";

export function SecuritySection() {
  return (
    <section id="security" className="py-20 bg-white border-b border-[#E5E5E3]">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-12">
        <div className="text-center space-y-3 max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F9F9F8] border border-[#E5E5E3] text-xs font-semibold text-[#111827]">
            <Lock className="w-3.5 h-3.5 text-[#059669]" />
            <span>Security & Privacy</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#111827]">
            Enterprise-grade data protection
          </h2>
          <p className="text-sm text-[#4B5563]">
            Your legal documents contain sensitive commercial agreements. We ensure strict tenant isolation and zero public model training.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">Zero AI Training</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Your uploaded contracts and generated clauses are never used to train public AI foundation models.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <Database className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">Row-Level Security (RLS)</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Postgres database enforced tenant isolation guarantees that only authorized users access their documents.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#F9F9F8] border border-[#E5E5E3] space-y-3">
            <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-700 w-fit">
              <Key className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-[#111827]">Encrypted Storage</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              All contract binaries, extracted text tokens, and export artifacts are encrypted in transit and at rest.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
