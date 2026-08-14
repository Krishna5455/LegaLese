export function Footer() {
  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto flex max-w-5xl flex-col gap-2 px-6 py-8 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm font-medium text-foreground">LegaLese</p>
        <p className="text-sm text-muted">
          Not legal advice. Always consult a qualified attorney for binding
          decisions.
        </p>
      </div>
    </footer>
  );
}
