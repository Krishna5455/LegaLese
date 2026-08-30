import React, { ReactNode } from "react";
import { FileText } from "lucide-react";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
  secondaryAction?: ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  secondaryAction,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`rounded-xl border border-dashed border-[#E7E5E2] bg-[#F7F7F5]/50 p-10 sm:p-14 text-center flex flex-col items-center justify-center space-y-4 ${className}`}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white border border-[#E7E5E2] text-[#666666] shadow-2xs">
        {icon || <FileText className="w-5 h-5 text-[#8A8A8A]" />}
      </div>

      <div className="space-y-1 max-w-sm mx-auto">
        <h4 className="text-[16px] font-semibold text-[#171717]">{title}</h4>
        <p className="text-[13px] sm:text-[14px] text-[#666666] leading-relaxed">
          {description}
        </p>
      </div>

      {(action || secondaryAction) && (
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          {action}
          {secondaryAction}
        </div>
      )}
    </div>
  );
}
