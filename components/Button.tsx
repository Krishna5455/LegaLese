import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Loader2 } from "lucide-react";

export type ButtonVariant = "primary" | "secondary" | "accent" | "outline" | "ghost";
export type ButtonSize = "sm" | "md" | "lg";

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-[#171717] text-white hover:bg-[#262626] border border-[#171717] shadow-xs active:scale-[0.99]",
  secondary:
    "bg-white text-[#171717] border border-[#E7E5E2] hover:bg-[#F7F7F5] hover:border-[#D4D2CD] shadow-2xs active:scale-[0.99]",
  accent:
    "bg-[#C2410C] text-white hover:bg-[#9A3412] border border-[#C2410C] shadow-xs active:scale-[0.99]",
  outline:
    "bg-transparent text-[#171717] border border-[#E7E5E2] hover:bg-[#F7F7F5] hover:border-[#D4D2CD] active:scale-[0.99]",
  ghost:
    "bg-transparent text-[#666666] hover:text-[#171717] hover:bg-[#F0EFEA] border border-transparent active:scale-[0.99]",
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-[13px] font-medium rounded-lg gap-1.5",
  md: "h-9.5 px-4 text-[14px] font-medium rounded-lg gap-2",
  lg: "h-11 px-5 text-[15px] font-medium rounded-lg gap-2.5",
};

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
  children: ReactNode;
};

export function Button({
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className = "",
  disabled,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center transition-all duration-150 select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#171717]/20 disabled:cursor-not-allowed disabled:opacity-50 ${
        variantStyles[variant]
      } ${sizeStyles[size]} ${fullWidth ? "w-full" : ""} ${className}`}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
      ) : (
        leftIcon && <span className="shrink-0">{leftIcon}</span>
      )}
      <span>{children}</span>
      {!isLoading && rightIcon && <span className="shrink-0">{rightIcon}</span>}
    </button>
  );
}

export type ButtonLinkProps = {
  href: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
  className?: string;
  children: ReactNode;
};

export function ButtonLink({
  href,
  variant = "primary",
  size = "md",
  leftIcon,
  rightIcon,
  fullWidth = false,
  className = "",
  children,
}: ButtonLinkProps) {
  return (
    <Link
      href={href}
      className={`inline-flex items-center justify-center transition-all duration-150 select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#171717]/20 ${
        variantStyles[variant]
      } ${sizeStyles[size]} ${fullWidth ? "w-full" : ""} ${className}`}
    >
      {leftIcon && <span className="shrink-0">{leftIcon}</span>}
      <span>{children}</span>
      {rightIcon && <span className="shrink-0">{rightIcon}</span>}
    </Link>
  );
}
