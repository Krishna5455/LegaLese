"use client";

import { type ReactNode, Children } from "react";

type AnimatedListProps = {
  children: ReactNode;
  className?: string;
};

export function AnimatedList({ children, className = "" }: AnimatedListProps) {
  const childrenArray = Children.toArray(children);

  return (
    <div className={`space-y-3 ${className}`}>
      {childrenArray.map((child, index) => (
        <div
          key={index}
          className="animate-fadeIn"
          style={{
            animationDelay: `${Math.min(index * 45, 270)}ms`,
            animationFillMode: "both",
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}
