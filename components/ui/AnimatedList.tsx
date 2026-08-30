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
          className="transition-all duration-300 ease-out animate-fadeIn"
          style={{
            animationDelay: `${index * 60}ms`,
            animationFillMode: "both",
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}
