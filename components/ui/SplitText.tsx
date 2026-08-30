"use client";

import React, { useEffect, useState } from "react";

interface SplitTextProps {
  text: string;
  className?: string;
  delay?: number; // ms per word
}

export function SplitText({ text, className = "", delay = 80 }: SplitTextProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsMounted(true), 50);
    return () => clearTimeout(timer);
  }, []);

  const words = text.split(" ");

  return (
    <span className={`inline-flex flex-wrap gap-x-[0.25em] ${className}`}>
      {words.map((word, idx) => (
        <span
          key={idx}
          className={`inline-block transition-all duration-700 cubic-bezier(0.16, 1, 0.3, 1) ${
            isMounted
              ? "opacity-100 translate-y-0 filter-none"
              : "opacity-0 translate-y-3 blur-[4px]"
          }`}
          style={{
            transitionDelay: `${idx * delay}ms`,
          }}
        >
          {word}
        </span>
      ))}
    </span>
  );
}
