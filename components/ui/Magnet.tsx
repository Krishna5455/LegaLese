"use client";

import React, { useRef, type MouseEvent } from "react";

interface MagnetProps {
  children: React.ReactNode;
  className?: string;
  strength?: number; // Distance multiplier
}

export function Magnet({ children, className = "", strength = 0.25 }: MagnetProps) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const { left, top, width, height } = ref.current.getBoundingClientRect();
    const centerX = left + width / 2;
    const centerY = top + height / 2;
    const deltaX = (e.clientX - centerX) * strength;
    const deltaY = (e.clientY - centerY) * strength;
    ref.current.style.transform = `translate3d(${deltaX}px, ${deltaY}px, 0)`;
  };

  const handleMouseLeave = () => {
    if (ref.current) {
      ref.current.style.transform = "translate3d(0px, 0px, 0)";
    }
  };

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={`inline-block transition-transform duration-200 ease-out ${className}`}
    >
      {children}
    </div>
  );
}
