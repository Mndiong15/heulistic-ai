'use client'

import { useEffect, useState } from 'react'

import { currentTheme, nextTheme, setTheme, type Theme } from '@/lib/theme'

/** Switches between light and dark. The theme is already applied before React
 * loads (see themeInitScript); this only reads it and flips it. */
export default function ThemeToggle({ className = '' }: { className?: string }) {
  const [theme, setThemeState] = useState<Theme | null>(null)
  useEffect(() => setThemeState(currentTheme()), [])

  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={() => {
        const next = nextTheme(currentTheme())
        setTheme(next)
        setThemeState(next)
      }}
      className={`flex items-center justify-center w-8 h-8 rounded-full text-ink-5 hover:text-ink hover:bg-ink/5 transition-colors border-none bg-transparent cursor-pointer ${className}`}
    >
      {/* Rendered only once mounted, so the server never guesses the icon wrong. */}
      {theme === 'dark' ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : theme === 'light' ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : null}
    </button>
  )
}
