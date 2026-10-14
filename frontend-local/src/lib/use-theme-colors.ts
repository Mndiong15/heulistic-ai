'use client'

import { useEffect, useState } from 'react'

import { themeColor } from './theme'

// Charts take colour values rather than classes, so they read the live theme
// variables and re-read them when the theme flips.

const LIGHT = {
  ink: '#1a1a1a', ink5: '#666666', ink7: '#999999', line: '#e9ecef', tint: '#f0f0f0',
  paper: '#ffffff', danger: '#dc3545', ok: '#22c55e', okFill: 'rgb(34 197 94 / 0.15)',
}
export type ThemeColors = typeof LIGHT

function read(): ThemeColors {
  return {
    ink: themeColor('ink'),
    ink5: themeColor('ink-5'),
    ink7: themeColor('ink-7'),
    line: themeColor('line-soft'),
    tint: themeColor('tint'),
    paper: themeColor('paper'),
    danger: themeColor('danger'),
    ok: themeColor('ok'),
    okFill: themeColor('ok', 0.15),
  }
}

export function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState<ThemeColors>(LIGHT)
  useEffect(() => {
    setColors(read())
    const observer = new MutationObserver(() => setColors(read()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])
  return colors
}
