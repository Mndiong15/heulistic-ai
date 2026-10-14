import { runInNewContext } from 'node:vm'
import { describe, expect, it } from 'vitest'

import { THEME_KEY, nextTheme, resolveTheme, themeInitScript } from './theme'

describe('resolveTheme', () => {
  it('follows the system until the user chooses', () => {
    expect(resolveTheme(null, true)).toBe('dark')
    expect(resolveTheme(null, false)).toBe('light')
  })

  it('a saved choice wins over the system', () => {
    expect(resolveTheme('light', true)).toBe('light')
    expect(resolveTheme('dark', false)).toBe('dark')
  })

  it('ignores anything else that ended up in storage', () => {
    expect(resolveTheme('purple', true)).toBe('dark')
  })
})

describe('nextTheme', () => {
  it('toggles', () => {
    expect(nextTheme('dark')).toBe('light')
    expect(nextTheme('light')).toBe('dark')
  })
})

// The init script runs inline in <head>, before React, so the first paint is
// already in the right theme. It is a string; run it against a fake browser.
describe('themeInitScript', () => {
  function run(stored: string | null, systemDark: boolean, storageThrows = false) {
    const classes = new Set<string>()
    runInNewContext(themeInitScript, {
      localStorage: {
        getItem: (k: string) => {
          if (storageThrows) throw new Error('blocked')
          return k === THEME_KEY ? stored : null
        },
      },
      window: { matchMedia: () => ({ matches: systemDark }) },
      document: {
        documentElement: {
          classList: { toggle: (c: string, on: boolean) => (on ? classes.add(c) : classes.delete(c)) },
        },
      },
    })
    return classes.has('dark')
  }

  it('applies the same rule as resolveTheme', () => {
    expect(run(null, true)).toBe(true)
    expect(run('light', true)).toBe(false)
    expect(run('dark', false)).toBe(true)
  })

  it('still follows the system when storage is blocked', () => {
    expect(run(null, true, true)).toBe(true)
  })
})
