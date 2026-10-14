// Light and dark themes. The choice is stored in localStorage; until the user
// makes one, the system preference decides. Colours themselves are CSS
// variables in globals.css, switched by a `dark` class on <html>.

export type Theme = 'light' | 'dark'

export const THEME_KEY = 'theme'

export function resolveTheme(stored: string | null, systemDark: boolean): Theme {
  if (stored === 'light' || stored === 'dark') return stored
  return systemDark ? 'dark' : 'light'
}

export function nextTheme(current: Theme): Theme {
  return current === 'dark' ? 'light' : 'dark'
}

/** Inline in <head> so the first paint is already in the right theme. Mirrors
 * resolveTheme; storage can throw (private windows, blocked cookies). */
export const themeInitScript = `(function(){var s=null;try{s=localStorage.getItem('${THEME_KEY}')}catch(e){}var d=s==='dark'||(s!=='light'&&window.matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.classList.toggle('dark',d)})()`

export function currentTheme(): Theme {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

export function setTheme(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    // Private window or blocked storage: the choice lasts for this page only.
  }
}

/** A theme colour as a CSS colour string, for libraries that need a value
 * rather than a class (Recharts). Reads the live CSS variable. */
export function themeColor(token: string, alpha = 1): string {
  const channels = getComputedStyle(document.documentElement).getPropertyValue(`--c-${token}`).trim()
  return channels ? `rgb(${channels} / ${alpha})` : 'currentColor'
}
