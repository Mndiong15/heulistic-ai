import type { Config } from 'tailwindcss'

const config: Config = {
  // Toggled by the theme script in the root layout and the navbar switch.
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      colors: {
        // Theme tokens — see globals.css for the light and dark values.
        'ink': 'rgb(var(--c-ink) / <alpha-value>)',
        'ink-2': 'rgb(var(--c-ink-2) / <alpha-value>)',
        'ink-3': 'rgb(var(--c-ink-3) / <alpha-value>)',
        'ink-4': 'rgb(var(--c-ink-4) / <alpha-value>)',
        'ink-5': 'rgb(var(--c-ink-5) / <alpha-value>)',
        'ink-6': 'rgb(var(--c-ink-6) / <alpha-value>)',
        'ink-7': 'rgb(var(--c-ink-7) / <alpha-value>)',
        'ink-8': 'rgb(var(--c-ink-8) / <alpha-value>)',
        'ink-9': 'rgb(var(--c-ink-9) / <alpha-value>)',
        'line-strong': 'rgb(var(--c-line-strong) / <alpha-value>)',
        'line': 'rgb(var(--c-line) / <alpha-value>)',
        'line-2': 'rgb(var(--c-line-2) / <alpha-value>)',
        'line-soft': 'rgb(var(--c-line-soft) / <alpha-value>)',
        'tint': 'rgb(var(--c-tint) / <alpha-value>)',
        'tint-2': 'rgb(var(--c-tint-2) / <alpha-value>)',
        'muted': 'rgb(var(--c-muted) / <alpha-value>)',
        'canvas': 'rgb(var(--c-canvas) / <alpha-value>)',
        'paper': 'rgb(var(--c-paper) / <alpha-value>)',
        'inverse': 'rgb(var(--c-inverse) / <alpha-value>)',
        'inverse-hover': 'rgb(var(--c-inverse-hover) / <alpha-value>)',
        'danger': 'rgb(var(--c-danger) / <alpha-value>)',
        'danger-bg': 'rgb(var(--c-danger-bg) / <alpha-value>)',
        'danger-line': 'rgb(var(--c-danger-line) / <alpha-value>)',
        'ok': 'rgb(var(--c-ok) / <alpha-value>)',
        'ok-2': 'rgb(var(--c-ok-2) / <alpha-value>)',
        'ok-ink': 'rgb(var(--c-ok-ink) / <alpha-value>)',
        'ok-bg': 'rgb(var(--c-ok-bg) / <alpha-value>)',
        'warn-ink': 'rgb(var(--c-warn-ink) / <alpha-value>)',
        'warn-strong': 'rgb(var(--c-warn-strong) / <alpha-value>)',
        'warn-bg': 'rgb(var(--c-warn-bg) / <alpha-value>)',
        'warn-bg-2': 'rgb(var(--c-warn-bg-2) / <alpha-value>)',
        'warn-line': 'rgb(var(--c-warn-line) / <alpha-value>)',
        'warn': 'rgb(var(--c-warn) / <alpha-value>)',
        'amber': 'rgb(var(--c-amber) / <alpha-value>)',
        'info-ink': 'rgb(var(--c-info-ink) / <alpha-value>)',
        'info-bg': 'rgb(var(--c-info-bg) / <alpha-value>)',
        brand: {
          black: '#1a1a1a',
          gray: '#333333',
          muted: '#f8f9fa',
          bg: '#fafafa',
          border: '#e9ecef',
          'border-dark': '#e0e0e0',
          secondary: '#666666',
          subtle: '#999999',
          danger: '#dc3545',
        },
      },
      letterSpacing: {
        hero: '-0.03em',
        title: '-0.02em',
      },
      boxShadow: {
        btn: '0 4px 20px rgba(0,0,0,0.1)',
        'btn-hover': '0 8px 30px rgba(0,0,0,0.15)',
        danger: '0 10px 40px rgba(220,53,69,0.15)',
      },
    },
  },
  plugins: [],
}

export default config
