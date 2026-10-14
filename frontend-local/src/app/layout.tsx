import type { Metadata } from 'next'
import './globals.css'
import Providers from './providers'
import { themeInitScript } from '@/lib/theme'

export const metadata: Metadata = {
  title: 'Heulistic',
  description: 'Fine-tune LLMs locally with Axolotl.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the theme script sets the `dark` class before
    // React hydrates, so the server-rendered <html> never matches exactly.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
