import Link from 'next/link'
import Navbar from '@/components/navbar'
import Footer from '@/components/footer'
import { getAllDocs } from '@/lib/docs'

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  const docs = getAllDocs()

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <div className="flex-1 mt-[80px]">
        {/* Upsell banner */}
        <div className="bg-inverse text-white py-3 px-4">
          <div className="container flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-[0.875rem] text-[#ccc] text-center sm:text-left leading-snug">
              Train locally with what you have. When your hardware ceiling isn&apos;t enough,{' '}
              <span className="text-white font-medium">managed infrastructure removes that limit.</span>
            </p>
            <a
              href="https://heulistic.com"
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 text-[0.8rem] font-semibold text-white border border-white/30 hover:border-white hover:bg-white/10 transition-colors px-4 py-1.5 rounded-full no-underline whitespace-nowrap"
            >
              Try Heulistic Cloud →
            </a>
          </div>
        </div>

        <div className="container py-12">
          <div className="flex gap-12 items-start">

            {/* Sidebar */}
            <aside className="hidden md:block w-56 shrink-0 sticky top-[136px]">
              <p className="label-caps mb-4">Documentation</p>
              <nav className="flex flex-col gap-0.5">
                {docs.map((doc) => (
                  <Link
                    key={doc.slug}
                    href={`/docs/${doc.slug}`}
                    className="text-[0.9rem] text-ink-5 hover:text-ink py-1.5 px-3 rounded-lg hover:bg-tint transition-colors no-underline font-medium"
                  >
                    {doc.title}
                  </Link>
                ))}
              </nav>
            </aside>

            {/* Content */}
            <main className="flex-1 min-w-0 max-w-3xl">
              {children}
            </main>
          </div>
        </div>
      </div>

      <Footer />
    </div>
  )
}
