import { notFound } from 'next/navigation'
import type { Metadata } from 'next'
import Link from 'next/link'
import { getAllDocs, getDoc } from '@/lib/docs'

type Props = { params: { slug: string } }

export function generateStaticParams() {
  return getAllDocs().map((doc) => ({ slug: doc.slug }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const doc = getDoc(params.slug)
  if (!doc) return {}
  return {
    title: `${doc.title} — Heulistic Docs`,
    description: doc.description,
  }
}

export default function DocPage({ params }: Props) {
  const doc = getDoc(params.slug)
  if (!doc) notFound()

  const allDocs = getAllDocs()
  const currentIndex = allDocs.findIndex((d) => d.slug === params.slug)
  const prev = allDocs[currentIndex - 1] ?? null
  const next = allDocs[currentIndex + 1] ?? null

  return (
    <article>
      <header className="mb-10 pb-8 border-b border-ink/5">
        <p className="label-caps mb-3">Docs</p>
        <h1 className="text-[2.4rem] font-[800] text-ink tracking-[-0.03em] leading-[1.1] mb-3">
          {doc.title}
        </h1>
        <p className="text-[1.05rem] text-ink-5 leading-[1.6]">{doc.description}</p>
      </header>

      <div
        className="docs-content"
        dangerouslySetInnerHTML={{ __html: doc.html }}
      />

      {(prev || next) && (
        <div className="mt-16 pt-8 border-t border-ink/5 flex justify-between gap-4">
          {prev ? (
            <Link href={`/docs/${prev.slug}`} className="group flex flex-col gap-1 no-underline">
              <span className="text-[0.75rem] font-semibold text-ink-7 uppercase tracking-wider">Previous</span>
              <span className="text-[0.95rem] font-semibold text-ink group-hover:underline">← {prev.title}</span>
            </Link>
          ) : <div />}
          {next && (
            <Link href={`/docs/${next.slug}`} className="group flex flex-col gap-1 no-underline text-right ml-auto">
              <span className="text-[0.75rem] font-semibold text-ink-7 uppercase tracking-wider">Next</span>
              <span className="text-[0.95rem] font-semibold text-ink group-hover:underline">{next.title} →</span>
            </Link>
          )}
        </div>
      )}
    </article>
  )
}
