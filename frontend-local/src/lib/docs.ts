import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { marked } from 'marked'

const DOCS_DIR = path.join(process.cwd(), 'src/content/docs')

export interface DocMeta {
  slug: string
  title: string
  description: string
  order: number
}

export interface Doc extends DocMeta {
  html: string
}

export function getAllDocs(): DocMeta[] {
  return fs
    .readdirSync(DOCS_DIR)
    .filter((f) => f.endsWith('.md'))
    .map((file) => {
      const slug = file.replace(/\.md$/, '')
      const { data } = matter(fs.readFileSync(path.join(DOCS_DIR, file), 'utf8'))
      return {
        slug,
        title: data.title as string,
        description: data.description as string,
        order: (data.order as number) ?? 99,
      }
    })
    .sort((a, b) => a.order - b.order)
}

export function getDoc(slug: string): Doc | null {
  const filePath = path.join(DOCS_DIR, `${slug}.md`)
  if (!fs.existsSync(filePath)) return null
  const { data, content } = matter(fs.readFileSync(filePath, 'utf8'))
  const html = marked(content) as string
  return {
    slug,
    title: data.title as string,
    description: data.description as string,
    order: (data.order as number) ?? 99,
    html,
  }
}
