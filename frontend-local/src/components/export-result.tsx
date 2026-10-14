'use client'

import { useState } from 'react'

import { exportName, ollamaSteps } from '@/lib/post-training'
import type { JobDetail } from '@/lib/types'

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        })
      }}
      className="text-xs text-ink-5 hover:text-ink transition-colors"
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function downloadText(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  URL.revokeObjectURL(url)
}

/** What to do with a finished export: the GGUF, its Modelfile, and Ollama. */
export default function ExportResult({ job, onDownload }: { job: JobDetail; onDownload: () => void }) {
  if (job.status !== 'completed' || !job.modelfile) return null
  const name = exportName(job.source_job_id ?? job.job_id)
  const steps = ollamaSteps(name)

  return (
    <section className="border border-line-soft rounded-[10px] p-6">
      <h2 className="text-[1.1rem] font-bold text-ink mb-1">Run it with Ollama</h2>
      <p className="text-sm text-ink-5 mb-5">
        Put the GGUF and the Modelfile in one folder, then create the model from it.
      </p>

      <ol className="space-y-5 text-sm text-ink">
        <li>
          <p className="font-semibold mb-2">1. Download both files</p>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={onDownload}
              className="text-sm px-4 py-2 rounded-[6px] font-semibold bg-inverse text-white hover:bg-inverse-hover transition-colors"
            >
              Download GGUF
            </button>
            <button
              onClick={() => downloadText('Modelfile', job.modelfile!)}
              className="text-sm px-4 py-2 rounded-[6px] font-semibold border border-line text-ink hover:border-ink-7 transition-colors"
            >
              Download Modelfile
            </button>
          </div>
        </li>
        <li>
          <div className="flex items-center justify-between mb-2">
            <p className="font-semibold">2. The Modelfile</p>
            <CopyButton text={job.modelfile} />
          </div>
          <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-canvas border border-line-soft rounded-[6px] p-3 text-ink-2">
            {job.modelfile}
          </pre>
        </li>
        <li>
          <div className="flex items-center justify-between mb-2">
            <p className="font-semibold">3. Create and run it</p>
            <CopyButton text={steps.join('\n')} />
          </div>
          <pre className="text-xs font-mono bg-inverse text-[#f0f0f0] rounded-[6px] p-3 overflow-x-auto">
            {steps.join('\n')}
          </pre>
        </li>
      </ol>
    </section>
  )
}
