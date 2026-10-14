'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { getExportOptions, startExport } from '@/lib/api'
import { formatGb, runsOn } from '@/lib/post-training'
import type { JobDetail } from '@/lib/types'
import StatusBadge from '@/components/status-badge'

const LOCAL_MODE = process.env.NEXT_PUBLIC_LOCAL_MODE === 'true'
const LIVE = ['pending', 'provisioning', 'running', 'checkpointing', 'completed']

/** Turn a finished training job into a GGUF + Modelfile for Ollama. */
export default function ExportPanel({ job }: { job: JobDetail }) {
  const router = useRouter()
  const { data, isLoading } = useQuery({
    queryKey: ['export-options', job.job_id],
    queryFn: () => getExportOptions(job.job_id),
  })
  const [chosen, setChosen] = useState<string | null>(null)
  const [hfToken, setHfToken] = useState('')
  const exportMutation = useMutation({
    mutationFn: (quant: string) => startExport(job.job_id, quant, hfToken || undefined),
    onSuccess: (res) => router.push(`/jobs/${res.job_id}`),
  })

  const existing = new Map(
    job.exports.filter((e) => LIVE.includes(e.status)).map((e) => [e.quant, e]),
  )
  const quant = chosen ?? data?.recommended ?? 'Q4_K_M'
  const already = existing.get(quant)

  return (
    <section className="border border-line-soft rounded-[10px] p-6">
      <h2 className="text-[1.1rem] font-bold text-ink mb-1">Run it locally</h2>
      <p className="text-sm text-ink-5 mb-5">
        Merge the fine-tune into the base model and quantize it to GGUF, with an Ollama
        Modelfile that uses the same prompt format it was trained on.
        {LOCAL_MODE
          ? ' Runs on this machine, free.'
          : ' Runs as a short job, billed like training.'}
      </p>

      {isLoading || !data ? (
        <p className="text-sm text-ink-7">Loading formats…</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5" role="radiogroup">
          {data.options.map((o) => {
            const selected = o.quant === quant
            const done = existing.get(o.quant)
            return (
              <button
                key={o.quant}
                role="radio"
                aria-checked={selected}
                onClick={() => setChosen(o.quant)}
                className={`text-left rounded-[8px] border p-4 transition-colors ${
                  selected ? 'border-ink bg-muted' : 'border-line hover:border-ink-7'
                }`}
              >
                <p className="font-mono font-semibold text-sm text-ink">
                  {o.quant}
                  {o.quant === data.recommended && (
                    <span className="ml-2 text-[0.7rem] font-sans font-medium text-ink-5">recommended</span>
                  )}
                </p>
                <p className="text-xs text-ink-5 mt-1">
                  {formatGb(o.file_gb)} file · needs {formatGb(o.memory_gb)}
                </p>
                <p className="text-xs text-ink-7 mt-1">{runsOn(o.memory_gb)}</p>
                <p className="text-xs text-ink-5 mt-2">
                  {done
                    ? `Exported — ${done.status}`
                    : LOCAL_MODE
                    ? `Free · ~${o.estimated_minutes} min`
                    : `~$${o.estimated_cost.toFixed(2)} · ~${o.estimated_minutes} min`}
                </p>
              </button>
            )
          })}
        </div>
      )}

      {!already && (
        <details className="mb-4">
          <summary className="text-xs text-ink-7 cursor-pointer">
            Gated base model? Add a Hugging Face token
          </summary>
          <input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_…"
            className="mt-2 w-full sm:w-80 border border-line rounded-[6px] px-3 py-2 text-sm font-mono bg-paper text-ink"
          />
        </details>
      )}

      <div className="flex items-center gap-4 flex-wrap">
        {already ? (
          <Link
            href={`/jobs/${already.job_id}`}
            className="text-sm px-5 py-3 rounded-[6px] font-semibold bg-inverse text-white no-underline hover:bg-inverse-hover transition-colors"
          >
            View {quant} export
          </Link>
        ) : (
          <button
            onClick={() => exportMutation.mutate(quant)}
            disabled={exportMutation.isPending || !data}
            className="text-sm px-5 py-3 rounded-[6px] font-semibold bg-inverse text-white hover:bg-inverse-hover transition-colors disabled:opacity-50"
          >
            {exportMutation.isPending ? 'Starting…' : `Export to ${quant}`}
          </button>
        )}
        {LOCAL_MODE && (
          <p className="text-xs text-ink-7">
            Or from the terminal: <code className="font-mono">heulistic export {job.job_id}</code>
          </p>
        )}
      </div>
      {exportMutation.isError && (
        <p className="text-sm text-danger mt-3">{(exportMutation.error as Error).message}</p>
      )}

      {job.exports.length > 0 && (
        <ul className="mt-5 space-y-2">
          {job.exports.map((e) => (
            <li key={e.job_id} className="flex items-center gap-3 text-sm">
              <StatusBadge status={e.status} />
              <Link href={`/jobs/${e.job_id}`} className="font-mono text-ink">
                {e.quant}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
