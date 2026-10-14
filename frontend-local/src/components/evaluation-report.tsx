'use client'

import { summarizeEvaluation } from '@/lib/post-training'
import type { JobDetail } from '@/lib/types'

/** Base model vs fine-tune, scored on rows held out from training. */
export default function EvaluationReport({ report }: { report: JobDetail['evaluation'] }) {
  if (!report) return null

  if ('skipped' in report) {
    return (
      <section className="border border-line-soft rounded-[10px] p-6">
        <h2 className="text-[1.1rem] font-bold text-ink mb-1">Before vs after</h2>
        <p className="text-sm text-ink-5">{report.skipped}</p>
      </section>
    )
  }

  const summary = summarizeEvaluation(report)!
  const tone = summary.better
    ? 'text-ok-ink'
    : summary.changePct > 0
    ? 'text-warn-strong'
    : 'text-ink'

  return (
    <section className="border border-line-soft rounded-[10px] p-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap mb-1">
        <h2 className="text-[1.1rem] font-bold text-ink">Before vs after</h2>
        <p className={`text-sm font-semibold ${tone}`}>{summary.headline}</p>
      </div>
      <p className="text-xs text-ink-7 mb-5">
        Both models answered rows held out from training — the fine-tune never saw them. Lower
        loss and perplexity mean the model finds the expected answers more likely.
      </p>

      <div className="grid grid-cols-2 gap-3 mb-6">
        {(
          [
            ['Base model', report.base],
            ['Fine-tuned', report.tuned],
          ] as const
        ).map(([label, scores]) => (
          <div key={label} className="bg-muted rounded-[8px] p-4">
            <p className="text-xs text-ink-7 mb-1">{label}</p>
            <p className="text-[1.3rem] font-bold text-ink font-mono">{scores.loss.toFixed(3)}</p>
            <p className="text-xs text-ink-5">loss · perplexity {scores.perplexity.toFixed(2)}</p>
          </div>
        ))}
      </div>

      {report.samples.length > 0 && (
        <div className="space-y-5">
          <h3 className="text-sm font-semibold text-ink">Side by side</h3>
          {report.samples.map((s, i) => (
            <div key={i} className="border-t border-line-soft pt-4">
              <details className="mb-3">
                <summary className="text-xs text-ink-7 cursor-pointer">Prompt</summary>
                <pre className="mt-2 text-xs text-ink-5 font-mono whitespace-pre-wrap break-words">{s.prompt}</pre>
              </details>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {(
                  [
                    ['Expected', s.reference],
                    ['Base model', s.base],
                    ['Fine-tuned', s.tuned],
                  ] as const
                ).map(([label, text]) => (
                  <div key={label}>
                    <p className="text-xs font-semibold text-ink-5 mb-1">{label}</p>
                    <pre className="text-xs text-ink font-mono whitespace-pre-wrap break-words bg-canvas border border-line-soft rounded-[6px] p-3 max-h-48 overflow-y-auto">
                      {text || '—'}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
