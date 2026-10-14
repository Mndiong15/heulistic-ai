// After training: the before/after report and running the exported model.

import type { EvaluationReport } from './types'

export interface EvaluationSummary {
  better: boolean
  changePct: number
  headline: string
}

// Below this, loss differences are noise on a few dozen examples.
const NEGLIGIBLE_PCT = 2

/** One honest line about the comparison, or null when there is no report. */
export function summarizeEvaluation(
  report: EvaluationReport | { skipped: string } | null | undefined,
): EvaluationSummary | null {
  if (!report || 'skipped' in report) return null
  const { base, tuned, examples } = report
  const changePct = Math.round(((tuned.loss - base.loss) / base.loss) * 100)
  const on = `on ${examples} held-out example${examples === 1 ? '' : 's'}`
  const headline =
    Math.abs(changePct) < NEGLIGIBLE_PCT
      ? `No meaningful change ${on}`
      : `Loss ${changePct < 0 ? 'down' : 'up'} ${Math.abs(changePct)}% ${on}`
  return { better: changePct < 0, changePct, headline }
}

/** The kind of machine that runs a model needing `memoryGb` — VRAM on a GPU,
 * unified memory on a Mac. Thresholds leave room for the OS and other apps. */
export function runsOn(memoryGb: number): string {
  if (memoryGb <= 7) return 'Most laptops and GPUs with 8 GB'
  if (memoryGb <= 11) return 'A 12 GB GPU or a 16 GB Mac'
  if (memoryGb <= 22) return 'A 24 GB GPU or a 32 GB Mac'
  return 'A workstation with 48 GB or more'
}

export function formatGb(gb: number): string {
  return `${Math.max(gb, 0.1).toFixed(1)} GB`
}

/** The model name `heulistic export` gives a job, so the web and CLI agree. */
export function exportName(sourceJobId: string): string {
  return `heulistic-${sourceJobId.slice(0, 8)}`
}

export function ollamaSteps(name: string): string[] {
  return [`ollama create ${name} -f Modelfile`, `ollama run ${name}`]
}
