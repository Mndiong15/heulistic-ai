import { describe, expect, it } from 'vitest'

import { exportName, formatGb, ollamaSteps, runsOn, summarizeEvaluation } from './post-training'

// ── summarizeEvaluation ────────────────────────────────────────────────────────
//
// The report compares the base model with the fine-tune on rows held out from
// training. The headline has to be honest in both directions: a fine-tune that
// made the held-out loss worse is reported as such, not dressed up.

const report = (base: number, tuned: number, examples = 12) => ({
  examples,
  base: { loss: base, perplexity: Math.exp(base) },
  tuned: { loss: tuned, perplexity: Math.exp(tuned) },
  samples: [],
})

describe('summarizeEvaluation', () => {
  it('reports how much the loss fell', () => {
    const s = summarizeEvaluation(report(2.0, 0.5))!
    expect(s.better).toBe(true)
    expect(s.changePct).toBe(-75)
    expect(s.headline).toBe('Loss down 75% on 12 held-out examples')
  })

  it('says so plainly when the fine-tune did worse', () => {
    const s = summarizeEvaluation(report(1.0, 1.5))!
    expect(s.better).toBe(false)
    expect(s.headline).toBe('Loss up 50% on 12 held-out examples')
  })

  it('calls a negligible change no change', () => {
    expect(summarizeEvaluation(report(1.0, 0.995))!.headline).toBe(
      'No meaningful change on 12 held-out examples',
    )
  })

  it('uses the singular for one example', () => {
    expect(summarizeEvaluation(report(2, 1, 1))!.headline).toContain('1 held-out example')
  })

  it('has nothing to say about a skipped or missing report', () => {
    expect(summarizeEvaluation(null)).toBeNull()
    expect(summarizeEvaluation({ skipped: 'No held-out examples' })).toBeNull()
  })
})

// ── Running the export ─────────────────────────────────────────────────────────

describe('runsOn', () => {
  it('speaks in machines a hobbyist owns', () => {
    expect(runsOn(5.4)).toBe('Most laptops and GPUs with 8 GB')
    expect(runsOn(9.1)).toBe('A 12 GB GPU or a 16 GB Mac')
    expect(runsOn(20)).toBe('A 24 GB GPU or a 32 GB Mac')
    expect(runsOn(45)).toBe('A workstation with 48 GB or more')
  })
})

describe('formatGb', () => {
  it('rounds to a tenth', () => {
    expect(formatGb(4.28)).toBe('4.3 GB')
    expect(formatGb(0.098)).toBe('0.1 GB')
  })
})

describe('ollamaSteps', () => {
  it('creates the model from the Modelfile, then runs it', () => {
    expect(ollamaSteps('heulistic-abcd1234')).toEqual([
      'ollama create heulistic-abcd1234 -f Modelfile',
      'ollama run heulistic-abcd1234',
    ])
  })
})

describe('exportName', () => {
  it('matches what `heulistic export` calls it', () => {
    expect(exportName('abcd1234-5678-90ab')).toBe('heulistic-abcd1234')
  })
})
