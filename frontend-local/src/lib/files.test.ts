import { describe, expect, it } from 'vitest'

import { matchesAccept } from './files'

// The `accept` attribute only filters the file picker. A dragged-and-dropped
// file skipped it entirely, so a CSV could be dropped on a JSONL-only box and
// was only refused at submit — after a large file had already been uploaded.

describe('matchesAccept', () => {
  it('accepts a listed extension', () => {
    expect(matchesAccept('train.jsonl', '.jsonl')).toBe(true)
    expect(matchesAccept('config.yml', '.yaml,.yml')).toBe(true)
  })

  it('refuses anything else', () => {
    expect(matchesAccept('train.csv', '.jsonl')).toBe(false)
    expect(matchesAccept('train.parquet', '.jsonl')).toBe(false)
    expect(matchesAccept('train.json', '.jsonl')).toBe(false)
    expect(matchesAccept('jsonl', '.jsonl')).toBe(false)
  })

  it('ignores case and spacing in the list', () => {
    expect(matchesAccept('TRAIN.JSONL', ' .jsonl ')).toBe(true)
  })

  it('accepts anything when nothing is listed', () => {
    expect(matchesAccept('whatever.bin', '')).toBe(true)
  })
})
