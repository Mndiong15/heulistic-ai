'use client'

import { useCallback, useRef, useState } from 'react'

import { matchesAccept } from '@/lib/files'

interface Props {
  accept: string
  file: File | null
  onFile: (file: File) => void
  hint?: string
  subhint?: string
}

export default function FileDropzone({ accept, file, onFile, hint, subhint }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState<string | null>(null)

  // Dropped files bypass the input's `accept` filter, so both paths check here.
  const take = useCallback(
    (f: File) => {
      if (!matchesAccept(f.name, accept)) {
        setRejected(`${f.name} is not a ${accept.split(',').map((e) => e.trim()).join(' or ')} file`)
        return
      }
      setRejected(null)
      onFile(f)
    },
    [accept, onFile],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const dropped = e.dataTransfer.files[0]
      if (dropped) take(dropped)
    },
    [take],
  )

  const sizeLabel = file
    ? file.size > 1024 * 1024
      ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
      : `${(file.size / 1024).toFixed(1)} KB`
    : null

  return (
    <div
      className={`border-2 border-dashed rounded-[6px] p-8 text-center cursor-pointer transition-all duration-300 ${
        dragging
          ? 'border-ink bg-muted'
          : file
          ? 'border-ink bg-muted'
          : 'border-line hover:border-ink-7'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) take(f); e.target.value = '' }}
      />

      {file ? (
        <div>
          <p className="font-semibold text-ink text-sm">{file.name}</p>
          <p className="text-ink-7 text-xs mt-1">{sizeLabel} · Click to replace</p>
        </div>
      ) : (
        <div>
          <p className="text-ink-5 text-sm font-medium">{hint ?? 'Drag & drop or click to browse'}</p>
          {subhint && <p className="text-ink-7 text-xs mt-1">{subhint}</p>}
        </div>
      )}
      {rejected && <p className="text-danger text-xs mt-2">{rejected}</p>}
    </div>
  )
}
