'use client'

import { useState } from 'react'
import { AXOLOTL_SCHEMA, BUILDER_FIELDS, type SchemaField } from '@/lib/axolotl-schema'

const ADAPTER_OPTIONS = [
  { value: 'lora', label: 'LoRA' },
  { value: 'qlora', label: 'QLoRA (4-bit)' },
  { value: '', label: 'Full fine-tune' },
]
const LR_SCHEDULERS = ['cosine', 'linear', 'constant', 'cosine_with_restarts', 'polynomial', 'one_cycle']

interface BuilderConfig {
  base_model: string
  tokenizer_type: string
  adapter: string
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  lora_target_linear: boolean
  val_set_size: number
  num_epochs: number
  micro_batch_size: number
  gradient_accumulation_steps: number
  learning_rate: string
  sequence_len: number
  bf16: string
  gradient_checkpointing: boolean
  sample_packing: boolean
  lr_scheduler: string
}

const DEFAULTS: BuilderConfig = {
  base_model: '',
  tokenizer_type: '',
  adapter: 'lora',
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_linear: true,
  val_set_size: 0.1,
  num_epochs: 3,
  micro_batch_size: 2,
  gradient_accumulation_steps: 4,
  learning_rate: '2e-4',
  sequence_len: 2048,
  bf16: 'auto',
  gradient_checkpointing: true,
  sample_packing: false,
  lr_scheduler: 'cosine',
}

function buildYaml(c: BuilderConfig, extras: Record<string, unknown>): string {
  const lines: string[] = []

  lines.push(`base_model: ${c.base_model || 'YOUR_MODEL_HERE'}`)
  if (c.tokenizer_type) lines.push(`tokenizer_type: ${c.tokenizer_type}`)

  lines.push('')

  if (c.adapter) {
    lines.push(`adapter: ${c.adapter}`)
    if (c.adapter === 'qlora') lines.push('load_in_4bit: true')
    lines.push('')
    lines.push(`lora_r: ${c.lora_r}`)
    lines.push(`lora_alpha: ${c.lora_alpha}`)
    lines.push(`lora_dropout: ${c.lora_dropout}`)
    if (c.lora_target_linear) lines.push('lora_target_linear: true')
  }

  lines.push('')
  if (c.val_set_size > 0) lines.push(`val_set_size: ${c.val_set_size}`)

  lines.push('')
  lines.push(`sequence_len: ${c.sequence_len}`)
  lines.push(`sample_packing: ${c.sample_packing}`)

  lines.push('')
  lines.push(`num_epochs: ${c.num_epochs}`)
  lines.push(`micro_batch_size: ${c.micro_batch_size}`)
  lines.push(`gradient_accumulation_steps: ${c.gradient_accumulation_steps}`)
  lines.push(`learning_rate: ${c.learning_rate}`)
  lines.push(`lr_scheduler: ${c.lr_scheduler}`)

  lines.push('')
  lines.push(`bf16: ${c.bf16}`)
  lines.push(`gradient_checkpointing: ${c.gradient_checkpointing}`)

  lines.push('')
  lines.push('output_dir: ./model-out')

  const extraEntries = Object.entries(extras).filter(([, v]) => v !== '' && v !== null && v !== undefined)
  if (extraEntries.length > 0) {
    lines.push('')
    for (const [key, val] of extraEntries) {
      lines.push(`${key}: ${val}`)
    }
  }

  return lines.join('\n')
}

interface Props {
  onConfig: (yaml: string, filename: string) => void
}

function Field({
  label,
  hint,
  required,
  status,
  children,
}: {
  label: string
  hint?: string
  required?: boolean
  status?: 'error' | 'success' | 'neutral'
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-[0.85rem] font-semibold text-ink-2 mb-1">
        {label}
        {required && <span className="ml-1 text-danger">*</span>}
        {hint && <span className="ml-2 text-[0.78rem] font-normal text-ink-7">{hint}</span>}
        {status === 'error' && (
          <span className="ml-2 text-[0.75rem] font-normal text-danger">Required</span>
        )}
      </label>
      {children}
    </div>
  )
}

const inputCls =
  'w-full px-3 py-2 text-[0.88rem] border border-line rounded-[4px] bg-canvas focus:outline-none focus:border-ink focus:bg-paper transition-colors'
const inputErrorCls =
  'w-full px-3 py-2 text-[0.88rem] border border-danger/50 rounded-[4px] bg-danger-bg focus:outline-none focus:border-danger focus:bg-paper transition-colors'
const inputSuccessCls =
  'w-full px-3 py-2 text-[0.88rem] border border-ok/50 rounded-[4px] bg-ok-bg focus:outline-none focus:border-ok focus:bg-paper transition-colors'
const selectCls = inputCls

function requiredInputCls(value: string): string {
  if (!value.trim()) return inputErrorCls
  return inputSuccessCls
}

function lrInputCls(value: string): string {
  if (!value.trim()) return inputErrorCls
  const n = parseFloat(value)
  if (isNaN(n) || n <= 0 || n > 0.01) return inputErrorCls
  return inputSuccessCls
}

function lrStatus(value: string): 'error' | 'success' | 'neutral' {
  if (!value.trim()) return 'error'
  const n = parseFloat(value)
  if (isNaN(n) || n <= 0 || n > 0.01) return 'neutral'
  return 'success'
}

const CATEGORIES = ['Training', 'Checkpointing', 'Hardware', 'Model', 'Dataset', 'Advanced'] as const

export default function ConfigBuilder({ onConfig }: Props) {
  const [cfg, setCfg] = useState<BuilderConfig>(DEFAULTS)
  const [extras, setExtras] = useState<Record<string, unknown>>({})

  function set<K extends keyof BuilderConfig>(key: K, val: BuilderConfig[K]) {
    setCfg((prev) => ({ ...prev, [key]: val }))
  }

  const isLoRA = cfg.adapter === 'lora' || cfg.adapter === 'qlora'

  // Schema fields available to add (not already in builder or extras)
  const availableFields = AXOLOTL_SCHEMA.filter(
    (f) => !BUILDER_FIELDS.has(f.key) && !(f.key in extras)
  )

  function addField(key: string) {
    const field = AXOLOTL_SCHEMA.find((f) => f.key === key)
    if (!field) return
    const defaultVal = field.default !== undefined
      ? field.default
      : field.type === 'boolean'
      ? false
      : field.type === 'number'
      ? ''
      : field.type === 'enum'
      ? (field.options?.[0] ?? '')
      : ''
    setExtras((prev) => ({ ...prev, [key]: defaultVal }))
  }

  function removeField(key: string) {
    setExtras((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }

  function setExtra(key: string, val: unknown) {
    setExtras((prev) => ({ ...prev, [key]: val }))
  }

  function handleGenerate() {
    const yaml = buildYaml(cfg, extras)
    const name = cfg.base_model
      ? `${cfg.base_model.split('/').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '-')}-${cfg.adapter || 'full'}.yaml`
      : 'config.yaml'
    onConfig(yaml, name)
  }

  return (
    <div className="space-y-5">
      {/* Model */}
      <div>
        <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-ink-7 mb-3">Model</p>
        <div className="space-y-3">
          <Field label="Base model" required status={cfg.base_model.trim() ? 'success' : 'error'}>
            <input
              type="text"
              className={requiredInputCls(cfg.base_model)}
              placeholder="meta-llama/Llama-3-8B"
              value={cfg.base_model}
              onChange={(e) => set('base_model', e.target.value)}
            />
          </Field>
          <Field label="Tokenizer type" hint="optional — leave blank to auto-detect">
            <input
              type="text"
              className={inputCls}
              placeholder="LlamaTokenizer"
              value={cfg.tokenizer_type}
              onChange={(e) => set('tokenizer_type', e.target.value)}
            />
          </Field>
        </div>
      </div>

      {/* Adapter */}
      <div>
        <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-ink-7 mb-3">Fine-tuning type</p>
        <div className="space-y-3">
          <Field label="Adapter">
            <select className={selectCls} value={cfg.adapter} onChange={(e) => set('adapter', e.target.value)}>
              {ADAPTER_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </Field>

          {isLoRA && (
            <div className="grid grid-cols-3 gap-3">
              <Field label="Rank (r)">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.lora_r}
                  min={1}
                  onChange={(e) => set('lora_r', Number(e.target.value))}
                />
              </Field>
              <Field label="Alpha">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.lora_alpha}
                  min={1}
                  onChange={(e) => set('lora_alpha', Number(e.target.value))}
                />
              </Field>
              <Field label="Dropout">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.lora_dropout}
                  step={0.01}
                  min={0}
                  max={1}
                  onChange={(e) => set('lora_dropout', Number(e.target.value))}
                />
              </Field>
            </div>
          )}

          {isLoRA && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={cfg.lora_target_linear}
                onChange={(e) => set('lora_target_linear', e.target.checked)}
                className="w-4 h-4 accent-ink"
              />
              <span className="text-[0.88rem] text-ink-4">Target all linear layers <span className="text-ink-7">(lora_target_linear)</span></span>
            </label>
          )}
        </div>
      </div>

      {/* Training */}
      <div>
        <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-ink-7 mb-3">Training</p>
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <Field label="Epochs">
              <input
                type="number"
                className={inputCls}
                value={cfg.num_epochs}
                min={1}
                step={0.5}
                onChange={(e) => set('num_epochs', Number(e.target.value))}
              />
            </Field>
            <Field label="Batch size" hint="per GPU">
              <input
                type="number"
                className={inputCls}
                value={cfg.micro_batch_size}
                min={1}
                onChange={(e) => set('micro_batch_size', Number(e.target.value))}
              />
            </Field>
            <Field label="Grad accum">
              <input
                type="number"
                className={inputCls}
                value={cfg.gradient_accumulation_steps}
                min={1}
                onChange={(e) => set('gradient_accumulation_steps', Number(e.target.value))}
              />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Learning rate" required status={lrStatus(cfg.learning_rate)}>
              <input
                type="text"
                className={lrInputCls(cfg.learning_rate)}
                value={cfg.learning_rate}
                onChange={(e) => set('learning_rate', e.target.value)}
              />
            </Field>
            <Field label="Sequence length">
              <input
                type="number"
                className={inputCls}
                value={cfg.sequence_len}
                min={64}
                step={128}
                onChange={(e) => set('sequence_len', Number(e.target.value))}
              />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="LR scheduler">
              <select className={selectCls} value={cfg.lr_scheduler} onChange={(e) => set('lr_scheduler', e.target.value)}>
                {LR_SCHEDULERS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </Field>
            <Field label="Validation split" hint="0 = no eval">
              <input
                type="number"
                className={inputCls}
                value={cfg.val_set_size}
                step={0.05}
                min={0}
                max={0.5}
                onChange={(e) => set('val_set_size', Number(e.target.value))}
              />
            </Field>
          </div>
        </div>
      </div>

      {/* Hardware */}
      <div>
        <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-ink-7 mb-3">Hardware</p>
        <div className="space-y-2">
          <Field label="BF16">
            <select className={selectCls} value={cfg.bf16} onChange={(e) => set('bf16', e.target.value)}>
              <option value="auto">auto (recommended)</option>
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </Field>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={cfg.gradient_checkpointing}
              onChange={(e) => set('gradient_checkpointing', e.target.checked)}
              className="w-4 h-4 accent-ink"
            />
            <span className="text-[0.88rem] text-ink-4">Gradient checkpointing <span className="text-ink-7">(saves VRAM)</span></span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={cfg.sample_packing}
              onChange={(e) => set('sample_packing', e.target.checked)}
              className="w-4 h-4 accent-ink"
            />
            <span className="text-[0.88rem] text-ink-4">Sample packing <span className="text-ink-7">(faster training)</span></span>
          </label>
        </div>
      </div>

      {/* Additional fields */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-ink-7">Additional fields</p>
          {availableFields.length > 0 && (
            <select
              value=""
              onChange={(e) => { if (e.target.value) addField(e.target.value) }}
              className="text-[0.82rem] border border-line rounded-[4px] bg-paper px-2 py-1 text-ink-5 focus:outline-none focus:border-ink cursor-pointer"
            >
              <option value="">+ Add attribute</option>
              {CATEGORIES.map((cat) => {
                const fields = availableFields.filter((f) => f.category === cat)
                if (fields.length === 0) return null
                return (
                  <optgroup key={cat} label={cat}>
                    {fields.map((f) => (
                      <option key={f.key} value={f.key}>{f.label}</option>
                    ))}
                  </optgroup>
                )
              })}
            </select>
          )}
        </div>

        {Object.keys(extras).length === 0 ? (
          <p className="text-[0.8rem] text-ink-9 italic">No additional fields added.</p>
        ) : (
          <div className="space-y-3">
            {Object.keys(extras).map((key) => {
              const field = AXOLOTL_SCHEMA.find((f) => f.key === key) as SchemaField
              return (
                <div key={key}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[0.85rem] font-semibold text-ink-2">
                      {field.label}
                      <span className="ml-2 text-[0.75rem] font-mono font-normal text-ink-8">{field.key}</span>
                    </label>
                    <button
                      onClick={() => removeField(key)}
                      className="text-[0.75rem] text-ink-9 hover:text-danger transition-colors ml-2 shrink-0"
                      title="Remove field"
                    >
                      ✕
                    </button>
                  </div>
                  <p className="text-[0.75rem] text-ink-7 mb-1 leading-snug">{field.description}</p>
                  {field.type === 'boolean' ? (
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={!!extras[key]}
                        onChange={(e) => setExtra(key, e.target.checked)}
                        className="w-4 h-4 accent-ink"
                      />
                      <span className="text-[0.85rem] text-ink-4">{extras[key] ? 'true' : 'false'}</span>
                    </label>
                  ) : field.type === 'enum' ? (
                    <select
                      value={String(extras[key] ?? '')}
                      onChange={(e) => setExtra(key, e.target.value)}
                      className={selectCls}
                    >
                      {field.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : field.type === 'number' ? (
                    <input
                      type="number"
                      value={extras[key] === '' ? '' : Number(extras[key])}
                      onChange={(e) => setExtra(key, e.target.value === '' ? '' : Number(e.target.value))}
                      className={inputCls}
                    />
                  ) : (
                    <input
                      type="text"
                      value={String(extras[key] ?? '')}
                      onChange={(e) => setExtra(key, e.target.value)}
                      className={inputCls}
                    />
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      <button
        onClick={handleGenerate}
        className="w-full btn-primary justify-center py-3 text-[0.9rem]"
      >
        Generate Config →
      </button>
    </div>
  )
}
