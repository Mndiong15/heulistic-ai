'use client'

const LOCAL_MODE = process.env.NEXT_PUBLIC_LOCAL_MODE === 'true'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import * as jsYaml from 'js-yaml'
import Navbar from '@/components/navbar'
import Footer from '@/components/footer'
import FileDropzone from '@/components/file-dropzone'
import ConfigBuilder from '@/components/config-builder'
import {
  estimateJob,
  fetchInstances,
  submitJob,
  validateConfig as validateConfigApi,
} from '@/lib/api'
import type { ValidationResult } from '@/lib/types'
import type { EstimateResponse } from '@/lib/types'
import { useAuth } from '@/components/AuthProvider'
import { usePostHog } from 'posthog-js/react'

const DATASET_FORMATS = ['alpaca', 'sharegpt', 'chat_template', 'completion', 'oasst', 'gpteacher', 'input_output']

function fmtDuration(minutes: number): string {
  if (minutes < 60) return `~${minutes}m`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m === 0 ? `~${h}h` : `~${h}h ${m}m`
}

function instanceFamily(instanceType: string): string {
  if (instanceType.startsWith('g5'))   return 'A10G'
  if (instanceType.startsWith('g4dn')) return 'T4'
  if (instanceType.startsWith('g6'))   return 'L4'
  if (instanceType.startsWith('p3dn')) return 'V100-32'
  if (instanceType.startsWith('p3'))   return 'V100'
  if (instanceType.startsWith('p4de')) return 'A100-80'
  if (instanceType.startsWith('p4d'))  return 'A100'
  if (instanceType.startsWith('p5'))   return 'H100'
  return instanceType.split('.')[0].toUpperCase()
}

const FAMILY_LABELS: Record<string, string> = {
  'T4':     'T4 · Budget (g4dn)',
  'A10G':   'A10G · Standard (g5)',
  'L4':     'L4 · Standard (g6)',
  'V100':   'V100 16 GB (p3)',
  'V100-32':'V100 32 GB (p3dn)',
  'A100':   'A100 40 GB (p4d)',
  'A100-80':'A100 80 GB (p4de)',
  'H100':   'H100 · Flagship (p5)',
}

const EXAMPLES: Record<string, string> = {
  'Llama-3.2 1B LoRA': `base_model: meta-llama/Llama-3.2-1B
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

val_set_size: 0.1
sequence_len: 2048
sample_packing: false

num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler: cosine

bf16: auto
gradient_checkpointing: true
output_dir: ./model-out`,

  'Mistral 7B QLoRA': `base_model: mistralai/Mistral-7B-v0.1
adapter: qlora
load_in_4bit: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

val_set_size: 0.1
sequence_len: 4096
sample_packing: true

num_epochs: 3
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
lr_scheduler: cosine

bf16: auto
gradient_checkpointing: true
output_dir: ./model-out`,

  'Llama-2 7B Full': `base_model: meta-llama/Llama-2-7b-hf
tokenizer_type: LlamaTokenizer

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

val_set_size: 0.1
sequence_len: 2048
sample_packing: false

num_epochs: 3
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5e-5
lr_scheduler: cosine

bf16: auto
gradient_checkpointing: true
output_dir: ./model-out`,
}

type ConfigMode = 'upload' | 'build'

export default function TrainPage() {
  const router = useRouter()
  const { user, isLoading: authLoading, creditBalance } = useAuth()
  const posthog = usePostHog()
  const [configMode, setConfigMode] = useState<ConfigMode>('upload')
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [configYaml, setConfigYaml] = useState<string | null>(null)
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [datasetFormat, setDatasetFormat] = useState('alpaca')
  const [hfToken, setHfToken] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null)
  const [estimating, setEstimating] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<ValidationResult[]>([])
  const [overrideInstance, setOverrideInstance] = useState<string | null>(null)
  const [showInstancePicker, setShowInstancePicker] = useState(false)
  const [hwValidation, setHwValidation] = useState<ValidationResult[]>([])
  const [copied, setCopied] = useState(false)
  const [showPreview, setShowPreview] = useState(false)

  const { data: instanceOptions = [] } = useQuery({
    queryKey: ['instances'],
    queryFn: fetchInstances,
    staleTime: 5 * 60 * 1000,
  })

  // Validation runs server-side so the browser, the API and the CLI share one
  // rule set. Debounced because this fires as the user types.
  useEffect(() => {
    if (!configYaml) {
      setValidation([])
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      validateConfigApi(configYaml)
        .then((res) => {
          if (!cancelled) setValidation(res.diagnostics)
        })
        .catch(() => {
          // A validation outage must not block the page; submission is
          // validated server-side regardless.
          if (!cancelled) setValidation([])
        })
    }, 400)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [configYaml])

  const configHasDatasets = useMemo(() => {
    if (!configYaml) return false
    try {
      const parsed = jsYaml.load(configYaml) as Record<string, unknown>
      return Array.isArray(parsed?.datasets) && (parsed.datasets as unknown[]).length > 0
    } catch {
      return false
    }
  }, [configYaml])

  const previewYaml = useMemo(() => {
    if (!configYaml) return configYaml
    if (!datasetPath && !datasetFile) return configYaml
    try {
      const parsed = jsYaml.load(configYaml) as Record<string, unknown>
      parsed.datasets = [{ path: datasetPath || datasetFile!.name, type: datasetFormat }]
      return jsYaml.dump(parsed)
    } catch {
      return configYaml
    }
  }, [configYaml, datasetPath, datasetFile, datasetFormat])

  // When a dataset path or file is provided in build mode, suppress the "no datasets" error —
  // the path will be injected into the YAML at submission time.
  const effectiveValidation = (datasetPath || datasetFile)
    ? validation.filter((r) => r.field !== 'datasets')
    : validation

  const hasErrors = effectiveValidation.some((r) => r.level === 'error') || hwValidation.some((r) => r.level === 'error')
  const hasDataset = configMode === 'upload'
    ? (configHasDatasets || !!datasetPath || !!datasetFile)
    : !!datasetPath || !!datasetFile

  const estimatedCostCents = estimate ? Math.round(estimate.estimated_cost * 100) : 0
  const heldCostCents = Math.ceil(estimatedCostCents * 1.5)
  const insufficientCredits = !LOCAL_MODE && creditBalance !== null && estimate !== null && creditBalance < heldCostCents

  const canSubmit = !!configFile && !hasErrors && hasDataset && !insufficientCredits

  const submitHint = !configFile
    ? 'Generate or upload a config to continue'
    : hasErrors
    ? 'Fix config errors to continue'
    : !hasDataset
    ? 'Add a dataset to continue'
    : insufficientCredits
    ? `Insufficient credits — $${(heldCostCents / 100).toFixed(2)} required (est. $${estimate!.estimated_cost.toFixed(2)} + 50% buffer), you have $${(creditBalance! / 100).toFixed(2)}`
    : null

  const runEstimate = useCallback(async (yaml: string, instanceType?: string) => {
    setEstimating(true)
    setEstimate(null)
    setHwValidation([])
    setError(null)
    try {
      const sizeMb = datasetFile ? datasetFile.size / (1024 * 1024) : 0
      const result = await estimateJob(yaml, sizeMb, instanceType)
      setEstimate(result)
      posthog?.capture('estimate_viewed', {
        instance_type: result.instance_type,
        estimated_cost: result.estimated_cost,
        estimated_duration_minutes: result.estimated_duration_minutes,
        dataset_known: result.dataset_known,
      })
      // Re-validate against the chosen instance; keep only the hardware-fit
      // findings, since the config-level ones are already shown above.
      try {
        const hw = await validateConfigApi(yaml, result.instance_type)
        setHwValidation(
          hw.diagnostics.filter(
            (d) => d.code.startsWith('vram.') || d.code.startsWith('hardware.'),
          ),
        )
      } catch {
        setHwValidation([])
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid config')
    } finally {
      setEstimating(false)
    }
  }, [datasetFile, posthog])

  const applyYaml = useCallback(
    async (yaml: string, file: File) => {
      setConfigFile(file)
      setConfigYaml(yaml)
      setOverrideInstance(null)
      setShowInstancePicker(false)
      setHwValidation([])
      await runEstimate(yaml)
    },
    [runEstimate],
  )

  // After returning from login, restore the config the user had built before sign-in.
  const applyYamlRef = useRef(applyYaml)
  applyYamlRef.current = applyYaml
  useEffect(() => {
    if (LOCAL_MODE) return
    const saved = sessionStorage.getItem('heulistic_pending_config')
    if (!saved) return
    sessionStorage.removeItem('heulistic_pending_config')
    try {
      const { yaml, filename, datasetPath: path, datasetFormat: fmt } = JSON.parse(saved) as {
        yaml: string; filename: string; datasetPath: string; datasetFormat: string
      }
      const file = new File([yaml], filename, { type: 'application/x-yaml' })
      void applyYamlRef.current(yaml, file)
      if (path) {
        setDatasetPath(path)
        setDatasetFormat(fmt)
      }
    } catch {
      // ignore malformed data
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleConfigFile = useCallback(
    async (file: File) => {
      const yaml = await file.text()
      await applyYaml(yaml, file)
    },
    [applyYaml],
  )

  const handleDatasetPath = useCallback(
    async (path: string) => {
      setDatasetPath(path)
      if (configYaml) await runEstimate(configYaml)
    },
    [configYaml, runEstimate],
  )

  const handleExample = useCallback(
    async (name: string) => {
      const yaml = EXAMPLES[name]
      const file = new File([yaml], `${name.toLowerCase().replace(/ /g, '-')}.yaml`, {
        type: 'application/x-yaml',
      })
      await applyYaml(yaml, file)
    },
    [applyYaml],
  )

  const handleBuilderConfig = useCallback(
    async (yaml: string, filename: string) => {
      const file = new File([yaml], filename, { type: 'application/x-yaml' })
      await applyYaml(yaml, file)
    },
    [applyYaml],
  )

  const handleSubmit = async () => {
    if (!configFile || !configYaml || !canSubmit) return

    // Gate on auth — save the user's work and send them to sign in.
    if (!LOCAL_MODE && !user) {
      sessionStorage.setItem('heulistic_pending_config', JSON.stringify({
        yaml: configYaml,
        filename: configFile.name,
        datasetPath,
        datasetFormat,
      }))
      router.push('/login?redirect=/train')
      return
    }

    setSubmitting(true)
    setError(null)
    try {
      let finalConfigFile = configFile
      if (datasetPath) {
        const parsed = jsYaml.load(configYaml) as Record<string, unknown>
        parsed.datasets = [{ path: datasetPath, type: datasetFormat }]
        const injected = jsYaml.dump(parsed)
        finalConfigFile = new File([injected], configFile.name, { type: configFile.type })
      } else if (datasetFile) {
        // Strip any datasets block from the config so the training container
        // uses the uploaded file exclusively (avoids conflicting dataset sources).
        const parsed = jsYaml.load(configYaml) as Record<string, unknown>
        delete parsed.datasets
        const stripped = jsYaml.dump(parsed)
        finalConfigFile = new File([stripped], configFile.name, { type: configFile.type })
      }
      const result = await submitJob(finalConfigFile, hfToken || undefined, overrideInstance || undefined, datasetFile || undefined)
      posthog?.capture('job_submitted', {
        job_id: result.job_id,
        instance_type: result.instance_type,
      })
      router.push(`/jobs/${result.job_id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Submission failed')
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <section className="mt-[80px] py-[60px] flex-1 bg-paper">
        <div className="container">
          <p className="label-caps mb-3">Submit Job</p>
          <h1 className="heading-lg mb-6">Start a Training Run</h1>

          <div className="flex items-start gap-2 bg-warn-bg border border-warn-line/50 rounded-lg px-4 py-3 mb-8 max-w-[720px]">
            <span className="text-warn-strong text-sm mt-px shrink-0">ⓘ</span>
            <p className="text-[0.82rem] text-warn-ink leading-snug">
              <strong className="font-semibold">Single-node training only.</strong>{' '}
              Jobs run on a single EC2 instance with up to 8 GPUs.
              Multi-GPU strategies such as DeepSpeed, FSDP, tensor parallelism, and sequence parallelism are all supported.
              Only multi-node Ray configs (<code className="font-mono text-[0.78rem]">use_ray: true</code>, <code className="font-mono text-[0.78rem]">ray_num_workers &gt; 1</code>) will be rejected.
            </p>
          </div>

          <div className={`flex gap-8 items-start ${configYaml ? 'flex-col lg:flex-row' : ''}`}>

            {/* ── Left: form ── */}
            <div className={configYaml ? 'w-full lg:w-[480px] shrink-0' : 'max-w-[720px] w-full'}>
              <div className="card-grid">

                {/* 01 Dataset */}
                <div className="card-item">
                  <div className="flex justify-between items-start mb-5">
                    <div>
                      <h3 className="text-[1.05rem] font-semibold text-ink">
                        Dataset{' '}
                        {!datasetPath && !datasetFile && !(configMode === 'upload' && configHasDatasets) && (
                          <span className="text-danger font-normal text-sm">Required</span>
                        )}
                        {configMode === 'upload' && configHasDatasets && !datasetPath && !datasetFile && (
                          <span className="text-ok-ink font-normal text-sm ml-1">✓ In config</span>
                        )}
                      </h3>
                      <p className="text-ink-5 text-sm mt-1">
                        {configMode === 'upload' && configHasDatasets
                          ? "Override your config's dataset, or leave as-is"
                          : 'Upload a file or provide a dataset path'}
                      </p>
                    </div>
                    <span className="text-[0.85rem] font-semibold text-ink-7">01</span>
                  </div>

                  {/* Override warning — shown when the config already specifies a dataset */}
                  {configHasDatasets && (datasetFile || datasetPath) && (
                    <div className="flex items-start gap-2 bg-warn-bg border border-warn-line/50 rounded-lg px-3 py-2.5 mb-4">
                      <span className="text-warn-strong text-sm mt-px shrink-0">⚠</span>
                      <p className="text-[0.8rem] text-warn-ink leading-snug">
                        This will override the dataset specified in your config file.
                      </p>
                    </div>
                  )}

                  {/* File upload — dimmed when URL is typed */}
                  <div className={datasetPath ? 'pointer-events-none opacity-40' : ''}>
                    <FileDropzone
                      accept=".jsonl"
                      file={datasetFile}
                      onFile={(f) => { setDatasetFile(f); setDatasetPath('') }}
                      hint="Drag & drop your dataset, or click to browse"
                      subhint="JSONL — one JSON example per line · max 50 GB"
                    />
                  </div>
                  {datasetFile && (
                    <button
                      onClick={() => setDatasetFile(null)}
                      className="mt-1.5 text-[0.75rem] text-ink-7 hover:text-danger transition-colors"
                    >
                      ✕ Remove file
                    </button>
                  )}

                  {/* Divider */}
                  <div className="flex items-center gap-3 my-4">
                    <div className="flex-1 h-px bg-line-soft" />
                    <span className="text-[0.75rem] font-medium text-ink-7 uppercase tracking-wider">or</span>
                    <div className="flex-1 h-px bg-line-soft" />
                  </div>

                  {/* URL input — disabled when file is uploaded */}
                  <input
                    type="text"
                    placeholder="tatsu-lab/alpaca  ·  s3://bucket/data.jsonl  ·  https://…"
                    value={datasetPath}
                    disabled={!!datasetFile}
                    onChange={(e) => { void handleDatasetPath(e.target.value) }}
                    className={`w-full px-3 py-2 text-[0.88rem] rounded-[4px] border transition-colors ${
                      datasetFile
                        ? 'border-line bg-tint text-ink-7 cursor-not-allowed'
                        : datasetPath
                        ? 'border-ok/50 bg-ok-bg focus:outline-none focus:border-ok'
                        : 'border-line bg-canvas focus:outline-none focus:border-ink focus:bg-paper'
                    }`}
                  />
                  <p className="text-[0.75rem] text-ink-7 mt-1">HuggingFace ID, S3 URI, or public URL</p>

                  {/* Format dropdown — shown when either input is active */}
                  {(datasetPath || datasetFile) && (
                    <div className="mt-3">
                      <label className="block text-[0.83rem] font-semibold text-ink-2 mb-1">
                        Dataset format
                      </label>
                      <select
                        value={datasetFormat}
                        onChange={(e) => setDatasetFormat(e.target.value)}
                        className="w-full px-3 py-2 text-[0.88rem] border border-line rounded-[4px] bg-canvas focus:outline-none focus:border-ink focus:bg-paper transition-colors"
                      >
                        {DATASET_FORMATS.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                {/* Config */}
                <div className="card-item">
                  <div className="flex justify-between items-start mb-5">
                    <div>
                      <h3 className="text-[1.05rem] font-semibold text-ink">Configuration</h3>
                      <p className="text-ink-5 text-sm mt-1">Axolotl-format YAML</p>
                    </div>
                    <span className="text-[0.85rem] font-semibold text-ink-7">02</span>
                  </div>

                  <div className="flex gap-1 mb-5 p-1 bg-tint rounded-[6px] w-fit">
                    {(['upload', 'build'] as ConfigMode[]).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setConfigMode(mode)}
                        className={`px-4 py-1.5 rounded-[4px] text-[0.85rem] font-medium transition-all duration-200 ${
                          configMode === mode
                            ? 'bg-paper text-ink shadow-sm'
                            : 'text-ink-5 hover:text-ink'
                        }`}
                      >
                        {mode === 'upload' ? 'Upload' : 'Build Config'}
                      </button>
                    ))}
                  </div>

                  {configMode === 'upload' ? (
                    <FileDropzone
                      accept=".yaml,.yml"
                      file={configFile}
                      onFile={handleConfigFile}
                      hint="Drag & drop your YAML config, or click to browse"
                    />
                  ) : (
                    <ConfigBuilder onConfig={handleBuilderConfig} />
                  )}

                  <div className="mt-4">
                    <p className="text-[0.8rem] text-ink-7 mb-2">Quick start examples:</p>
                    <div className="flex gap-2 flex-wrap">
                      {Object.keys(EXAMPLES).map((name) => (
                        <button
                          key={name}
                          onClick={() => handleExample(name)}
                          className="text-[0.85rem] font-medium text-ink-5 border border-line-soft px-3 py-1.5 rounded-[4px] hover:border-ink hover:text-ink transition-colors"
                        >
                          {name}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 03 Advanced */}
                <div className="card-item">
                  <button
                    onClick={() => setShowAdvanced((v) => !v)}
                    className="flex items-center justify-between w-full text-left"
                  >
                    <div className="flex items-center gap-2">
                      <h3 className="text-[1.05rem] font-semibold text-ink">Advanced</h3>
                      {hfToken && (
                        <span className="text-[0.72rem] font-semibold bg-ok-bg text-ok-ink border border-ok/30 px-2 py-0.5 rounded-full">
                          HF token set
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-[0.85rem] font-semibold text-ink-7">03</span>
                      <span className="text-ink-7 text-sm">{showAdvanced ? '▲' : '▼'}</span>
                    </div>
                  </button>

                  {showAdvanced && (
                    <div className="mt-5 space-y-2">
                      <label className="block text-[0.85rem] font-semibold text-ink-2">
                        HuggingFace token
                        <span className="ml-2 text-[0.78rem] font-normal text-ink-7">
                          required for gated models (Llama-2, Llama-3)
                        </span>
                      </label>
                      <input
                        type="password"
                        placeholder="hf_••••••••••••••••••••••••••••••••••••••••••"
                        value={hfToken}
                        onChange={(e) => setHfToken(e.target.value.trim())}
                        className="w-full px-3 py-2 text-[0.88rem] border border-line rounded-[4px] bg-canvas focus:outline-none focus:border-ink focus:bg-paper transition-colors font-mono"
                        autoComplete="off"
                      />
                      <p className="text-[0.75rem] text-ink-7">
                        Token is stored securely and deleted after your job completes.{' '}
                        <a
                          href="https://huggingface.co/settings/tokens"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline hover:text-ink"
                        >
                          Get a token →
                        </a>
                      </p>
                    </div>
                  )}
                </div>

                {/* 04 Hardware / Cost */}
                {LOCAL_MODE && (
                  <div className="card-item">
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="text-[1.05rem] font-semibold text-ink">Hardware</h3>
                      <span className="text-[0.85rem] font-semibold text-ink-7">04</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-info-bg flex items-center justify-center text-[1rem]">⚡</div>
                      <div>
                        <p className="text-[0.95rem] font-bold text-ink">Local GPU · your machine</p>
                        <p className="text-[0.78rem] text-ink-7 mt-0.5">All available GPUs · no cost</p>
                      </div>
                    </div>
                  </div>
                )}
                {!LOCAL_MODE && (estimate || estimating) && (
                  <div className="card-item">
                    <div className="flex justify-between items-start mb-5">
                      <h3 className="text-[1.05rem] font-semibold text-ink">Cost Estimate</h3>
                      <span className="text-[0.85rem] font-semibold text-ink-7">04</span>
                    </div>

                    {estimating ? (
                      <p className="text-ink-7 text-sm">Calculating…</p>
                    ) : estimate ? (
                      <>
                      {/* Disclaimer when dataset size is unknown */}
                      {!estimate.dataset_known && (
                        <div className="flex items-start gap-2 bg-warn-bg-2 border border-warn/40 rounded-lg px-3 py-2.5 mb-4">
                          <span className="text-amber text-sm mt-px">⚠</span>
                          <p className="text-[0.8rem] text-warn-ink leading-snug">
                            Dataset size unknown — this estimate may be much lower than actual cost. You&apos;ll see a live cost ticker once training starts.
                          </p>
                        </div>
                      )}

                      {/* Cost + Duration row */}
                      <div className="grid grid-cols-2 gap-6 mb-3">
                        <div>
                          <p className="label-caps text-[0.75rem] mb-2">Estimated Cost</p>
                          <p className="text-[2rem] font-[900] text-ink tracking-[-0.03em] leading-none">
                            ${estimate.estimated_cost.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="label-caps text-[0.75rem] mb-2">Duration</p>
                          <p className="text-[2rem] font-[900] text-ink tracking-[-0.03em] leading-none">
                            {fmtDuration(estimate.estimated_duration_minutes)}
                          </p>
                        </div>
                      </div>
                      <p className="text-[0.75rem] text-ink-7 mb-5">
                        ${(Math.ceil(estimate.estimated_cost * 150) / 100).toFixed(2)} held at submission (50% buffer) — unused credits refunded at completion.
                      </p>

                      {/* Instance row */}
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <p className="label-caps text-[0.75rem]">Instance</p>
                          <button
                            onClick={() => setShowInstancePicker((v) => !v)}
                            className="text-[0.72rem] font-medium text-ink-5 hover:text-ink underline underline-offset-2 transition-colors"
                          >
                            {showInstancePicker ? 'Done' : 'Change'}
                          </button>
                        </div>
                        {showInstancePicker ? (
                          <select
                            value={overrideInstance ?? estimate.instance_type}
                            onChange={(e) => {
                              const val = e.target.value
                              setOverrideInstance(val)
                              if (configYaml) void runEstimate(configYaml, val)
                            }}
                            className="w-full px-2 py-1.5 text-[0.82rem] border border-line rounded-[4px] bg-paper focus:outline-none focus:border-ink transition-colors"
                          >
                            {Array.from(new Set(instanceOptions.map((o) => instanceFamily(o.instance_type)))).map((family) => (
                              <optgroup key={family} label={FAMILY_LABELS[family] ?? family}>
                                {instanceOptions.filter((o) => instanceFamily(o.instance_type) === family).map((opt) => (
                                  <option key={opt.instance_type} value={opt.instance_type}>
                                    {opt.instance_type} — {opt.gpus} GPU{opt.gpus > 1 ? 's' : ''} · {opt.vram_gb} GB · ${opt.our_rate}/hr
                                  </option>
                                ))}
                              </optgroup>
                            ))}
                          </select>
                        ) : (
                          <div className="flex items-baseline gap-3">
                            <p className="text-[1rem] font-bold text-ink">{estimate.instance_type}</p>
                            <p className="text-ink-7 text-sm">${estimate.hourly_rate}/hr</p>
                            {overrideInstance && (
                              <button
                                onClick={() => {
                                  setOverrideInstance(null)
                                  if (configYaml) void runEstimate(configYaml)
                                }}
                                className="text-[0.72rem] text-ink-7 hover:text-danger transition-colors"
                              >
                                Reset to auto
                              </button>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Hardware validation messages */}
                      {hwValidation.length > 0 && (
                        <div className="mt-4 space-y-1.5">
                          {hwValidation.map((r, i) => (
                            <div
                              key={i}
                              className={`flex gap-2 items-start text-[0.82rem] rounded-[4px] px-3 py-2 ${
                                r.level === 'error'
                                  ? 'bg-danger-bg text-danger'
                                  : r.level === 'warning'
                                  ? 'bg-warn-bg text-warn-ink'
                                  : 'bg-info-bg text-info-ink'
                              }`}
                            >
                              <span className="mt-px shrink-0">
                                {r.level === 'error' ? '✕' : r.level === 'warning' ? '⚠' : 'ℹ'}
                              </span>
                              <span>{r.message}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      </>
                    ) : null}
                  </div>
                )}
              </div>

              {/* Config validation messages */}
              {effectiveValidation.filter((r) => r.level !== 'info').length > 0 && (
                <div className="mt-4 space-y-1.5">
                  {effectiveValidation
                    .filter((r) => r.level !== 'info')
                    .map((r, i) => (
                      <div
                        key={i}
                        className={`flex gap-2 items-start text-[0.82rem] rounded-[4px] px-3 py-2 ${
                          r.level === 'error'
                            ? 'bg-danger-bg text-danger'
                            : 'bg-warn-bg text-warn-ink'
                        }`}
                      >
                        <span className="mt-px shrink-0">{r.level === 'error' ? '✕' : '⚠'}</span>
                        <span>
                          {r.field && <strong className="font-semibold">{r.field}: </strong>}
                          {r.message}
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {error && (
                <p className="mt-4 text-danger text-sm font-medium">{error}</p>
              )}

              <div className="mt-8 flex flex-col items-end gap-2">
                <button
                  onClick={handleSubmit}
                  disabled={!canSubmit || submitting}
                  className="btn-primary"
                >
                  {submitting ? 'Submitting…' : (!LOCAL_MODE && !user && !authLoading) ? 'Sign in to Start →' : 'Start Training →'}
                </button>
                {submitHint && !submitting && (
                  <p className="text-[0.8rem] text-ink-7">
                    {submitHint}
                    {!LOCAL_MODE && insufficientCredits && (
                      <> — <a href="/credits" className="text-ink font-medium hover:underline">Top up credits</a></>
                    )}
                  </p>
                )}
              </div>
            </div>

            {/* ── Mobile preview toggle ── */}
            {configYaml && (
              <button
                className="lg:hidden mt-2 text-[0.85rem] font-semibold text-ink border border-line-soft rounded-[6px] px-4 py-2.5 bg-paper hover:bg-muted transition-colors self-start"
                onClick={() => setShowPreview((v) => !v)}
              >
                {showPreview ? 'Hide preview ▴' : 'Preview config ▾'}
              </button>
            )}

            {/* ── Right: YAML preview ── */}
            {configYaml && (
              <div className={`flex-1 min-w-0 ${showPreview ? 'block' : 'hidden'} lg:block`}>
                <div className="sticky top-[100px]">
                  <div
                    className="rounded-xl overflow-hidden border border-line-soft"
                    style={{ background: '#1a1a1a' }}
                  >
                    <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                        <span className="w-3 h-3 rounded-full bg-[#febc2e]" />
                        <span className="w-3 h-3 rounded-full bg-[#28c840]" />
                      </div>
                      <span className="text-[0.75rem] font-medium text-white/40 font-mono">
                        {configFile?.name ?? 'config.yaml'}
                      </span>
                      <button
                        onClick={() => {
                          void navigator.clipboard.writeText(previewYaml ?? '').then(() => {
                            setCopied(true)
                            setTimeout(() => setCopied(false), 2000)
                          })
                        }}
                        className="text-[0.72rem] font-medium text-white/40 hover:text-white/80 transition-colors w-[52px] text-right"
                      >
                        {copied ? '✓ Copied' : 'Copy'}
                      </button>
                    </div>

                    <div className="overflow-y-auto max-h-[calc(100vh-280px)]">
                      <pre className="p-5 m-0 text-[0.82rem] leading-[1.65] font-mono text-white/80 whitespace-pre overflow-x-auto">
                        {previewYaml}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}
