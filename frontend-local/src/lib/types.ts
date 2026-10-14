export type JobStatus = 'pending' | 'provisioning' | 'running' | 'checkpointing' | 'completed' | 'failed' | 'cancelled'

export interface JobCosts {
  estimated: number | null
  actual: number | null
}

export interface Job {
  job_id: string
  status: JobStatus
  config_name: string | null
  instance_type: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  costs: JobCosts
  queue_position: number | null
  /** "export" turns a finished training job's model into a GGUF for Ollama. */
  kind: 'train' | 'export'
}

export interface JobDetail extends Job {
  config_yaml: string
  dataset_size_mb: number | null
  error_message: string | null
  estimated_duration_minutes: number | null
  hourly_rate: number | null
  provisioned_at: string | null
  checkpointed_at: string | null
  last_checkpoint_step: number | null
  last_checkpoint_s3_path: string | null
  resume_from_checkpoint_path: string | null
  source_job_id: string | null
  export_quant: string | null
  modelfile: string | null
  evaluation: EvaluationReport | { skipped: string } | null
  exports: ExportSummary[]
}

/** Base model vs fine-tune, scored on rows held out from training. */
export interface EvaluationReport {
  examples: number
  base: { loss: number; perplexity: number }
  tuned: { loss: number; perplexity: number }
  samples: { prompt: string; reference: string; base: string; tuned: string }[]
}

export interface ExportSummary {
  job_id: string
  quant: string | null
  status: JobStatus
}

export interface ExportOption {
  quant: string
  file_gb: number
  memory_gb: number
  estimated_cost: number
  estimated_minutes: number
}

export interface ExportOptions {
  options: ExportOption[]
  recommended: string
}

export interface InstanceOption {
  instance_type: string
  gpus: number
  vram_gb: number
  our_rate: number
}

export interface Metric {
  step: number
  epoch: number | null
  train_loss: number | null
  learning_rate: number | null
  gpu_utilization: number | null
  gpu_memory_used_mb: number | null
  tokens_per_second: number | null
  timestamp: string
}

export interface JobSubmitResponse {
  job_id: string
  status: string
  estimated_cost: number | null
  estimated_duration_minutes: number | null
  instance_type: string | null
  created_at: string
  message: string
}

export interface EstimateResponse {
  estimated_cost: number
  estimated_duration_minutes: number
  instance_type: string
  hourly_rate: number
  dataset_known: boolean
}

export interface DownloadResponse {
  url: string
  expires_in_seconds: number
}

export interface CreditPackage {
  id: string
  price_cents: number
  credits: number
  label: string
  bonus_pct: number
  price_display: string
  credits_display: string
}

export interface CreditsResponse {
  balance_cents: number
  balance_display: string
  packages: CreditPackage[]
  payment_past_due: boolean
}

export interface CreditTransaction {
  id: number
  amount_cents: number
  description: string | null
  job_id: string | null
  created_at: string
}

export interface ConfirmResponse {
  balance_cents: number
  balance_display: string
  credits_added: number
}

export type ValidationLevel = 'error' | 'warning' | 'info'

/** One finding from POST /api/validate. `code` is a stable rule id. */
export interface ValidationResult {
  level: ValidationLevel
  code: string
  message: string
  field: string | null
  line: number | null
}

export interface ValidateResponse {
  ok: boolean
  counts: Record<ValidationLevel, number>
  diagnostics: ValidationResult[]
}
