import { getIdToken } from './auth'
import type {
  ConfirmResponse,
  CreditTransaction,
  CreditsResponse,
  DownloadResponse,
  EstimateResponse,
  ExportOptions,
  InstanceOption,
  Job,
  JobDetail,
  JobSubmitResponse,
  Metric,
  ValidateResponse,
} from './types'

// Empty — all /api/* calls go through the Next.js rewrite proxy in next.config.js,
// which targets API_URL at runtime (so no value is baked into the Docker image).
const BASE = ''

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = await getIdToken()
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

function normaliseCosts<T extends { costs: { estimated: unknown; actual: unknown } }>(j: T): T {
  j.costs.estimated = j.costs.estimated != null ? parseFloat(j.costs.estimated as string) : null
  j.costs.actual = j.costs.actual != null ? parseFloat(j.costs.actual as string) : null
  return j
}

export async function listJobs(): Promise<Job[]> {
  const jobs = await req<Job[]>('/api/jobs')
  return jobs.map(normaliseCosts)
}

export async function getJob(id: string): Promise<JobDetail> {
  const job = normaliseCosts(await req<JobDetail>(`/api/jobs/${id}`))
  if (job.hourly_rate != null) job.hourly_rate = parseFloat(job.hourly_rate as unknown as string)
  return job
}

export function fetchInstances(): Promise<InstanceOption[]> {
  return req<InstanceOption[]>('/api/instances')
}

export function getMetrics(id: string): Promise<Metric[]> {
  return req(`/api/jobs/${id}/metrics`)
}

export function cancelJob(id: string): Promise<void> {
  return req(`/api/jobs/${id}`, { method: 'DELETE' })
}

export function getDownloadUrl(id: string): Promise<DownloadResponse> {
  return req(`/api/jobs/${id}/download`)
}

export async function estimateJob(
  configYaml: string,
  datasetSizeMb = 0,
  instanceType?: string,
): Promise<EstimateResponse> {
  const raw = await req<Record<string, unknown>>('/api/estimate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_yaml: configYaml,
      dataset_size_mb: datasetSizeMb,
      ...(instanceType ? { instance_type: instanceType } : {}),
    }),
  })
  return {
    estimated_cost: parseFloat(raw.estimated_cost as string),
    estimated_duration_minutes: Number(raw.estimated_duration_minutes),
    instance_type: raw.instance_type as string,
    hourly_rate: parseFloat(raw.hourly_rate as string),
    dataset_known: raw.dataset_known !== false,
  }
}

/**
 * Validate an Axolotl config server-side.
 *
 * The rules live in the `heulistic-axolotl` Python package, which also backs
 * job submission and the `heulistic validate-config` CLI — so the browser, the
 * API and the CLI cannot disagree about what a valid config is.
 *
 * Pass `instanceType` to additionally check the config fits that instance's
 * VRAM.
 */
export function validateConfig(
  configYaml: string,
  instanceType?: string,
): Promise<ValidateResponse> {
  return req('/api/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_yaml: configYaml,
      ...(instanceType ? { instance_type: instanceType } : {}),
    }),
  })
}

export function submitJob(
  config: File,
  hfToken?: string,
  instanceType?: string,
  dataset?: File,
): Promise<JobSubmitResponse> {
  const form = new FormData()
  form.append('config', config)
  if (hfToken) form.append('hf_token', hfToken)
  if (instanceType) form.append('instance_type', instanceType)
  if (dataset) form.append('dataset', dataset)
  return req('/api/jobs', { method: 'POST', body: form })
}

export async function getExportOptions(id: string): Promise<ExportOptions> {
  const raw = await req<ExportOptions>(`/api/jobs/${id}/export-options`)
  raw.options.forEach((o) => { o.estimated_cost = Number(o.estimated_cost) })
  return raw
}

export function startExport(id: string, quant: string, hfToken?: string): Promise<JobSubmitResponse> {
  return req(`/api/jobs/${id}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ quant, ...(hfToken ? { hf_token: hfToken } : {}) }),
  })
}

export function resumeJob(id: string): Promise<JobSubmitResponse> {
  return req(`/api/jobs/${id}/resume`, { method: 'POST' })
}

export function getCredits(): Promise<CreditsResponse> {
  return req('/api/credits')
}

export async function createCheckoutSession(
  packageId: string | null,
  customAmountCents?: number,
): Promise<{ checkout_url: string }> {
  return req('/api/credits/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(
      customAmountCents
        ? { custom_amount_cents: customAmountCents }
        : { package_id: packageId },
    ),
  })
}

export function confirmCheckout(sessionId: string): Promise<ConfirmResponse> {
  return req(`/api/credits/checkout/confirm?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'POST',
  })
}

export function listTransactions(limit = 50): Promise<CreditTransaction[]> {
  return req(`/api/credits/transactions?limit=${limit}`)
}

/** Returns the API base URL for use in SSE/streaming (needs custom headers via fetch). */
export function apiBase(): string {
  return BASE
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export interface AdminJob {
  job_id: string
  user_id: string
  user_email: string
  status: string
  config_name: string | null
  instance_type: string | null
  instance_id: string | null
  estimated_cost: number | null
  actual_cost: number | null
  created_at: string
  provisioned_at: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
}

export interface AdminUser {
  id: string
  email: string
  credit_balance_cents: number
  job_count: number
}

export interface SchedulerStatus {
  alive: boolean
  ticked_at: string | null
  pending_count: number
  active_count: number
  dispatched_count: number
}

export function getAdminStatus(): Promise<{ is_admin: boolean }> {
  return req('/api/admin/status')
}

export function getSchedulerStatus(): Promise<SchedulerStatus> {
  return req('/api/admin/scheduler')
}

export function fetchAdminJobs(params?: { status?: string; user_id?: string }): Promise<AdminJob[]> {
  const qs = new URLSearchParams()
  if (params?.status) qs.set('status', params.status)
  if (params?.user_id) qs.set('user_id', params.user_id)
  const query = qs.toString() ? `?${qs.toString()}` : ''
  return req(`/api/admin/jobs${query}`)
}

export function fetchAdminUsers(): Promise<AdminUser[]> {
  return req('/api/admin/users')
}

export function adminAdjustCredits(
  userId: string,
  amountCents: number,
  description: string,
): Promise<{ new_balance_cents: number }> {
  return req(`/api/admin/users/${userId}/credits`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount_cents: amountCents, description }),
  })
}

export interface AdminMetrics {
  total_jobs: number
  jobs_by_status: Record<string, number>
  total_users: number
  total_revenue_cents: number
  outstanding_credits_cents: number
  jobs_per_day: { date: string; count: number }[]
  revenue_per_day: { date: string; revenue_cents: number }[]
  new_users_per_day: { date: string; count: number }[]
}

export function getAdminMetrics(): Promise<AdminMetrics> {
  return req('/api/admin/metrics')
}
