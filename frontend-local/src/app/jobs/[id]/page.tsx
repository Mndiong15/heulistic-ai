'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import Navbar from '@/components/navbar'
import Spinner from '@/components/spinner'
import Footer from '@/components/footer'
import StatusBadge from '@/components/status-badge'
import EvaluationReport from '@/components/evaluation-report'
import ExportPanel from '@/components/export-panel'
import ExportResult from '@/components/export-result'
import { getJob, getMetrics, cancelJob, getDownloadUrl, resumeJob, apiBase } from '@/lib/api'
import { getIdToken } from '@/lib/auth'
import type { JobDetail, Metric } from '@/lib/types'
import { useThemeColors, type ThemeColors } from '@/lib/use-theme-colors'

type Tab = 'logs' | 'metrics' | 'config'

const TERMINAL = ['completed', 'failed', 'cancelled']
const ACTIVE    = ['pending', 'provisioning', 'running', 'checkpointing']

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZone: 'UTC', timeZoneName: 'short',
  })
}

function elapsed(from: string, to?: string | null): string {
  const ms = (to ? new Date(to).getTime() : Date.now()) - new Date(from).getTime()
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

function ProgressBar({ job }: { job: JobDetail }) {
  if (!job.started_at || job.status === 'pending' || job.status === 'provisioning') return null
  const done = !!job.completed_at
  const failed = job.status === 'failed'

  return (
    <div className="mt-4">
      <div className="flex justify-between text-xs text-ink-5 mb-1.5">
        <span>{done ? (failed ? 'Failed after' : 'Completed in') : 'Running for'}</span>
        <span>{elapsed(job.started_at, job.completed_at)}</span>
      </div>
      <div className="h-1 bg-inverse-hover rounded-full overflow-hidden">
        {done ? (
          <div className={`h-full w-full rounded-full ${failed ? 'bg-danger' : 'bg-paper'}`} />
        ) : (
          <div className="h-full w-full rounded-full bg-paper opacity-40 animate-pulse" />
        )}
      </div>
    </div>
  )
}

function LiveCost({ job }: { job: JobDetail }) {
  const [cost, setCost] = useState<number | null>(null)
  const rate = job.hourly_rate ?? null

  useEffect(() => {
    if (!job.started_at || !rate || TERMINAL.includes(job.status)) {
      setCost(null)
      return
    }
    const tick = () => {
      const hours = (Date.now() - new Date(job.started_at!).getTime()) / 3_600_000
      setCost(hours * rate)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [job.started_at, job.status, rate])

  if (cost == null) return null
  return (
    <span className="text-ink-7 text-sm tabular-nums">
      ${cost.toFixed(3)} running
    </span>
  )
}

function Timeline({ job }: { job: JobDetail }) {
  const steps: { label: string; time: string | null | undefined; note?: string }[] = [
    { label: 'Submitted', time: job.created_at },
    { label: 'Provisioning', time: job.provisioned_at },
    { label: 'Training started', time: job.started_at },
    job.last_checkpoint_step != null
      ? { label: 'Last checkpoint', time: job.checkpointed_at, note: `step ${job.last_checkpoint_step}` }
      : null,
    job.status === 'completed'
      ? { label: 'Completed', time: job.completed_at,
          note: job.started_at ? elapsed(job.started_at, job.completed_at) : undefined }
      : job.status === 'failed'
      ? { label: 'Failed', time: job.completed_at }
      : job.status === 'cancelled'
      ? { label: 'Cancelled', time: job.completed_at }
      : null,
  ].filter(Boolean) as { label: string; time: string | null | undefined; note?: string }[]

  const visible = steps.filter((s) => s.time)
  if (visible.length === 0) return null

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1 mt-5">
      {visible.map((s, i) => (
        <div key={i} className="flex items-center gap-2 text-xs text-ink-5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#444] shrink-0" />
          <span className="font-medium text-ink-7">{s.label}</span>
          <span className="text-ink-5 font-mono">{fmt(s.time)}</span>
          {s.note && <span className="text-ink-4">({s.note})</span>}
          {i < visible.length - 1 && <span className="text-ink-2 ml-2">→</span>}
        </div>
      ))}
    </div>
  )
}

function MetricChart({
  title,
  data,
  dataKey,
  color,
  formatter,
}: {
  title: string
  data: Metric[]
  dataKey: keyof Metric
  color: keyof ThemeColors
  formatter?: (v: number) => string
}) {
  const colors = useThemeColors()
  const stroke = colors[color]
  const chartData = data
    .filter((m) => m[dataKey] != null)
    .map((m) => ({ step: m.step, value: m[dataKey] as number }))

  return (
    <div className="bg-paper border border-line-soft rounded-xl p-6">
      <p className="text-sm font-semibold text-ink mb-4">{title}</p>
      {chartData.length === 0 ? (
        <div className="h-[160px] flex items-center justify-center text-ink-7 text-sm">
          No data yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.line} />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 11, fill: colors.ink7 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: colors.ink7 }}
              tickLine={false}
              axisLine={false}
              width={40}
              tickFormatter={formatter}
            />
            <Tooltip
              contentStyle={{
                background: colors.paper,
                border: `1px solid ${colors.line}`,
                color: colors.ink,
                borderRadius: '6px',
                fontSize: '12px',
                boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
              }}
              formatter={(v: number) => [formatter ? formatter(v) : v, title]}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={stroke}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: stroke }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<Tab>('logs')
  const [logs, setLogs] = useState<string[]>([])
  const logsEndRef = useRef<HTMLDivElement>(null)
  const logContainerRef = useRef<HTMLDivElement>(null)
  const isNearBottomRef = useRef(true)

  const { data: job, isLoading } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id),
    refetchInterval: (q) =>
      TERMINAL.includes(q.state.data?.status ?? '') ? false : 5_000,
  })

  const { data: metrics = [] } = useQuery({
    queryKey: ['metrics', id],
    queryFn: () => getMetrics(id),
    refetchInterval: job?.status === 'running' || job?.status === 'checkpointing' ? 5_000 : false,
  })

  const startStream = useCallback(() => {
    const controller = new AbortController()
    ;(async () => {
      try {
        const token = await getIdToken()
        const res = await fetch(`${apiBase()}/api/jobs/${id}/logs`, {
          headers: { 'Authorization': `Bearer ${token}` },
          signal: controller.signal,
        })
        if (!res.body) return
        const reader = res.body.getReader()
        const dec = new TextDecoder()
        let buf = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buf += dec.decode(value, { stream: true })
          const parts = buf.split('\n')
          buf = parts.pop() ?? ''
          for (const part of parts) {
            if (!part.startsWith('data: ')) continue
            try {
              const d = JSON.parse(part.slice(6)) as { line?: string; event?: string }
              if (d.event === 'done') { reader.cancel(); return }
              if (d.line) setLogs((prev) => [...prev, d.line!])
            } catch { /* skip malformed */ }
          }
        }
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') return
      }
    })()
    return () => controller.abort()
  }, [id])

  useEffect(() => { return startStream() }, [startStream])
  useEffect(() => {
    if (isNearBottomRef.current) {
      const el = logContainerRef.current
      if (el) el.scrollTop = el.scrollHeight
    }
  }, [logs])

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['job', id] }),
  })

  const downloadMutation = useMutation({
    mutationFn: () => getDownloadUrl(id),
    onSuccess: (data) => window.open(data.url, '_blank'),
  })

  const resumeMutation = useMutation({
    mutationFn: () => resumeJob(id),
    onSuccess: (data) => router.push(`/jobs/${data.job_id}`),
  })

  if (isLoading || !job) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <div className="flex-1 flex items-center justify-center"><Spinner size="lg" /></div>
      </div>
    )
  }

  const isActive = ACTIVE.includes(job.status)
  const isExport = job.kind === 'export'
  const showAfterTraining = job.status === 'completed' && (isExport ? !!job.modelfile : true)
  const canResume = job.status === 'failed' && !!job.last_checkpoint_s3_path
  const tabs: { key: Tab; label: string }[] = [
    { key: 'logs', label: 'Logs' },
    { key: 'metrics', label: 'Metrics' },
    { key: 'config', label: 'Configuration' },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <div className="mt-[80px] flex-1">
        {/* Header */}
        <div className="bg-inverse text-white py-[60px]">
          <div className="container">
            <div className="flex items-start justify-between gap-6 flex-wrap">
              <div className="flex-1 min-w-0">
                <Link
                  href={isExport && job.source_job_id ? `/jobs/${job.source_job_id}` : '/jobs'}
                  className="text-ink-5 text-sm no-underline hover:text-white transition-colors"
                >
                  {isExport ? '← Trained model' : '← My Jobs'}
                </Link>
                <h1 className="text-[1.8rem] font-bold tracking-[-0.02em] mt-3 mb-1 text-white [overflow-wrap:anywhere]">
                  {isExport ? `Export · ${job.export_quant}` : (job.config_name ?? 'Training Job')}
                </h1>
                {isExport && job.config_name && (
                  <p className="text-ink-7 text-sm mb-1">{job.config_name}</p>
                )}
                <p className="text-ink-5 text-sm font-mono">{job.job_id}</p>

                <div className="flex items-center gap-5 mt-5 flex-wrap">
                  <StatusBadge status={job.status} />

                  {/* Queue position while waiting */}
                  {job.status === 'pending' && job.queue_position != null && job.queue_position > 0 && (
                    <span className="text-ink-7 text-sm">
                      Queue position {job.queue_position}
                    </span>
                  )}

                  {job.instance_type && (
                    <span className="text-ink-7 text-sm">{job.instance_type}</span>
                  )}

                  {/* Live cost ticker while running */}
                  <LiveCost job={job} />

                  {/* Final cost when done */}
                  {TERMINAL.includes(job.status) && (
                    <span className="text-ink-7 text-sm">
                      {job.costs.actual != null
                        ? `$${Number(job.costs.actual).toFixed(2)} actual`
                        : job.costs.estimated != null
                        ? `~$${Number(job.costs.estimated).toFixed(2)} estimated`
                        : null}
                    </span>
                  )}

                  {/* Last checkpoint indicator */}
                  {job.last_checkpoint_step != null && (
                    <span className="text-ink-5 text-xs font-mono">
                      ckpt step {job.last_checkpoint_step}
                    </span>
                  )}
                </div>

                <ProgressBar job={job} />
                <Timeline job={job} />

                {job.status === 'provisioning' && (
                  <div className="mt-5 flex items-start gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3">
                    <Spinner size="sm" />
                    <div>
                      <p className="text-white text-sm font-medium">GPU instance is booting</p>
                      <p className="text-ink-6 text-xs mt-0.5">
                        The training server is starting up — this takes 5–15 minutes.
                        {job.provisioned_at && ` Started ${new Date(job.provisioned_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short' })}.`}
                      </p>
                    </div>
                  </div>
                )}

                {job.status === 'pending' && (
                  <div className="mt-5 flex items-start gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3">
                    <Spinner size="sm" />
                    <div>
                      <p className="text-white text-sm font-medium">Waiting for a slot</p>
                      <p className="text-ink-6 text-xs mt-0.5">
                        {job.queue_position != null && job.queue_position > 0
                          ? `Position ${job.queue_position} in queue. The scheduler picks up jobs every 10 seconds.`
                          : 'Your job will be picked up within seconds.'}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 flex-shrink-0 pt-6">
                {job.status === 'completed' && (
                  <button
                    onClick={() => downloadMutation.mutate()}
                    disabled={downloadMutation.isPending}
                    className="text-sm px-5 py-3 rounded-[6px] font-semibold border border-ok-2 text-ok-2 hover:bg-ok-2 hover:text-white transition-all duration-300"
                  >
                    {downloadMutation.isPending ? 'Generating…' : isExport ? 'Download GGUF' : 'Download Model'}
                  </button>
                )}
                {canResume && (
                  <button
                    onClick={() => {
                      if (confirm(`Resume from checkpoint at step ${job.last_checkpoint_step}? A new job will be created.`))
                        resumeMutation.mutate()
                    }}
                    disabled={resumeMutation.isPending}
                    className="text-sm px-5 py-3 rounded-[6px] font-semibold border border-white/20 text-white hover:bg-white/10 transition-all duration-300"
                  >
                    {resumeMutation.isPending ? 'Queuing…' : `Resume from step ${job.last_checkpoint_step}`}
                  </button>
                )}
                {isActive && (
                  <button
                    onClick={() => { if (confirm('Cancel this job?')) cancelMutation.mutate() }}
                    disabled={cancelMutation.isPending}
                    className="text-sm px-5 py-3 rounded-[6px] font-semibold border border-danger text-danger hover:bg-danger hover:text-white transition-all duration-300"
                  >
                    {cancelMutation.isPending ? 'Cancelling…' : 'Cancel Job'}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Provisioning / queue state info bar */}
        {(job.status === 'provisioning' || (job.status === 'pending' && job.queue_position != null && job.queue_position > 0)) && (
          <div className="bg-muted border-b border-line-soft py-3">
            <div className="container">
              <p className="text-sm text-ink-5">
                {job.status === 'provisioning'
                  ? 'Launching GPU instance — this takes about 2–3 minutes…'
                  : `Your job is #${job.queue_position} in the queue. It will start when a slot opens.`}
              </p>
            </div>
          </div>
        )}

        {/* Error banner for failed jobs */}
        {job.status === 'failed' && job.error_message && (
          <div className="bg-danger-bg border-b border-danger/20 py-4">
            <div className="container">
              <p className="text-sm font-semibold text-danger mb-1">
                Job failed{job.started_at ? ` after ${elapsed(job.started_at, job.completed_at)}` : ''}
              </p>
              <pre className="text-xs text-ink-5 font-mono whitespace-pre-wrap break-words line-clamp-3">
                {job.error_message}
              </pre>
              {canResume && (
                <p className="text-xs text-ink-7 mt-2">
                  A checkpoint was saved at step {job.last_checkpoint_step} — use &quot;Resume&quot; above to continue training from there.
                </p>
              )}
            </div>
          </div>
        )}

        {/* After training: the before/after report and export — or, for an export, how to run it */}
        {showAfterTraining && (
          <div className="container py-8 space-y-6">
            {isExport ? (
              <ExportResult job={job} onDownload={() => downloadMutation.mutate()} />
            ) : (
              <>
                <EvaluationReport report={job.evaluation} />
                <ExportPanel job={job} />
              </>
            )}
          </div>
        )}

        {/* Tab bar */}
        <div className="bg-paper border-b border-line-soft sticky top-[80px] z-40">
          <div className="container">
            <div className="flex gap-0 overflow-x-auto">
              {tabs.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={`px-6 py-4 text-sm font-semibold border-b-2 transition-colors duration-200 ${
                    tab === key
                      ? 'border-ink text-ink'
                      : 'border-transparent text-ink-5 hover:text-ink'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Tab content */}
        <div className="section-pad bg-canvas">
          <div className="container">

            {/* Logs */}
            {tab === 'logs' && (
              <div
                className="rounded-xl overflow-hidden"
                style={{ background: '#1a1a1a', border: '1px solid #333' }}
              >
                <div className="flex items-center justify-between px-6 py-3 border-b border-[#333]">
                  <p className="text-ink-7 text-xs font-mono">Training output</p>
                  {(job.status === 'running' || job.status === 'checkpointing') && (
                    <span className="flex items-center gap-1.5 text-ink-7 text-xs">
                      <span className="w-1.5 h-1.5 rounded-full bg-danger animate-pulse" />
                      Live
                    </span>
                  )}
                </div>
                <div
                  ref={logContainerRef}
                  onScroll={() => {
                    const el = logContainerRef.current
                    if (!el) return
                    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
                  }}
                  className="p-6 h-[300px] sm:h-[480px] overflow-y-auto font-mono text-sm leading-relaxed"
                >
                  {logs.length === 0 ? (
                    <p className="text-ink-5">
                      {job.status === 'pending'
                        ? 'Waiting for a queue slot…'
                        : job.status === 'provisioning'
                        ? 'Launching instance…'
                        : isActive
                        ? 'Waiting for training to start…'
                        : 'No logs captured.'}
                    </p>
                  ) : (
                    logs.map((line, i) => (
                      <p key={i} className="text-[#ccc] whitespace-pre-wrap break-all">
                        {line}
                      </p>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            )}

            {/* Metrics */}
            {tab === 'metrics' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <MetricChart
                  title="Training Loss"
                  data={metrics}
                  dataKey="train_loss"
                  color="danger"
                  formatter={(v) => v.toFixed(4)}
                />
                <MetricChart
                  title="Learning Rate"
                  data={metrics}
                  dataKey="learning_rate"
                  color="ink"
                  formatter={(v) => v.toExponential(2)}
                />
                <MetricChart
                  title="GPU Utilization (%)"
                  data={metrics}
                  dataKey="gpu_utilization"
                  color="ink5"
                  formatter={(v) => `${v.toFixed(1)}%`}
                />
                <MetricChart
                  title="Throughput (tokens/s)"
                  data={metrics}
                  dataKey="tokens_per_second"
                  color="ink7"
                />
              </div>
            )}

            {/* Config */}
            {tab === 'config' && (
              <div className="max-w-[720px]">
                <div className="card-grid">
                  <div className="card-item">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-sm font-semibold text-ink">Axolotl Config</h3>
                      <div className="flex gap-4 text-[0.8rem] text-ink-7">
                        {job.dataset_size_mb && <span>Dataset: {job.dataset_size_mb} MB</span>}
                        {job.resume_from_checkpoint_path && (
                          <span className="text-ink-5">Resumed from checkpoint</span>
                        )}
                      </div>
                    </div>
                    <pre className="text-[0.85rem] text-ink-2 leading-relaxed overflow-x-auto whitespace-pre-wrap break-words bg-muted rounded-[6px] p-4 font-mono">
                      {job.config_yaml}
                    </pre>
                  </div>
                </div>
              </div>
            )}

          </div>
        </div>
      </div>

      <Footer />
    </div>
  )
}
