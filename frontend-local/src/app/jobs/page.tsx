'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import Navbar from '@/components/navbar'
import Footer from '@/components/footer'
import StatusBadge from '@/components/status-badge'
import Spinner from '@/components/spinner'
import { listJobs } from '@/lib/api'
import type { Job } from '@/lib/types'

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const m = Math.floor(diff / 60_000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

function fmtUTCFull(iso: string): string {
  return new Date(iso).toLocaleString([], {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZone: 'UTC', timeZoneName: 'short',
  })
}

function jobCost(job: Job): string {
  if (job.costs.actual != null) return `$${Number(job.costs.actual).toFixed(2)}`
  if (job.costs.estimated != null) return `~$${Number(job.costs.estimated).toFixed(2)} est`
  return '—'
}

function JobRow({ job }: { job: Job }) {
  return (
    <Link
      href={`/jobs/${job.job_id}`}
      className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-6 items-center
                 bg-paper px-10 py-7 no-underline group
                 hover:bg-muted transition-colors duration-300"
    >
      <div>
        <p className="font-semibold text-ink text-sm group-hover:underline underline-offset-2">
          {job.config_name ?? 'Untitled'}
          {job.kind === 'export' && (
            <span className="ml-2 text-[0.7rem] font-medium text-ink-5 border border-line rounded px-1.5 py-0.5 no-underline">Export</span>
          )}
        </p>
        <p className="text-ink-7 text-xs mt-0.5">{job.job_id.slice(0, 8)}…</p>
      </div>

      <div className="text-right">
        <p className="text-[0.8rem] font-semibold text-ink-7 uppercase tracking-[0.05em]">Instance</p>
        <p className="text-sm text-ink mt-0.5">{job.instance_type ?? '—'}</p>
      </div>

      <div className="text-right">
        <p className="text-[0.8rem] font-semibold text-ink-7 uppercase tracking-[0.05em]">Cost</p>
        <p className="text-sm text-ink mt-0.5">
          {job.costs.actual != null
            ? `$${Number(job.costs.actual).toFixed(2)}`
            : job.costs.estimated != null
            ? `~$${Number(job.costs.estimated).toFixed(2)}`
            : '—'}
        </p>
      </div>

      <div className="text-right">
        <p className="text-[0.8rem] font-semibold text-ink-7 uppercase tracking-[0.05em]">Started</p>
        <p className="text-sm text-ink mt-0.5" title={fmtUTCFull(job.created_at)}>{timeAgo(job.created_at)}</p>
      </div>

      <StatusBadge status={job.status} />
    </Link>
  )
}

function JobCard({ job }: { job: Job }) {
  return (
    <Link
      href={`/jobs/${job.job_id}`}
      className="bg-paper rounded-xl border border-ink/5 p-5 no-underline group hover:bg-muted transition-colors duration-300"
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <p className="font-semibold text-ink text-sm group-hover:underline underline-offset-2 leading-snug">
          {job.config_name ?? 'Untitled'}
          {job.kind === 'export' && (
            <span className="ml-2 text-[0.7rem] font-medium text-ink-5 border border-line rounded px-1.5 py-0.5 no-underline">Export</span>
          )}
        </p>
        <StatusBadge status={job.status} />
      </div>
      <p className="text-ink-7 text-xs mb-3">
        {[job.instance_type, jobCost(job)].filter(Boolean).join(' · ')}
      </p>
      <div className="flex items-center justify-between">
        <p className="text-ink-9 text-xs" title={fmtUTCFull(job.created_at)}>
          Started {timeAgo(job.created_at)}
        </p>
        <span className="text-[0.8rem] font-semibold text-ink">View job →</span>
      </div>
    </Link>
  )
}

export default function JobsPage() {
  const { data: jobs = [], isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: 5_000,
  })

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <section className="mt-[80px] section-pad flex-1">
        <div className="container">
          <div className="mb-[60px]">
            <p className="label-caps mb-4">Session</p>
            <h1 className="heading-xl">My Training Jobs</h1>
          </div>

          {isLoading ? (
            <div className="flex justify-center py-16"><Spinner size="lg" /></div>
          ) : jobs.length === 0 ? (
            /* Empty state — mirrors heulistic.com stat-highlight card */
            <div
              className="theme-light text-center py-[60px] px-10 rounded-2xl max-w-[480px] mx-auto"
              style={{
                background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                border: '2px solid #e9ecef',
              }}
            >
              <p className="text-[3rem] font-[900] text-ink tracking-[-0.03em] leading-none mb-4">
                0
              </p>
              <p className="text-[1.1rem] font-semibold text-ink-2">jobs submitted yet</p>
              <p className="text-ink-5 text-sm mt-2 mb-8">
                Upload an Axolotl config to kick off your first fine-tune.
              </p>
              <Link href="/train" className="btn-primary">
                Start Training →
              </Link>
            </div>
          ) : (
            <>
              {/* Desktop table */}
              <div className="hidden md:block card-grid">
                {jobs.map((job) => (
                  <JobRow key={job.job_id} job={job} />
                ))}
              </div>
              {/* Mobile cards */}
              <div className="md:hidden flex flex-col gap-3">
                {jobs.map((job) => (
                  <JobCard key={job.job_id} job={job} />
                ))}
              </div>
            </>
          )}
        </div>
      </section>

      <Footer />
    </div>
  )
}
