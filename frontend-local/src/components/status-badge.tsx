import type { JobStatus } from '@/lib/types'

const styles: Record<JobStatus, string> = {
  pending:       'border border-line-soft text-ink-5',
  provisioning:  'border border-line-soft text-ink-5',
  running:       'bg-inverse text-white',
  checkpointing: 'bg-inverse text-white',
  completed:     'bg-ok-2 text-white',
  failed:        'bg-danger text-white',
  cancelled:     'bg-muted text-ink-7',
}

const dots: Record<JobStatus, string> = {
  pending:       'bg-[#999]',
  provisioning:  'bg-[#999] animate-pulse',
  running:       'bg-paper animate-pulse',
  checkpointing: 'bg-paper animate-pulse',
  completed:     'bg-paper',
  failed:        'bg-paper',
  cancelled:     'bg-[#ccc]',
}

const labels: Record<JobStatus, string> = {
  pending:       'Pending',
  provisioning:  'Provisioning',
  running:       'Running',
  checkpointing: 'Checkpointing',
  completed:     'Completed',
  failed:        'Failed',
  cancelled:     'Cancelled',
}

export default function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[0.8rem] font-semibold ${styles[status]}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dots[status]}`} />
      {labels[status]}
    </span>
  )
}
