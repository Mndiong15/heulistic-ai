'use client'

import Link from 'next/link'
import { useAuth } from './AuthProvider'

export default function CreditBadge() {
  const { creditDisplay, user } = useAuth()

  if (!user) return null

  return (
    <Link
      href="/credits"
      className="flex items-center gap-2 no-underline"
    >
      <span className="text-[0.85rem] font-medium text-ink-5 hover:text-ink transition-colors">
        {creditDisplay}
      </span>
      <span className="text-[0.8rem] font-semibold bg-inverse text-white px-2.5 py-1 rounded-full hover:bg-inverse-hover transition-colors">
        + Add credits
      </span>
    </Link>
  )
}
