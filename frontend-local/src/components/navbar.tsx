'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuth } from './AuthProvider'
import CreditBadge from './CreditBadge'
import ThemeToggle from './theme-toggle'

function UserMenu() {
  const { user, signOut } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  if (!user) return null

  const initials = user.email.slice(0, 2).toUpperCase()

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center justify-center w-8 h-8 rounded-full bg-inverse text-white text-[0.75rem] font-bold cursor-pointer border-none focus:outline-none hover:bg-inverse-hover transition-colors"
        aria-label="Account menu"
      >
        {initials}
      </button>

      {open && (
        <div className="absolute right-0 top-[calc(100%+8px)] w-[220px] bg-paper rounded-xl shadow-[0_4px_24px_rgba(0,0,0,0.12)] border border-ink/5 py-1 z-50">
          <div className="px-4 py-3 border-b border-tint">
            <p className="text-[0.7rem] font-semibold text-ink-7 uppercase tracking-wider mb-0.5">Signed in as</p>
            <p className="text-[0.875rem] font-medium text-ink truncate">{user.email}</p>
          </div>
          <button
            onClick={() => { setOpen(false); signOut() }}
            className="w-full text-left px-4 py-2.5 text-[0.875rem] text-ink-5 hover:text-ink hover:bg-muted transition-colors bg-transparent border-none cursor-pointer"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

const LOCAL_MODE = process.env.NEXT_PUBLIC_LOCAL_MODE === 'true'

export default function Navbar({ variant = 'app' }: { variant?: 'landing' | 'app' }) {
  const path = usePathname()
  const { user, isAdmin } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    setMenuOpen(false)
  }, [path])

  return (
    <nav className="fixed top-0 w-full bg-paper/95 backdrop-blur-xl z-50 py-5 border-b border-ink/5">
      <div className="container flex justify-between items-center">
        <Link
          href="/"
          className="text-[1.5rem] font-bold text-ink tracking-[-0.02em] no-underline"
        >
          Heulistic
        </Link>

        {/* Desktop nav */}
        <ul className="hidden md:flex list-none gap-8 m-0 p-0 items-center">
          {variant === 'landing' ? (
            <>
              <li><a href="/#how-it-works" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300">How it works</a></li>
              <li><a href="/#pricing" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300">Pricing</a></li>
              <li><a href="/#consulting" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300">Consulting</a></li>
              {user && <li><Link href="/platform" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300">Platform</Link></li>}
              <li><Link href="/blog" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300">Blog</Link></li>
            </>
          ) : (
            <>
              {user && (
                <li>
                  <Link
                    href="/jobs"
                    className={`no-underline font-medium text-[0.95rem] transition-colors duration-300 ${
                      path.startsWith('/jobs') ? 'text-ink' : 'text-ink-5 hover:text-ink'
                    }`}
                  >
                    My Jobs
                  </Link>
                </li>
              )}
              {user && !LOCAL_MODE && (
                <li>
                  <CreditBadge />
                </li>
              )}
              <li>
                <Link
                  href="/docs"
                  className={`no-underline font-medium text-[0.95rem] transition-colors duration-300 ${
                    path.startsWith('/docs') ? 'text-ink' : 'text-ink-5 hover:text-ink'
                  }`}
                >
                  Docs
                </Link>
              </li>
              {isAdmin && (
                <li>
                  <Link
                    href="/admin"
                    className={`no-underline font-medium text-[0.95rem] transition-colors duration-300 ${
                      path.startsWith('/admin') ? 'text-ink' : 'text-ink-5 hover:text-ink'
                    }`}
                  >
                    Admin
                  </Link>
                </li>
              )}
            </>
          )}
          {LOCAL_MODE ? (
            <>
              <li>
                <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5">
                  Start Training →
                </Link>
              </li>
              <li>
                <span className="text-[0.72rem] font-semibold text-ink-5 bg-tint px-2.5 py-1 rounded-full uppercase tracking-wider">
                  Local Mode
                </span>
              </li>
            </>
          ) : user ? (
            <>
              <li>
                <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5">
                  Start Training →
                </Link>
              </li>
              <li>
                <UserMenu />
              </li>
            </>
          ) : (
            <>
              {variant === 'app' && (
                <li>
                  <Link
                    href="/login"
                    className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink transition-colors duration-300"
                  >
                    Sign in
                  </Link>
                </li>
              )}
              <li>
                <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5">
                  Get started →
                </Link>
              </li>
            </>
          )}
          <li><ThemeToggle /></li>
        </ul>

        {/* Mobile: theme switch and hamburger */}
        <div className="md:hidden flex items-center gap-2">
          <ThemeToggle />
          <button
            className="md:hidden flex flex-col justify-center items-center w-8 h-8 gap-1.5 border-none bg-transparent cursor-pointer"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            <span className={`block w-5 h-0.5 bg-ink transition-all duration-200 ${menuOpen ? 'rotate-45 translate-y-2' : ''}`} />
            <span className={`block w-5 h-0.5 bg-ink transition-all duration-200 ${menuOpen ? 'opacity-0' : ''}`} />
            <span className={`block w-5 h-0.5 bg-ink transition-all duration-200 ${menuOpen ? '-rotate-45 -translate-y-2' : ''}`} />
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="md:hidden absolute top-full left-0 w-full bg-paper border-b border-ink/5 shadow-[0_8px_24px_rgba(0,0,0,0.08)] px-4 py-4 flex flex-col gap-1">
          {variant === 'landing' ? (
            <>
              <a href="/#how-it-works" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">How it works</a>
              <a href="/#pricing" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Pricing</a>
              <a href="/#consulting" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Consulting</a>
              {user && <Link href="/platform" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Platform</Link>}
              <Link href="/blog" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Blog</Link>
            </>
          ) : (
            <>
              {user && <Link href="/jobs" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">My Jobs</Link>}
              {user && !LOCAL_MODE && <Link href="/credits" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Credits</Link>}
              <Link href="/docs" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Docs</Link>
              {isAdmin && <Link href="/admin" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Admin</Link>}
            </>
          )}
          {LOCAL_MODE ? (
            <>
              <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5 mt-2 justify-center">Start Training →</Link>
              <span className="text-[0.72rem] font-semibold text-ink-5 mt-2 text-center uppercase tracking-wider">Local Mode</span>
            </>
          ) : user ? (
            <>
              <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5 mt-2 justify-center">Start Training →</Link>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-ink/5">
                <span className="text-[0.875rem] text-ink-5">{user.email}</span>
                <UserMenu />
              </div>
            </>
          ) : (
            <>
              {variant === 'app' && (
                <Link href="/login" className="no-underline font-medium text-[0.95rem] text-ink-5 hover:text-ink py-2.5 border-b border-ink/5">Sign in</Link>
              )}
              <Link href="/train" className="btn-primary text-[0.9rem] px-5 py-2.5 mt-2 justify-center">Get started →</Link>
            </>
          )}
        </div>
      )}
    </nav>
  )
}
