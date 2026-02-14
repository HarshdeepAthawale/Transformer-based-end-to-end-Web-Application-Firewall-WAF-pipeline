'use client'

import { useSession } from 'next-auth/react'
import { signOut } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import Link from 'next/link'
import {
  Shield,
  BarChart3,
  AlertTriangle,
  Users,
  Settings,
  LogOut,
  Eye,
  Lock,
  Globe,
  Bot,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function AdminPage() {
  const { data: session, status } = useSession()
  const router = useRouter()

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login?callbackUrl=/admin')
    }
  }, [status, router])

  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse" style={{ color: 'var(--positivus-green)' }}>
          Loading...
        </div>
      </div>
    )
  }

  if (!session) {
    return null
  }

  const adminLinks = [
    { href: '/dashboard', icon: BarChart3, label: 'Dashboard Overview' },
    { href: '/analytics', icon: BarChart3, label: 'Analytics' },
    { href: '/traffic', icon: Eye, label: 'Traffic' },
    { href: '/threats', icon: AlertTriangle, label: 'Threats' },
    { href: '/ip-management', icon: Lock, label: 'IP Management' },
    { href: '/geo-rules', icon: Globe, label: 'Geo Rules' },
    { href: '/bot-detection', icon: Bot, label: 'Bot Detection' },
    { href: '/users', icon: Users, label: 'User Management' },
    { href: '/audit-logs', icon: Shield, label: 'Audit Logs' },
    { href: '/security-rules', icon: Lock, label: 'Security Rules' },
    { href: '/settings', icon: Settings, label: 'Settings' },
  ]

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <header
        className="sticky top-0 z-50 border-b"
        style={{
          backgroundColor: 'var(--positivus-white)',
          borderColor: 'var(--positivus-gray)',
        }}
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link href="/admin" className="flex items-center gap-2">
              <Shield className="w-6 h-6" style={{ color: 'var(--positivus-green)' }} />
              <span
                className="text-lg font-bold"
                style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
              >
                WAF Admin
              </span>
            </Link>
            <div className="flex items-center gap-4">
              <span className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                {session.user?.email}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => signOut({ callbackUrl: '/' })}
                className="rounded-none"
                style={{
                  borderColor: 'var(--positivus-black)',
                  color: 'var(--positivus-black)',
                }}
              >
                <LogOut className="w-4 h-4 mr-1" />
                Sign out
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1
            className="text-3xl font-bold mb-2"
            style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Admin Panel
          </h1>
          <p style={{ color: 'var(--positivus-gray-dark)' }}>
            Manage your WAF dashboard, security rules, and monitor threats.
          </p>
        </div>

        <div
          className="p-6 rounded-lg border-2 mb-8"
          style={{
            backgroundColor: 'var(--positivus-white)',
            borderColor: 'var(--positivus-gray)',
          }}
        >
          <h2
            className="text-lg font-semibold mb-4"
            style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Quick access
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {adminLinks.map((link) => {
              const Icon = link.icon
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className="flex items-center gap-3 p-4 rounded-lg transition-colors hover:bg-[var(--positivus-green-bg)]"
                  style={{
                    backgroundColor: 'var(--positivus-gray)',
                  }}
                >
                  <Icon
                    className="w-5 h-5 flex-shrink-0"
                    style={{ color: 'var(--positivus-green)' }}
                  />
                  <span className="font-medium" style={{ color: 'var(--positivus-black)' }}>
                    {link.label}
                  </span>
                </Link>
              )
            })}
          </div>
        </div>

        <div
          className="p-6 rounded-lg border-2"
          style={{
            backgroundColor: 'var(--positivus-white)',
            borderColor: 'var(--positivus-gray)',
          }}
        >
          <h2
            className="text-lg font-semibold mb-4"
            style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Session info
          </h2>
          <dl className="space-y-2 text-sm">
            <div className="flex gap-2">
              <dt style={{ color: 'var(--positivus-gray-dark)' }}>Email:</dt>
              <dd style={{ color: 'var(--positivus-black)' }}>{session.user?.email}</dd>
            </div>
            <div className="flex gap-2">
              <dt style={{ color: 'var(--positivus-gray-dark)' }}>Name:</dt>
              <dd style={{ color: 'var(--positivus-black)' }}>{session.user?.name ?? '—'}</dd>
            </div>
            <div className="flex gap-2">
              <dt style={{ color: 'var(--positivus-gray-dark)' }}>Role:</dt>
              <dd style={{ color: 'var(--positivus-black)' }}>
                {(session.user as { role?: string })?.role ?? 'user'}
              </dd>
            </div>
          </dl>
        </div>
      </main>
    </div>
  )
}
