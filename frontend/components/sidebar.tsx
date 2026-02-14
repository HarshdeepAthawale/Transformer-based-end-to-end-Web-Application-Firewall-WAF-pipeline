'use client'

import { useState, useMemo } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { useSession, signOut } from 'next-auth/react'
import Link from 'next/link'
import {
  Shield,
  BarChart3,
  AlertTriangle,
  Settings,
  LogOut,
  Menu,
  X,
  Home,
  Eye,
  Lock,
  Ban,
  Globe,
  Bot,
  FileSearch,
  FileText,
  Users,
  ClipboardList,
} from 'lucide-react'

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true)
  const pathname = usePathname()
  const router = useRouter()
  const { data: session } = useSession()
  const isAdmin = (session?.user as { role?: string })?.role === 'admin'

  const menuItems = useMemo(() => {
    const items = [
      { icon: Home, label: 'Overview', href: '/dashboard' },
      { icon: BarChart3, label: 'Analytics', href: '/analytics' },
      { icon: Eye, label: 'Traffic', href: '/traffic' },
      { icon: AlertTriangle, label: 'Threats', href: '/threats' },
      { icon: Ban, label: 'IP Management', href: '/ip-management' },
      { icon: Globe, label: 'Geo Rules', href: '/geo-rules' },
      { icon: Bot, label: 'Bot Detection', href: '/bot-detection' },
      { icon: FileSearch, label: 'Threat Intel', href: '/threat-intelligence' },
      { icon: ClipboardList, label: 'Security Rules', href: '/security-rules' },
      { icon: Users, label: 'Users', href: '/users', adminOnly: true },
      { icon: FileText, label: 'Audit Logs', href: '/audit-logs' },
    ]
    return items.filter((item) => !(item as { adminOnly?: boolean }).adminOnly || isAdmin)
  }, [isAdmin])

  const isActive = (href: string) => {
    if (href === '/dashboard') return pathname === '/dashboard'
    return pathname.startsWith(href)
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-none border-2"
        style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
      >
        {isOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside
        className={`${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        } fixed lg:relative w-64 h-screen flex flex-col transition-transform duration-300 z-40 lg:z-0 shadow-lg lg:shadow-none`}
        style={{ backgroundColor: 'var(--positivus-white)', borderRight: '2px solid var(--positivus-gray)' }}
      >
        {/* Logo */}
        <Link href="/" className="p-6 flex items-center gap-3 hover:opacity-80 transition-opacity" style={{ borderBottom: '2px solid var(--positivus-gray)' }}>
          <div
            className="p-2 rounded-none"
            style={{ backgroundColor: 'var(--positivus-green-bg)' }}
          >
            <Shield size={24} style={{ color: 'var(--positivus-green)' }} />
          </div>
          <div>
            <h1 className="font-bold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>WAF</h1>
            <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>Dashboard</p>
          </div>
        </Link>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            return (
              <button
                key={item.label}
                onClick={() => router.push(item.href)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-none transition-colors group ${
                  active ? '' : 'hover:bg-accent'
                }`}
                style={{
                  backgroundColor: active ? 'var(--positivus-green)' : 'transparent',
                  color: 'var(--positivus-black)',
                }}
              >
                <Icon
                  size={20}
                  style={{ color: active ? 'var(--positivus-black)' : 'var(--positivus-gray-dark)' }}
                  className="group-hover:transition-colors"
                />
                <span className="text-sm font-medium" style={{ fontFamily: 'var(--font-space-grotesk)' }}>{item.label}</span>
              </button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 space-y-2" style={{ borderTop: '2px solid var(--positivus-gray)' }}>
          <button
            onClick={() => router.push('/settings')}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-none transition-colors hover:bg-[var(--positivus-green-bg)]"
            style={{ color: 'var(--positivus-black)' }}
          >
            <Settings size={20} style={{ color: 'var(--positivus-gray-dark)' }} />
            <span className="text-sm font-medium" style={{ fontFamily: 'var(--font-space-grotesk)' }}>Settings</span>
          </button>
          <button
            onClick={() => signOut({ callbackUrl: '/' })}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-none transition-colors hover:bg-[var(--positivus-green-bg)]"
            style={{ color: 'var(--positivus-black)' }}
          >
            <LogOut size={20} style={{ color: 'var(--positivus-gray-dark)' }} />
            <span className="text-sm font-medium" style={{ fontFamily: 'var(--font-space-grotesk)' }}>Logout</span>
          </button>
        </div>
      </aside>
    </>
  )
}
