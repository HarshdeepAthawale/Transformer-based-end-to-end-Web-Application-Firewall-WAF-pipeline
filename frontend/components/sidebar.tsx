'use client'

import { useState, useMemo, useEffect, useCallback, useRef } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { useSession, signOut } from 'next-auth/react'
import Link from 'next/link'
import {
  Shield,
  ShieldCheck,
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
  Package,
  Cpu,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

const SIDEBAR_STORAGE_KEY = 'waf-sidebar-collapsed'
const SIDEBAR_KEYBOARD_SHORTCUT = 'b'

type NavItem =
  | { icon: React.ComponentType<{ size?: number; style?: React.CSSProperties; className?: string }>; label: string; href: string; adminOnly?: boolean }
  | { icon: React.ComponentType<{ size?: number; style?: React.CSSProperties; className?: string }>; label: string; action: 'logout'; adminOnly?: boolean }

function isLogoutItem(item: NavItem): item is NavItem & { action: 'logout' } {
  return 'action' in item && item.action === 'logout'
}

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true)
  const [isCollapsed, setIsCollapsed] = useState(false)
  const pathname = usePathname()
  const router = useRouter()
  const { data: session } = useSession()
  const isAdmin = (session?.user as { role?: string })?.role === 'admin'

  // Restore collapsed state from localStorage (desktop only, client-side)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(SIDEBAR_STORAGE_KEY)
      if (stored !== null) setIsCollapsed(stored === 'true')
    } catch {
      // ignore
    }
  }, [])

  const persistCollapsed = useCallback((value: boolean) => {
    try {
      localStorage.setItem(SIDEBAR_STORAGE_KEY, String(value))
    } catch {
      // ignore
    }
  }, [])

  const toggleCollapsed = useCallback(() => {
    setIsCollapsed((prev) => {
      const next = !prev
      persistCollapsed(next)
      return next
    })
  }, [persistCollapsed])

  const navRef = useRef<HTMLElement>(null)

  // Keep active nav item in view when route changes (e.g. after clicking Users, Audit Logs, Settings)
  useEffect(() => {
    const el = navRef.current?.querySelector<HTMLElement>('[data-active="true"]')
    if (el) {
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [pathname])

  // Keyboard shortcut: Ctrl/Cmd + B to toggle sidebar (desktop)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === SIDEBAR_KEYBOARD_SHORTCUT && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        toggleCollapsed()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [toggleCollapsed])

  const menuItems = useMemo((): NavItem[] => {
    const items: NavItem[] = [
      { icon: Home, label: 'Overview', href: '/dashboard' },
      { icon: Bot, label: 'AI Copilot', href: '/copilot' },
      { icon: BarChart3, label: 'Analytics', href: '/analytics' },
      { icon: Eye, label: 'Traffic', href: '/traffic' },
      { icon: AlertTriangle, label: 'Threats', href: '/threats' },
      { icon: ShieldCheck, label: 'DoS/DDoS Protection', href: '/dos-protection' },
      { icon: FileSearch, label: 'Upload Scanning', href: '/upload-scanning' },
      { icon: Cpu, label: 'Firewall for AI', href: '/firewall-ai' },
      { icon: Lock, label: 'Credential protection', href: '/credential-protection' },
      { icon: Ban, label: 'IP Management', href: '/ip-management' },
      { icon: Globe, label: 'Geo Rules', href: '/geo-rules' },
      { icon: Bot, label: 'Bot Detection', href: '/bot-detection' },
      { icon: FileSearch, label: 'Threat Intel', href: '/threat-intelligence' },
      { icon: ClipboardList, label: 'Security Rules', href: '/security-rules' },
      { icon: Package, label: 'Managed Rules', href: '/managed-rules' },
      { icon: Users, label: 'Users', href: '/users', adminOnly: true },
      { icon: FileText, label: 'Audit Logs', href: '/audit-logs' },
      { icon: Settings, label: 'Settings', href: '/settings' },
      { icon: LogOut, label: 'Logout', action: 'logout' },
    ]
    return items.filter((item) => !item.adminOnly || isAdmin)
  }, [isAdmin])

  const isActive = (href: string) => {
    if (href === '/dashboard') return pathname === '/dashboard'
    return pathname.startsWith(href)
  }

  const handleItemClick = (item: NavItem) => {
    if (isLogoutItem(item)) {
      signOut({ callbackUrl: '/' })
      return
    }
    router.push(item.href)
  }

  const navItemContent = (item: NavItem, active: boolean) => {
    const Icon = item.icon
    return (
      <>
        <Icon
          size={20}
          style={{ color: active ? 'var(--positivus-black)' : 'var(--positivus-gray-dark)' }}
          className="shrink-0"
        />
        <span
          className="text-sm font-medium whitespace-nowrap overflow-hidden transition-[opacity,width] duration-200 ease-in-out"
          style={{
            fontFamily: 'var(--font-space-grotesk)',
            opacity: isCollapsed ? 0 : 1,
            width: isCollapsed ? 0 : undefined,
            minWidth: isCollapsed ? 0 : undefined,
          }}
        >
          {item.label}
        </span>
      </>
    )
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
        } fixed lg:relative h-screen flex flex-col z-40 lg:z-0 shadow-lg lg:shadow-none shrink-0 transition-[width] duration-200 ease-in-out w-64 ${
          isCollapsed ? 'lg:w-16' : 'lg:w-64'
        }`}
        style={{ backgroundColor: 'var(--positivus-white)', borderRight: '2px solid var(--positivus-gray)' }}
      >
        {/* Logo */}
          <Link
            href="/"
            className="p-4 lg:p-6 flex items-center gap-3 hover:opacity-80 transition-opacity shrink-0"
            style={{ borderBottom: '2px solid var(--positivus-gray)' }}
          >
            <div
              className="p-2 rounded-none shrink-0"
              style={{ backgroundColor: 'var(--positivus-green-bg)' }}
            >
              <Shield size={24} style={{ color: 'var(--positivus-green)' }} />
            </div>
            <div
              className="overflow-hidden transition-[opacity,width] duration-200 ease-in-out"
              style={{
                opacity: isCollapsed ? 0 : 1,
                width: isCollapsed ? 0 : undefined,
                minWidth: isCollapsed ? 0 : undefined,
              }}
            >
              <h1 className="font-bold whitespace-nowrap" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>WAF</h1>
              <p className="text-xs whitespace-nowrap" style={{ color: 'var(--positivus-gray-dark)' }}>Dashboard</p>
            </div>
          </Link>

          {/* Navigation - single list including Settings and Logout */}
          <nav ref={navRef} className="flex-1 p-4 space-y-2 overflow-y-auto">
            {menuItems.map((item) => {
              const Icon = item.icon
              const href = isLogoutItem(item) ? '' : item.href
              const active = !isLogoutItem(item) && isActive(href)
              const button = (
                <button
                  data-active={active}
                  onClick={() => handleItemClick(item)}
                  className={`w-full flex items-center gap-3 rounded-none transition-colors group ${
                    isCollapsed ? 'justify-center px-0 py-3' : 'px-4 py-3'
                  } ${active ? '' : 'hover:bg-accent'}`}
                  style={{
                    backgroundColor: active ? 'var(--positivus-green)' : 'transparent',
                    color: 'var(--positivus-black)',
                  }}
                >
                  {navItemContent(item, active)}
                </button>
              )
              if (isCollapsed) {
                return (
                  <Tooltip key={item.label}>
                    <TooltipTrigger asChild>{button}</TooltipTrigger>
                    <TooltipContent side="right" sideOffset={8}>
                      {item.label}
                    </TooltipContent>
                  </Tooltip>
                )
              }
              return <span key={item.label}>{button}</span>
            })}
          </nav>

          {/* Toggle - desktop only */}
          <div
            className="hidden lg:flex items-center justify-center p-2 border-t shrink-0"
            style={{ borderColor: 'var(--positivus-gray)' }}
          >
            <button
              type="button"
              onClick={toggleCollapsed}
              className="p-2 rounded-none transition-colors hover:bg-[var(--positivus-green-bg)]"
              style={{ color: 'var(--positivus-gray-dark)' }}
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              title={isCollapsed ? 'Expand sidebar (Ctrl+B)' : 'Collapse sidebar (Ctrl+B)'}
            >
              {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            </button>
          </div>
      </aside>
    </>
  )
}
