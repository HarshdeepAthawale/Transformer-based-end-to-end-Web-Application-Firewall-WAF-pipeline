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

const SIDEBAR_STORAGE_KEY = 'waf-sidebar-collapsed'
const SIDEBAR_KEYBOARD_SHORTCUT = 'b'
const HOVER_LEAVE_DELAY_MS = 180

type NavItem =
  | { icon: React.ComponentType<{ size?: number; style?: React.CSSProperties; className?: string }>; label: string; href: string; adminOnly?: boolean }
  | { icon: React.ComponentType<{ size?: number; style?: React.CSSProperties; className?: string }>; label: string; action: 'logout'; adminOnly?: boolean }

function isLogoutItem(item: NavItem): item is NavItem & { action: 'logout' } {
  return 'action' in item && item.action === 'logout'
}

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true)
  const [isPinned, setIsPinned] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const pathname = usePathname()
  const router = useRouter()
  const { data: session } = useSession()
  const isAdmin = (session?.user as { role?: string })?.role === 'admin'
  const leaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const desktopExpanded = isPinned || isHovered

  // Restore pinned state from localStorage (desktop only): 'false' = pinned (start expanded), 'true' = unpinned (start collapsed)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(SIDEBAR_STORAGE_KEY)
      if (stored === 'false') setIsPinned(true)
    } catch {
      // ignore
    }
  }, [])

  const persistCollapsed = useCallback((collapsed: boolean) => {
    try {
      localStorage.setItem(SIDEBAR_STORAGE_KEY, String(collapsed))
    } catch {
      // ignore
    }
  }, [])

  const togglePinned = useCallback(() => {
    setIsPinned((prev) => {
      const next = !prev
      persistCollapsed(!next)
      return next
    })
  }, [persistCollapsed])

  const handleDesktopMouseEnter = useCallback(() => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current)
      leaveTimeoutRef.current = null
    }
    setIsHovered(true)
  }, [])

  const handleDesktopMouseLeave = useCallback(() => {
    if (isPinned) return
    leaveTimeoutRef.current = setTimeout(() => {
      setIsHovered(false)
      leaveTimeoutRef.current = null
    }, HOVER_LEAVE_DELAY_MS)
  }, [isPinned])

  useEffect(() => {
    return () => {
      if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    }
  }, [])

  const navRef = useRef<HTMLElement>(null)

  // Keep active nav item in view when route changes (e.g. after clicking Users, Audit Logs, Settings)
  useEffect(() => {
    const nav = navRef.current
    const el = nav?.querySelector<HTMLElement>('[data-active="true"]')
    if (!el || !nav) return
    const navRect = nav.getBoundingClientRect()
    const elRect = el.getBoundingClientRect()
    const isOutOfView = elRect.top < navRect.top || elRect.bottom > navRect.bottom
    if (isOutOfView) {
      el.scrollIntoView({ block: 'nearest', behavior: 'auto' })
    }
  }, [pathname])

  // Keyboard shortcut: Ctrl/Cmd + B to pin/unpin sidebar (desktop)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === SIDEBAR_KEYBOARD_SHORTCUT && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        togglePinned()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [togglePinned])

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
          className="text-sm font-medium whitespace-nowrap overflow-hidden"
          style={{ fontFamily: 'var(--font-space-grotesk)' }}
        >
          {item.label}
        </span>
      </>
    )
  }

  const sidebarContent = (
    <>
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
        <div className="overflow-hidden">
          <h1 className="font-bold whitespace-nowrap" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>WAF</h1>
          <p className="text-xs whitespace-nowrap" style={{ color: 'var(--positivus-gray-dark)' }}>Dashboard</p>
        </div>
      </Link>
      <nav ref={navRef} className="flex-1 min-h-0 p-4 space-y-2 overflow-y-auto">
        {menuItems.map((item) => {
          const href = isLogoutItem(item) ? '' : item.href
          const active = !isLogoutItem(item) && isActive(href)
          return (
            <button
              key={item.label}
              data-active={active}
              onClick={() => handleItemClick(item)}
              className={`w-full flex items-center gap-3 rounded-none px-4 py-3 transition-colors ${
                active ? '' : 'hover:bg-accent'
              }`}
              style={{
                backgroundColor: active ? 'var(--positivus-green)' : 'transparent',
                color: 'var(--positivus-black)',
              }}
            >
              {navItemContent(item, active)}
            </button>
          )
        })}
      </nav>
      <div
        className="hidden lg:flex items-center justify-center p-2 border-t shrink-0"
        style={{ borderColor: 'var(--positivus-gray)' }}
      >
        <button
          type="button"
          onClick={togglePinned}
          className="p-2 rounded-none transition-colors hover:bg-[var(--positivus-green-bg)]"
          style={{ color: 'var(--positivus-gray-dark)' }}
          aria-label={isPinned ? 'Unpin sidebar' : 'Pin sidebar open'}
          title={isPinned ? 'Unpin sidebar (Ctrl+B)' : 'Pin sidebar open (Ctrl+B)'}
        >
          {isPinned ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>
      </div>
    </>
  )

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

      {/* Desktop: hover-to-expand wrapper with thin strip when collapsed */}
      <div
        className="hidden lg:flex shrink-0 h-screen flex-col overflow-hidden transition-[width] duration-200 ease-in-out"
        style={{
          width: desktopExpanded ? 256 : 12,
          backgroundColor: 'var(--positivus-white)',
          borderRight: '2px solid var(--positivus-gray)',
        }}
        onMouseEnter={handleDesktopMouseEnter}
        onMouseLeave={handleDesktopMouseLeave}
      >
        <aside
          className="h-full w-64 flex flex-col shrink-0 relative"
          style={{ backgroundColor: 'var(--positivus-white)' }}
        >
          {/* Grip strip (visible when collapsed; first 12px of sidebar) */}
          <div
            className="absolute left-0 top-0 bottom-0 w-3 flex items-center justify-center shrink-0 z-10"
            style={{ backgroundColor: 'var(--positivus-white)', borderRight: '1px solid var(--positivus-gray)' }}
            aria-hidden
          >
            <ChevronRight size={14} style={{ color: 'var(--positivus-gray-dark)' }} />
          </div>
          <div className="pl-3 flex flex-col flex-1 min-w-0 min-h-0">
            {sidebarContent}
          </div>
        </aside>
      </div>

      {/* Mobile: overlay drawer */}
      <aside
        className={`${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:hidden fixed h-screen flex flex-col min-h-0 z-40 shadow-lg shrink-0 w-64 transition-[transform] duration-200 ease-in-out`}
        style={{ backgroundColor: 'var(--positivus-white)', borderRight: '2px solid var(--positivus-gray)' }}
      >
        {sidebarContent}
      </aside>
    </>
  )
}
