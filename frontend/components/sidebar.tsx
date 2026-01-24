'use client'

import { useState } from 'react'
import { usePathname, useRouter } from 'next/navigation'
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
  Zap,
  Lock,
} from 'lucide-react'

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true)
  const pathname = usePathname()
  const router = useRouter()

  const menuItems = [
    { icon: Home, label: 'Overview', href: '/' },
    { icon: BarChart3, label: 'Analytics', href: '/analytics' },
    { icon: Eye, label: 'Traffic', href: '/traffic' },
    { icon: AlertTriangle, label: 'Threats', href: '/threats' },
    { icon: Zap, label: 'Performance', href: '/performance' },
    { icon: Lock, label: 'Security', href: '/security' },
  ]

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/'
    return pathname.startsWith(href)
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-card text-foreground rounded-lg"
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
        } fixed lg:relative w-64 h-screen bg-sidebar border-r border-sidebar-border flex flex-col transition-transform duration-300 z-40 lg:z-0 shadow-lg lg:shadow-none`}
      >
        {/* Logo */}
        <div className="p-6 border-b border-sidebar-border flex items-center gap-3">
            <div className="p-2 bg-black rounded-lg">
              <Shield size={24} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-sidebar-foreground">WAF</h1>
            <p className="text-xs text-sidebar-foreground/60">Dashboard</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            return (
              <button
                key={item.label}
                onClick={() => router.push(item.href)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors group ${
                  active
                    ? 'bg-black text-white'
                    : 'text-sidebar-foreground hover:bg-sidebar-accent'
                }`}
              >
                <Icon
                  size={20}
                  className={`transition-colors ${
                    active
                      ? 'text-white'
                      : 'text-sidebar-foreground/70 group-hover:text-black'
                  }`}
                />
                <span className="text-sm font-medium">{item.label}</span>
              </button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-sidebar-border space-y-2">
          <button
            onClick={() => router.push('/settings')}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sidebar-foreground hover:bg-sidebar-accent transition-colors group"
          >
            <Settings size={20} className="text-sidebar-foreground/70" />
            <span className="text-sm font-medium">Settings</span>
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sidebar-foreground hover:bg-sidebar-accent transition-colors group">
            <LogOut size={20} className="text-sidebar-foreground/70" />
            <span className="text-sm font-medium">Logout</span>
          </button>
        </div>
      </aside>
    </>
  )
}
