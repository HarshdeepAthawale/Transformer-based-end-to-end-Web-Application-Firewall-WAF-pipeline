'use client'

import Link from 'next/link'
import { ChevronDown, Search, Settings, Globe } from 'lucide-react'
import { ThemeToggle } from '@/components/theme-toggle'
import { NotificationCenter } from '@/components/notification-center'
import { useTimezone, TIMEZONE_OPTIONS } from '@/contexts/timezone-context'
import { useDomain } from '@/contexts/domain-context'

interface HeaderProps {
  timeRange?: string
  onTimeRangeChange?: (range: string) => void
}

export function Header({ timeRange = '24h', onTimeRangeChange = () => {} }: HeaderProps) {
  const { timezone, setTimezone } = useTimezone()
  const { selectedDomain } = useDomain()
  const timeRanges = [
    { label: '1h', value: '1h' },
    { label: '6h', value: '6h' },
    { label: '24h', value: '24h' },
    { label: '7d', value: '7d' },
    { label: '30d', value: '30d' },
    { label: '90d', value: '90d' },
  ]

  return (
    <header
      className="border-b-2 sticky top-0 z-30"
      style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
    >
      <div className="px-6 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-6 flex-1">
          <div className="flex items-center gap-2">
            <h1
              className="text-2xl font-bold"
              style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              Dashboard
            </h1>
            {selectedDomain && (
              <>
                <span className="text-lg" style={{ color: 'var(--positivus-gray-dark)' }}>/</span>
                <span
                  className="text-lg font-medium"
                  style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                >
                  {selectedDomain.domain}
                </span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Search */}
          <div
            className="hidden md:flex items-center gap-2 rounded-none px-4 py-2 border-2"
            style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
          >
            <Search size={18} style={{ color: 'var(--positivus-gray-dark)' }} />
            <input
              type="text"
              placeholder="Search threats..."
              className="bg-transparent border-none outline-none text-sm w-48 placeholder-[var(--positivus-gray-dark)]"
              style={{ color: 'var(--positivus-black)' }}
            />
          </div>

          {/* Time Range Dropdown */}
          <div
            className="flex items-center gap-2 rounded-none px-3 py-2 border-2"
            style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
          >
            <select
              value={timeRange}
              onChange={(e) => onTimeRangeChange(e.target.value)}
              className="bg-transparent border-none outline-none text-sm cursor-pointer"
              style={{ color: 'var(--positivus-black)' }}
            >
              {timeRanges.map((range) => (
                <option key={range.value} value={range.value}>
                  {range.label}
                </option>
              ))}
            </select>
            <ChevronDown size={16} style={{ color: 'var(--positivus-gray-dark)' }} className="pointer-events-none" />
          </div>

          {/* Timezone - client company's country */}
          <div
            className="flex items-center gap-2 rounded-none px-3 py-2 border-2"
            style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
            title="Display times in your company's timezone"
          >
            <Globe size={16} style={{ color: 'var(--positivus-gray-dark)' }} />
            <select
              value={timezone ?? ''}
              onChange={(e) => setTimezone(e.target.value || undefined)}
              className="bg-transparent border-none outline-none text-sm cursor-pointer max-w-[140px]"
              style={{ color: 'var(--positivus-black)' }}
            >
              {TIMEZONE_OPTIONS.map((opt) => (
                <option key={opt.value || 'local'} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Notification Center */}
          <NotificationCenter />

          {/* Settings */}
          <Link
            href="/settings"
            className="p-2 rounded-none transition-colors hover:bg-accent"
            style={{ color: 'var(--positivus-black)' }}
            aria-label="Settings"
          >
            <Settings size={20} />
          </Link>
        </div>
      </div>
    </header>
  )
}
