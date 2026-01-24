'use client'

import { ChevronDown, Search, Bell, Settings } from 'lucide-react'
import { StatusIndicator } from './status-indicator'

interface HeaderProps {
  timeRange: string
  onTimeRangeChange: (range: string) => void
}

export function Header({ timeRange, onTimeRangeChange }: HeaderProps) {
  const timeRanges = [
    { label: '1h', value: '1h' },
    { label: '6h', value: '6h' },
    { label: '24h', value: '24h' },
    { label: '7d', value: '7d' },
    { label: '30d', value: '30d' },
  ]

  return (
    <header className="border-b border-border bg-card/40 backdrop-blur-sm sticky top-0 z-30">
      <div className="px-6 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-6 flex-1">
          <h1 className="text-2xl font-bold">Dashboard</h1>
        </div>

        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="hidden md:flex items-center gap-2 bg-input rounded-lg px-4 py-2">
            <Search size={18} className="text-muted-foreground" />
            <input
              type="text"
              placeholder="Search threats..."
              className="bg-transparent border-none outline-none text-sm text-foreground placeholder-muted-foreground w-48"
            />
          </div>

          {/* Time Range Dropdown */}
          <div className="flex items-center gap-2 bg-input rounded-lg px-3 py-2">
            <select
              value={timeRange}
              onChange={(e) => onTimeRangeChange(e.target.value)}
              className="bg-transparent border-none outline-none text-sm text-foreground cursor-pointer"
            >
              {timeRanges.map((range) => (
                <option key={range.value} value={range.value}>
                  {range.label}
                </option>
              ))}
            </select>
            <ChevronDown size={16} className="text-muted-foreground pointer-events-none" />
          </div>


          {/* Notification Bell */}
          <button className="p-2 hover:bg-input rounded-lg transition-colors relative">
            <Bell size={20} className="text-foreground" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full" />
          </button>

          {/* Settings */}
          <button className="p-2 hover:bg-input rounded-lg transition-colors">
            <Settings size={20} className="text-foreground" />
          </button>
        </div>
      </div>
    </header>
  )
}
