'use client'

import { createContext, useContext, useState, useCallback, useEffect } from 'react'

const STORAGE_KEY = 'waf-timezone'

/** Common timezones by region for client company display */
export const TIMEZONE_OPTIONS = [
  { value: '', label: 'Local (browser)', description: 'Your device timezone' },
  { value: 'UTC', label: 'UTC' },
  { value: 'America/New_York', label: 'Eastern (US)' },
  { value: 'America/Chicago', label: 'Central (US)' },
  { value: 'America/Los_Angeles', label: 'Pacific (US)' },
  { value: 'Europe/London', label: 'London' },
  { value: 'Europe/Paris', label: 'Paris' },
  { value: 'Europe/Berlin', label: 'Berlin' },
  { value: 'Asia/Dubai', label: 'Dubai' },
  { value: 'Asia/Kolkata', label: 'India (IST)' },
  { value: 'Asia/Singapore', label: 'Singapore' },
  { value: 'Asia/Tokyo', label: 'Tokyo' },
  { value: 'Australia/Sydney', label: 'Sydney' },
] as const

type TimezoneContextValue = {
  /** Selected IANA timezone, or undefined for browser local */
  timezone: string | undefined
  setTimezone: (tz: string | undefined) => void
}

const TimezoneContext = createContext<TimezoneContextValue | null>(null)

export function TimezoneProvider({ children }: { children: React.ReactNode }) {
  const [timezone, setTimezoneState] = useState<string | undefined>(undefined)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY)
      setTimezoneState(stored === '' ? undefined : stored ?? undefined)
      setMounted(true)
    }
  }, [])

  const setTimezone = useCallback((tz: string | undefined) => {
    setTimezoneState(tz)
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, tz ?? '')
    }
  }, [])

  if (!mounted) {
    return <>{children}</>
  }

  return (
    <TimezoneContext.Provider value={{ timezone: timezone || undefined, setTimezone }}>
      {children}
    </TimezoneContext.Provider>
  )
}

export function useTimezone() {
  const ctx = useContext(TimezoneContext)
  return ctx ?? { timezone: undefined, setTimezone: () => {} }
}
