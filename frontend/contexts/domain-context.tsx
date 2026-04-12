'use client'

import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import type { DNSZoneData } from '@/lib/api'

const STORAGE_KEY = 'waf-selected-domain'

type DomainContextValue = {
  domains: DNSZoneData[]
  selectedDomainId: number | null
  selectedDomain: DNSZoneData | null
  isLoading: boolean
  selectDomain: (id: number | null) => void
  refreshDomains: () => Promise<void>
}

const DomainContext = createContext<DomainContextValue | null>(null)

export function DomainProvider({ children }: { children: React.ReactNode }) {
  const { status } = useSession()
  const [domains, setDomains] = useState<DNSZoneData[]>([])
  const [selectedDomainId, setSelectedDomainId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Restore selection from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = parseInt(stored, 10)
        if (!isNaN(parsed)) setSelectedDomainId(parsed)
      }
      setMounted(true)
    }
  }, [])

  const fetchDomains = useCallback(async () => {
    if (status !== 'authenticated') return
    setIsLoading(true)
    try {
      const { dnsApi } = await import('@/lib/api')
      const data = await dnsApi.getZones()
      setDomains(Array.isArray(data) ? data : [])
    } catch {
      // Silently fail - domains are optional
      setDomains([])
    } finally {
      setIsLoading(false)
    }
  }, [status])

  // Fetch domains when authenticated
  useEffect(() => {
    if (status === 'authenticated' && mounted) {
      fetchDomains()
    }
  }, [status, mounted, fetchDomains])

  const selectDomain = useCallback((id: number | null) => {
    setSelectedDomainId(id)
    if (typeof window !== 'undefined') {
      if (id !== null) {
        localStorage.setItem(STORAGE_KEY, String(id))
      } else {
        localStorage.removeItem(STORAGE_KEY)
      }
    }
  }, [])

  const selectedDomain = domains.find(d => d.id === selectedDomainId) ?? null

  // If selected domain no longer exists in the list, clear selection
  useEffect(() => {
    if (selectedDomainId !== null && domains.length > 0 && !selectedDomain) {
      setSelectedDomainId(null)
      if (typeof window !== 'undefined') {
        localStorage.removeItem(STORAGE_KEY)
      }
    }
  }, [domains, selectedDomainId, selectedDomain])

  if (!mounted) {
    return <>{children}</>
  }

  return (
    <DomainContext.Provider value={{
      domains,
      selectedDomainId,
      selectedDomain,
      isLoading,
      selectDomain,
      refreshDomains: fetchDomains,
    }}>
      {children}
    </DomainContext.Provider>
  )
}

export function useDomain() {
  const ctx = useContext(DomainContext)
  return ctx ?? {
    domains: [],
    selectedDomainId: null,
    selectedDomain: null,
    isLoading: false,
    selectDomain: () => {},
    refreshDomains: async () => {},
  }
}
