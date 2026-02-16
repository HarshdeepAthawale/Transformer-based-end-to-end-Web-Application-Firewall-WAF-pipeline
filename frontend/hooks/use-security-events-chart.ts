'use client'

import { useEffect, useState, useCallback } from 'react'
import { chartsApi } from '@/lib/api'
import { formatTimeLocal } from '@/lib/chart-utils'

const MAX_POINTS = 60

export interface SecurityEventsChartPoint {
  time: string
  timeFormatted: string
  rateLimit: number
  ddos: number
}

export function useSecurityEventsChart(timeRange: string, timezone?: string) {
  const [data, setData] = useState<SecurityEventsChartPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setError(null)
    try {
      setIsLoading(true)
      const [rateLimitRes, ddosRes] = await Promise.all([
        chartsApi.getRateLimit(timeRange),
        chartsApi.getDdos(timeRange),
      ])

      const rateLimitMap = new Map<string, number>()
      if (rateLimitRes.success && rateLimitRes.data?.length) {
        rateLimitRes.data.forEach((p) => {
          const k = p.time || ''
          rateLimitMap.set(k, (rateLimitMap.get(k) ?? 0) + (p.count ?? 0))
        })
      }

      const ddosMap = new Map<string, number>()
      if (ddosRes.success && ddosRes.data?.length) {
        ddosRes.data.forEach((p) => {
          const k = p.time || ''
          ddosMap.set(k, (ddosMap.get(k) ?? 0) + (p.count ?? 0))
        })
      }

      const allTimes = new Set([
        ...rateLimitMap.keys(),
        ...ddosMap.keys(),
      ])
      const merged = Array.from(allTimes)
        .filter(Boolean)
        .sort()
        .map((time) => ({
          time,
          timeFormatted: formatTimeLocal(time, timezone),
          rateLimit: rateLimitMap.get(time) ?? 0,
          ddos: ddosMap.get(time) ?? 0,
        }))
        .slice(-MAX_POINTS)

      setData(merged)
    } catch (err) {
      setError('Failed to load security events')
      setData([])
    } finally {
      setIsLoading(false)
    }
  }, [timeRange, timezone])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return { data, isLoading, error, refetch: fetchData }
}
