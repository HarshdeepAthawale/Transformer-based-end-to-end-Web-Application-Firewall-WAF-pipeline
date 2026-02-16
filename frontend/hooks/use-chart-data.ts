'use client'

import { useEffect, useState, useCallback } from 'react'
import { chartsApi, trafficApi, threatsApi, wsManager } from '@/lib/api'
import type { ChartDataPoint } from '@/lib/api'
import {
  roundToMinute,
  aggregateByMinute,
  buildTop10ThreatTypes,
  buildTop10FromThreats,
  REALTIME_RANGES,
  type TopThreatItem,
  type ChartTimeRangeValue,
} from '@/lib/chart-utils'

const MAX_POINTS = 60
const POLL_INTERVAL_MS = 30_000

export interface UseChartDataResult {
  requestData: ChartDataPoint[]
  topThreatTypes: TopThreatItem[]
  isLoading: boolean
  error: string | null
  refetch: () => Promise<void>
}

export function useChartData(timeRange: string): UseChartDataResult {
  const [requestData, setRequestData] = useState<ChartDataPoint[]>([])
  const [topThreatTypes, setTopThreatTypes] = useState<TopThreatItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTrafficByMinute = useCallback(async () => {
    try {
      const response = await trafficApi.getRecent(1000)
      if (!response.success) return
      const minuteMap = new Map<string, { requests: number; blocked: number; allowed: number }>()
      response.data.forEach((traffic: { timestamp?: string; time?: string; was_blocked?: boolean; wasBlocked?: boolean }) => {
        const timestamp = traffic.timestamp || traffic.time
        if (!timestamp) return
        const minuteKey = roundToMinute(timestamp)
        const existing = minuteMap.get(minuteKey) || { requests: 0, blocked: 0, allowed: 0 }
        const wasBlocked = traffic.was_blocked ?? traffic.wasBlocked ?? false
        minuteMap.set(minuteKey, {
          requests: existing.requests + 1,
          blocked: existing.blocked + (wasBlocked ? 1 : 0),
          allowed: existing.allowed + (wasBlocked ? 0 : 1),
        })
      })
      const chartData: ChartDataPoint[] = Array.from(minuteMap.entries())
        .map(([time, counts]) => ({ time, ...counts }))
        .sort((a, b) => new Date(a.time || '').getTime() - new Date(b.time || '').getTime())
        .slice(-MAX_POINTS)
      setRequestData(chartData)
    } catch (err: unknown) {
      if ((err as { isNetworkError?: boolean })?.isNetworkError) {
        console.debug('[useChartData] Traffic fetch failed (backend may be offline)')
      } else {
        console.error('[useChartData] Failed to fetch traffic:', err)
      }
    }
  }, [])

  const fetchTopThreats = useCallback(
    async (range: string) => {
      try {
        const statsRes = await threatsApi.getStats(range)
        if (statsRes.success && statsRes.data && Object.keys(statsRes.data).length > 0) {
          setTopThreatTypes(buildTop10ThreatTypes(statsRes.data))
          return
        }
        const threatsRes = await threatsApi.getByTimeRange(range)
        if (threatsRes.success && threatsRes.data?.length) {
          setTopThreatTypes(buildTop10FromThreats(threatsRes.data))
        }
      } catch {
        try {
          const threatsRes = await threatsApi.getByTimeRange(range)
          if (threatsRes.success && threatsRes.data?.length) {
            setTopThreatTypes(buildTop10FromThreats(threatsRes.data))
          }
        } catch {
          // ignore
        }
      }
    },
    []
  )

  const refetch = useCallback(async () => {
    setError(null)
    const range = timeRange as ChartTimeRangeValue
    const useRealtime = REALTIME_RANGES.includes(range)

    try {
      setIsLoading(true)
      if (useRealtime) {
        await fetchTrafficByMinute()
      } else {
        const res = await chartsApi.getRequests(timeRange)
        if (res.success && res.data?.length) {
          const aggregated = aggregateByMinute(res.data)
          setRequestData(aggregated.slice(-MAX_POINTS))
        } else {
          setRequestData([])
        }
      }
      await fetchTopThreats(timeRange)
    } catch (err: unknown) {
      if ((err as { isNetworkError?: boolean })?.isNetworkError) {
        setError(null)
        setRequestData([])
        setTopThreatTypes([])
      } else {
        setError('Failed to load chart data')
        console.error('[useChartData]', err)
      }
    } finally {
      setIsLoading(false)
    }
  }, [timeRange, fetchTrafficByMinute, fetchTopThreats])

  useEffect(() => {
    setRequestData([])
    setTopThreatTypes([])
    refetch()
  }, [timeRange, refetch])

  useEffect(() => {
    const poll = async () => {
      const range = timeRange as ChartTimeRangeValue
      if (REALTIME_RANGES.includes(range)) {
        await fetchTrafficByMinute()
      }
      await fetchTopThreats(timeRange)
    }
    const interval = setInterval(poll, POLL_INTERVAL_MS)

    const handleTraffic = (data: { was_blocked?: boolean }) => {
      const now = new Date()
      const minuteKey = roundToMinute(now.toISOString())
      setRequestData((prev) => {
        const idx = prev.findIndex((p) => roundToMinute(p.time || '') === minuteKey)
        if (idx >= 0) {
          const next = [...prev]
          const cur = next[idx]
          next[idx] = {
            ...cur,
            requests: (cur.requests ?? 0) + 1,
            blocked: (cur.blocked ?? 0) + (data.was_blocked ? 1 : 0),
            allowed: (cur.allowed ?? 0) + (data.was_blocked ? 0 : 1),
          }
          return next.slice(-MAX_POINTS)
        }
        const newPoint: ChartDataPoint = {
          time: minuteKey,
          requests: 1,
          blocked: data.was_blocked ? 1 : 0,
          allowed: data.was_blocked ? 0 : 1,
        }
        return [...prev, newPoint].slice(-MAX_POINTS)
      })
    }

    const handleThreat = () => fetchTopThreats(timeRange)

    wsManager.subscribe('traffic', handleTraffic)
    wsManager.subscribe('threat', handleThreat)
    return () => {
      clearInterval(interval)
      wsManager.unsubscribe('traffic')
      wsManager.unsubscribe('threat')
    }
  }, [timeRange, fetchTrafficByMinute, fetchTopThreats])

  return { requestData, topThreatTypes, isLoading, error, refetch }
}
