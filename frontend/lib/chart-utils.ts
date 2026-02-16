/**
 * Shared chart utilities and constants for dashboard charts.
 * Used by useChartData and ChartsSection for scalable, service-backed charts.
 */

import type { ChartDataPoint } from '@/lib/api'

/** Supported time range values for charts (aligned with backend parse_time_range). */
export const CHART_TIME_RANGES = [
  { value: '1h', label: 'Last 1 hour' },
  { value: '6h', label: 'Last 6 hours' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: '90d', label: 'Last 3 months' },
] as const

export type ChartTimeRangeValue = (typeof CHART_TIME_RANGES)[number]['value']

/** Ranges that use traffic API (per-minute) instead of charts API (hourly). */
export const REALTIME_RANGES: ChartTimeRangeValue[] = ['1h', '6h', '24h']

export interface TopThreatItem {
  name: string
  count: number
}

export function formatTimeIST(timestamp: string | Date): string {
  try {
    let date: Date
    if (typeof timestamp === 'string') {
      let timestampStr = timestamp.trim()
      const hasTimezone =
        timestampStr.endsWith('Z') ||
        timestampStr.includes('+') ||
        (timestampStr.includes('-') &&
          timestampStr.length > 19 &&
          (timestampStr[19] === '-' || timestampStr[19] === '+'))
      if (!hasTimezone && timestampStr.length > 0) {
        timestampStr = timestampStr.replace(/[^\d\-:T\s]/g, '') + 'Z'
      }
      date = new Date(timestampStr)
      if (isNaN(date.getTime())) date = new Date(timestamp)
    } else {
      date = timestamp
    }
    if (isNaN(date.getTime())) return 'Invalid Time'
    return new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Kolkata',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    }).format(date)
  } catch (error) {
    console.error('Error formatting time:', error, timestamp)
    return 'Invalid Time'
  }
}

export function roundToMinute(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  date.setUTCSeconds(0, 0)
  return date.toISOString()
}

export function aggregateByMinute(data: ChartDataPoint[]): ChartDataPoint[] {
  const minuteMap = new Map<string, ChartDataPoint>()
  data.forEach((point) => {
    const minuteKey = roundToMinute(point.time || point.date || '')
    const existing = minuteMap.get(minuteKey) || {
      time: minuteKey,
      requests: 0,
      blocked: 0,
      allowed: 0,
      sql: 0,
      xss: 0,
      ddos: 0,
      other: 0,
    }
    minuteMap.set(minuteKey, {
      ...existing,
      requests: (existing.requests || 0) + (point.requests || 0),
      blocked: (existing.blocked || 0) + (point.blocked || 0),
      allowed: (existing.allowed || 0) + (point.allowed || 0),
      sql: (existing.sql || 0) + (point.sql || 0),
      xss: (existing.xss || 0) + (point.xss || 0),
      ddos: (existing.ddos || 0) + (point.ddos || 0),
      other: (existing.other || 0) + (point.other || 0),
    })
  })
  return Array.from(minuteMap.values()).sort(
    (a, b) => new Date(a.time || '').getTime() - new Date(b.time || '').getTime()
  )
}

export function buildTop10ThreatTypes(stats: Record<string, number>): TopThreatItem[] {
  return Object.entries(stats)
    .filter(([, count]) => count > 0)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
}

export function buildTop10FromThreats(threats: { type: string }[]): TopThreatItem[] {
  const counts = new Map<string, number>()
  threats.forEach((t) => {
    const type = t.type || 'Other'
    counts.set(type, (counts.get(type) ?? 0) + 1)
  })
  return Array.from(counts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
}
