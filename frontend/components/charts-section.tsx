'use client'

import { useEffect, useState, useCallback, useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { chartsApi, ChartDataPoint, wsManager, trafficApi, threatsApi } from '@/lib/api'

interface ChartsSectionProps {
  timeRange: string
}

// Utility function to format time in IST with 12-hour format and AM/PM
function formatTimeIST(timestamp: string | Date): string {
  try {
    let date: Date
    
    if (typeof timestamp === 'string') {
      // Backend sends UTC timestamps - ensure they're treated as UTC
      let timestampStr = timestamp.trim()
      
      // Check if it already has timezone indicator
      const hasTimezone = timestampStr.endsWith('Z') || 
                         timestampStr.includes('+') || 
                         (timestampStr.includes('-') && timestampStr.length > 19 && (timestampStr[19] === '-' || timestampStr[19] === '+'))
      
      // If no timezone info, treat as UTC by appending 'Z'
      if (!hasTimezone && timestampStr.length > 0) {
        timestampStr = timestampStr.replace(/[^\d\-:T\s]/g, '') + 'Z'
      }
      
      date = new Date(timestampStr)
      
      // Validate the date
      if (isNaN(date.getTime())) {
        date = new Date(timestamp) // Last resort
      }
    } else {
      date = timestamp
    }
    
    // Validate date
    if (isNaN(date.getTime())) {
      return 'Invalid Time'
    }
    
    // Use Intl.DateTimeFormat to convert to IST and format in 12-hour format
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Kolkata',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    })
    
    return formatter.format(date)
  } catch (error) {
    console.error('Error formatting time:', error, timestamp)
    return 'Invalid Time'
  }
}

// Function to round timestamp to nearest minute
function roundToMinute(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  date.setUTCSeconds(0, 0)
  return date.toISOString()
}

// Build top 10 threat types for bar chart from stats or threat list
interface TopThreatItem {
  name: string
  count: number
}

function buildTop10ThreatTypes(stats: Record<string, number>): TopThreatItem[] {
  return Object.entries(stats)
    .filter(([, count]) => count > 0)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
}

function buildTop10FromThreats(threats: { type: string }[]): TopThreatItem[] {
  const counts = new Map<string, number>()
  threats.forEach(t => {
    const type = t.type || 'Other'
    counts.set(type, (counts.get(type) ?? 0) + 1)
  })
  return Array.from(counts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
}

// Function to aggregate data by 1-minute intervals
function aggregateByMinute(data: ChartDataPoint[]): ChartDataPoint[] {
  const minuteMap = new Map<string, ChartDataPoint>()
  
  data.forEach(point => {
    const minuteKey = roundToMinute(point.time || point.date || '')
    const existing = minuteMap.get(minuteKey) || {
      time: minuteKey,
      requests: 0,
      blocked: 0,
      allowed: 0,
      sql: 0,
      xss: 0,
      ddos: 0,
      other: 0
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
  
  return Array.from(minuteMap.values()).sort((a, b) => 
    new Date(a.time || '').getTime() - new Date(b.time || '').getTime()
  )
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        className="rounded-md p-3 shadow-lg border-2"
        style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
      >
        <p className="text-sm font-medium mb-2" style={{ color: 'var(--positivus-black)' }}>{`Time: ${label}`}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {`${entry.name}: ${entry.value}`}
          </p>
        ))}
      </div>
    )
  }
  return null
}

export function ChartsSection({ timeRange }: ChartsSectionProps) {
  const router = useRouter()
  const [requestData, setRequestData] = useState<ChartDataPoint[]>([])
  const [topThreatTypes, setTopThreatTypes] = useState<TopThreatItem[]>([])
  
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [trafficCounts, setTrafficCounts] = useState<Map<string, { requests: number; blocked: number; allowed: number }>>(new Map())

  // Fetch real-time traffic data and aggregate by minute
  const fetchRealTimeTraffic = useCallback(async () => {
    try {
      // Get recent traffic (last 4 hours for 1-minute intervals = 240 points max)
      const response = await trafficApi.getRecent(1000)
      if (response.success) {
        // Group by minute
        const minuteMap = new Map<string, { requests: number; blocked: number; allowed: number }>()
        
        response.data.forEach((traffic: any) => {
          const timestamp = traffic.timestamp || traffic.time
          if (!timestamp) return
          
          const minuteKey = roundToMinute(timestamp)
          const existing = minuteMap.get(minuteKey) || { requests: 0, blocked: 0, allowed: 0 }
          const wasBlocked = traffic.was_blocked || traffic.wasBlocked || false
          
          minuteMap.set(minuteKey, {
            requests: existing.requests + 1,
            blocked: existing.blocked + (wasBlocked ? 1 : 0),
            allowed: existing.allowed + (wasBlocked ? 0 : 1),
          })
        })
        
        setTrafficCounts(minuteMap)
        
        // Convert to chart data format, keep last 60 minutes
        const chartData: ChartDataPoint[] = Array.from(minuteMap.entries())
          .map(([time, counts]) => ({
            time,
            requests: counts.requests,
            blocked: counts.blocked,
            allowed: counts.allowed,
          }))
          .sort((a, b) => new Date(a.time || '').getTime() - new Date(b.time || '').getTime())
          .slice(-60) // Keep last 60 minutes
        
        console.log('[ChartsSection] Fetched traffic data:', {
          trafficCount: response.data.length,
          chartDataPoints: chartData.length,
          samplePoint: chartData[0]
        })
        setRequestData(chartData)
      }
    } catch (err: any) {
      if (!err?.isNetworkError) {
        console.error('[ChartsSection] Failed to fetch real-time traffic:', err)
      }
    }
  }, [])

  // Fetch chart data on mount and when timeRange changes
  useEffect(() => {
    // Clear previous data on timeRange change
    setRequestData([])
    setTopThreatTypes([])
    setTrafficCounts(new Map())

    const fetchChartData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // For real-time view, fetch traffic AND threats
        if (timeRange === '1h' || timeRange === '24h') {
          await fetchRealTimeTraffic()
          // Always fetch threat data and top 10 threat types
          try {
            const statsRes = await threatsApi.getStats(timeRange)
            if (statsRes.success && statsRes.data && Object.keys(statsRes.data).length) {
              setTopThreatTypes(buildTop10ThreatTypes(statsRes.data))
            } else {
              const threatsByRange = await threatsApi.getByTimeRange(timeRange)
              if (threatsByRange.success && threatsByRange.data?.length) {
                setTopThreatTypes(buildTop10FromThreats(threatsByRange.data))
              }
            }
          } catch (_) {
            try {
              const threatsByRange = await threatsApi.getByTimeRange(timeRange)
              if (threatsByRange.success && threatsByRange.data?.length) {
                setTopThreatTypes(buildTop10FromThreats(threatsByRange.data))
              }
            } catch (__) {}
          }
        } else {
          const [requestsResponse, statsRes] = await Promise.all([
            chartsApi.getRequests(timeRange),
            threatsApi.getStats(timeRange),
          ])

          if (requestsResponse.success) {
            const aggregated = aggregateByMinute(requestsResponse.data)
            setRequestData(aggregated.slice(-60))
          }
          // Set top 10 threat types from stats
          if (statsRes.success && statsRes.data && Object.keys(statsRes.data).length) {
            setTopThreatTypes(buildTop10ThreatTypes(statsRes.data))
          } else {
            try {
              const threatsByRange = await threatsApi.getByTimeRange(timeRange)
              if (threatsByRange.success && threatsByRange.data?.length) {
                setTopThreatTypes(buildTop10FromThreats(threatsByRange.data))
              }
            } catch (_) {}
          }
        }

      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[ChartsSection] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[ChartsSection] Failed to fetch chart data:', err)
          setError('Failed to load chart data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchChartData()

    // Real-time polling every 30 seconds to update top 10 threat types
    const pollTopThreats = async () => {
      try {
        const r = await threatsApi.getStats(timeRange)
        if (r.success && r.data && Object.keys(r.data).length) {
          setTopThreatTypes(buildTop10ThreatTypes(r.data))
        } else {
          const tr = await threatsApi.getByTimeRange(timeRange)
          if (tr.success && tr.data?.length) setTopThreatTypes(buildTop10FromThreats(tr.data))
        }
      } catch (_) {}
    }
    const pollingInterval = setInterval(() => {
      if (timeRange === '1h' || timeRange === '24h') {
        fetchRealTimeTraffic()
        pollTopThreats()
      } else {
        pollTopThreats()
      }
    }, 30000) // Poll every 30 seconds

    // Subscribe to real-time updates
    const handleTrafficUpdate = (data: any) => {
      // Update request data when new traffic arrives
      const now = new Date()
      const minuteKey = roundToMinute(now.toISOString())

      setTrafficCounts(prev => {
        const existing = prev.get(minuteKey) || { requests: 0, blocked: 0, allowed: 0 }
        const updated = new Map(prev)
        updated.set(minuteKey, {
          requests: existing.requests + 1,
          blocked: existing.blocked + (data.was_blocked ? 1 : 0),
          allowed: existing.allowed + (data.was_blocked ? 0 : 1),
        })
        return updated
      })

      setRequestData(prev => {
        const minuteKey = roundToMinute(now.toISOString())
        const existingIndex = prev.findIndex(p => roundToMinute(p.time || '') === minuteKey)

        if (existingIndex >= 0) {
          // Update existing minute
          const updated = [...prev]
          const existing = updated[existingIndex]
          updated[existingIndex] = {
            ...existing,
            requests: (existing.requests || 0) + 1,
            blocked: (existing.blocked || 0) + (data.was_blocked ? 1 : 0),
            allowed: (existing.allowed || 0) + (data.was_blocked ? 0 : 1),
          }
          return updated.slice(-60) // Keep last 60 minutes
        } else {
          // Add new minute
          const newPoint: ChartDataPoint = {
            time: minuteKey,
            requests: 1,
            blocked: data.was_blocked ? 1 : 0,
            allowed: data.was_blocked ? 0 : 1,
          }
          return [...prev, newPoint].slice(-60) // Keep last 60 minutes
        }
      })
    }

    const handleThreatUpdate = () => {
      pollTopThreats()
    }

    wsManager.subscribe('traffic', handleTrafficUpdate)
    wsManager.subscribe('threat', handleThreatUpdate)

    return () => {
      clearInterval(pollingInterval)
      wsManager.unsubscribe('traffic')
      wsManager.unsubscribe('threat')
    }
  }, [timeRange, fetchRealTimeTraffic])

  // Format chart data with IST timestamps for display
  const formattedRequestData = useMemo(() => {
    const formatted = requestData.map(point => ({
      ...point,
      timeFormatted: formatTimeIST(point.time || ''),
    }))
    // Debug logging
    if (formatted.length > 0) {
      console.log('[ChartsSection] Formatted request data:', formatted.slice(0, 3), `... (${formatted.length} total points)`)
    } else {
      console.log('[ChartsSection] No request data available')
    }
    return formatted
  }, [requestData])

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
          <div className="flex items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--positivus-green)' }}></div>
            <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>Loading chart data...</span>
          </div>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-6">
        <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
          <div className="flex items-center justify-center py-16">
            <div className="text-destructive">Failed to load chart data: {error}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Charts Grid - Responsive Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Requests & Threats Chart */}
        <div className="rounded-md p-4 md:p-6 xl:col-span-2 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
          <div className="mb-4 md:mb-6 flex items-center">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Request Volume & Threats</h2>
          </div>
        {formattedRequestData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md" style={{ borderColor: 'var(--positivus-gray)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--positivus-gray-dark)' }}>No data available</p>
            <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>Waiting for traffic data...</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={formattedRequestData}>
              <defs>
                <pattern id="requestsPattern" patternUnits="userSpaceOnUse" width="4" height="4">
                  <rect width="4" height="4" fill="var(--positivus-green-bg)"/>
                  <rect width="2" height="2" fill="var(--positivus-green-light)"/>
                </pattern>
                <pattern id="blockedPattern" patternUnits="userSpaceOnUse" width="4" height="4">
                  <rect width="4" height="4" fill="var(--destructive-bg)"/>
                  <rect width="2" height="2" fill="var(--destructive)"/>
                </pattern>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--positivus-gray-dark)" />
              <XAxis 
                dataKey="timeFormatted" 
                stroke="var(--positivus-gray-dark)" 
                fontSize={11}
                angle={-45}
                textAnchor="end"
                height={80}
                interval="preserveStartEnd"
              />
              <YAxis stroke="var(--positivus-gray-dark)" fontSize={12} />
              <Tooltip 
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="rounded-md p-3 shadow-lg border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                        <p className="text-sm font-medium mb-2" style={{ color: 'var(--positivus-black)' }}>{`Time: ${label} IST`}</p>
                        {payload.map((entry: any, index: number) => (
                          <p key={index} className="text-sm" style={{ color: entry.color }}>
                            {`${entry.name}: ${entry.value}`}
                          </p>
                        ))}
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="requests"
                stroke="#C5E246"
                strokeWidth={2}
                fillOpacity={0.3}
                fill="url(#requestsPattern)"
                name="Total Requests"
              />
              <Area
                type="monotone"
                dataKey="blocked"
                stroke="#dc2626"
                strokeWidth={2}
                fillOpacity={0.3}
                fill="url(#blockedPattern)"
                name="Blocked Threats"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

        {/* Top 10 Threat Types - Horizontal Bar Chart (full width) */}
        <div className="rounded-md p-4 md:p-6 xl:col-span-2 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
          <div className="flex items-center justify-between gap-4 mb-4 md:mb-6">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Top 10 Threat Types</h2>
            <button
              onClick={() => router.push('/threats')}
              className="px-3 py-1 text-xs rounded-none border-2 shrink-0 hover:bg-accent"
              style={{ backgroundColor: 'var(--positivus-gray)', borderColor: 'var(--positivus-gray)', color: 'var(--positivus-gray-dark)' }}
            >
              Details
            </button>
          </div>
        {topThreatTypes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md text-center" style={{ borderColor: 'var(--positivus-gray)' }}>
            <p className="text-sm font-medium mb-1" style={{ color: 'var(--positivus-gray-dark)' }}>No threats detected</p>
            <p className="text-xs mb-4" style={{ color: 'var(--positivus-gray-dark)' }}>Top attack types will appear when threats are blocked</p>
            <div className="flex gap-2 flex-wrap justify-center">
              <span className="px-2 py-1 text-xs rounded bg-security-critical/20 text-security-critical">SQL Injection</span>
              <span className="px-2 py-1 text-xs rounded bg-security-high/20 text-security-high">XSS</span>
              <span className="px-2 py-1 text-xs rounded bg-security-medium/20 text-security-medium">DDoS</span>
              <span className="px-2 py-1 text-xs rounded bg-security-config/20 text-security-config">Path Traversal</span>
            </div>
          </div>
        ) : (
        <ResponsiveContainer width="100%" height={Math.max(250, topThreatTypes.length * 36)}>
          <BarChart data={topThreatTypes} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--positivus-gray)" horizontal={false} />
            <XAxis type="number" stroke="var(--positivus-gray-dark)" fontSize={11} allowDecimals={false} />
            <YAxis type="category" dataKey="name" stroke="var(--positivus-gray-dark)" fontSize={11} width={120} tick={{ fontSize: 11 }} />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload
                  return (
                    <div className="rounded-none p-3 shadow-lg border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                      <p className="text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>{item.name}</p>
                      <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>{item.count} blocked</p>
                    </div>
                  )
                }
                return null
              }}
            />
            <Bar dataKey="count" fill="var(--positivus-green)" radius={[0, 4, 4, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
        )}
      </div>
      </div>
    </div>
   )
}
