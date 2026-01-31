'use client'

import { useEffect, useState, useCallback, useMemo } from 'react'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { chartsApi, ChartDataPoint, wsManager, trafficApi } from '@/lib/api'

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
      <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
        <p className="text-sm font-medium text-foreground mb-2">{`Time: ${label}`}</p>
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
  const [requestData, setRequestData] = useState<ChartDataPoint[]>([])
  const [threatData, setThreatData] = useState<ChartDataPoint[]>([])
  
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
    setThreatData([])
    setTrafficCounts(new Map())

    const fetchChartData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // For real-time view, fetch actual traffic data
        if (timeRange === '1h' || timeRange === '24h') {
          await fetchRealTimeTraffic()
        } else {
          const [requestsResponse, threatsResponse] = await Promise.all([
            chartsApi.getRequests(timeRange),
            chartsApi.getThreats(timeRange),
          ])

          if (requestsResponse.success) {
            const aggregated = aggregateByMinute(requestsResponse.data)
            setRequestData(aggregated.slice(-60)) // Keep last 60 minutes
          }
          if (threatsResponse.success) {
            const aggregated = aggregateByMinute(threatsResponse.data)
            setThreatData(aggregated.slice(-60))
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

    // Real-time polling every 30 seconds to update chart
    const pollingInterval = setInterval(() => {
      if (timeRange === '1h' || timeRange === '24h') {
        fetchRealTimeTraffic()
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

    const handleThreatUpdate = (data: any) => {
      // Update threat data when new threat detected
      const now = new Date()
      const minuteKey = roundToMinute(now.toISOString())
      const threatType = data.type?.toLowerCase() || 'other'

      setThreatData(prev => {
        const existingIndex = prev.findIndex(p => roundToMinute(p.time || '') === minuteKey)

        if (existingIndex >= 0) {
          const updated = [...prev]
          const existing = updated[existingIndex]
          updated[existingIndex] = {
            ...existing,
            sql: (existing.sql || 0) + (threatType.includes('sql') || threatType.includes('injection') ? 1 : 0),
            xss: (existing.xss || 0) + (threatType.includes('xss') || threatType.includes('cross-site') ? 1 : 0),
            ddos: (existing.ddos || 0) + (threatType.includes('ddos') || threatType.includes('dos') ? 1 : 0),
            other: (existing.other || 0) + (!threatType.includes('sql') && !threatType.includes('xss') && !threatType.includes('ddos') ? 1 : 0),
          }
          return updated.slice(-60)
        } else {
          const newPoint: ChartDataPoint = {
            time: minuteKey,
            sql: threatType.includes('sql') || threatType.includes('injection') ? 1 : 0,
            xss: threatType.includes('xss') || threatType.includes('cross-site') ? 1 : 0,
            ddos: threatType.includes('ddos') || threatType.includes('dos') ? 1 : 0,
            other: !threatType.includes('sql') && !threatType.includes('xss') && !threatType.includes('ddos') ? 1 : 0,
          }
          return [...prev, newPoint].slice(-60)
        }
      })
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

  const formattedThreatData = useMemo(() => {
    return threatData.map(point => ({
      ...point,
      timeFormatted: formatTimeIST(point.time || ''),
    }))
  }, [threatData])

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-card rounded-lg border border-border p-4 md:p-6">
          <div className="flex items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2 text-muted-foreground">Loading chart data...</span>
          </div>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-6">
        <div className="bg-card rounded-lg border border-border p-4 md:p-6">
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
        <div className="bg-card rounded-lg border border-border p-4 md:p-6 xl:col-span-2">
          <div className="mb-4 md:mb-6">
            <h2 className="text-lg font-semibold text-foreground security-text-metric">Request Volume & Threats</h2>
          </div>
        {formattedRequestData.length === 0 ? (
          <div className="flex items-center justify-center h-[300px] border-2 border-dashed border-border rounded-lg">
            <div className="text-center">
              <p className="text-muted-foreground text-sm mb-2">No data available</p>
              <p className="text-muted-foreground text-xs">Waiting for traffic data...</p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={formattedRequestData}>
              <defs>
                <pattern id="requestsPattern" patternUnits="userSpaceOnUse" width="4" height="4">
                  <rect width="4" height="4" fill="#ecfeff"/>
                  <rect width="2" height="2" fill="#cffafe"/>
                </pattern>
                <pattern id="blockedPattern" patternUnits="userSpaceOnUse" width="4" height="4">
                  <rect width="4" height="4" fill="#e0f2fe"/>
                  <rect width="2" height="2" fill="#bae6fd"/>
                </pattern>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timeFormatted" 
                stroke="#6b7280" 
                fontSize={11}
                angle={-45}
                textAnchor="end"
                height={80}
                interval="preserveStartEnd"
              />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip 
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
                        <p className="text-sm font-medium text-foreground mb-2">{`Time: ${label} IST`}</p>
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
                stroke="#06b6d4"
                strokeWidth={2}
                fillOpacity={0.3}
                fill="url(#requestsPattern)"
                name="Total Requests"
              />
              <Area
                type="monotone"
                dataKey="blocked"
                stroke="#0891b2"
                strokeWidth={2}
                fillOpacity={0.3}
                fill="url(#blockedPattern)"
                name="Blocked Threats"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

        {/* Threat Types Breakdown */}
        <div className="bg-card rounded-lg border border-border p-4 md:p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 md:mb-6 gap-2">
            <h2 className="text-lg font-semibold text-foreground security-text-metric">Threat Types</h2>
            <button className="px-3 py-1 text-xs bg-muted text-muted-foreground rounded-md hover:bg-muted/80">
              Details
            </button>
          </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={formattedThreatData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="timeFormatted" 
              stroke="#6b7280" 
              fontSize={11}
              angle={-45}
              textAnchor="end"
              height={80}
              interval="preserveStartEnd"
            />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
                      <p className="text-sm font-medium text-foreground mb-2">{`Time: ${label} IST`}</p>
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
            <Bar dataKey="sql" stackId="a" fill="#dc2626" name="SQL Injection" />
            <Bar dataKey="xss" stackId="a" fill="#ea580c" name="XSS Attacks" />
            <Bar dataKey="ddos" stackId="a" fill="#f59e0b" name="DDoS Attacks" />
            <Bar dataKey="other" stackId="a" fill="#7c3aed" name="Other Threats" />
</BarChart>
        </ResponsiveContainer>
      </div>
      </div>
    </div>
   )
}
