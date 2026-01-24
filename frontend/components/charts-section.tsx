'use client'

import { useEffect, useState } from 'react'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { chartsApi, ChartDataPoint, wsManager } from '@/lib/api'

interface ChartsSectionProps {
  timeRange: string
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
        <p className="text-sm font-medium text-foreground mb-2">{`Time: ${label}`}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {`${entry.name}: ${entry.value}${entry.dataKey === 'latency' || entry.dataKey === 'cpu' || entry.dataKey === 'memory' ? '%' : ''}`}
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
  const [performanceData, setPerformanceData] = useState<ChartDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch chart data on mount and when timeRange changes
  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const [requestsResponse, threatsResponse, performanceResponse] = await Promise.all([
          chartsApi.getRequests(timeRange),
          chartsApi.getThreats(timeRange),
          chartsApi.getPerformance(timeRange),
        ])

        if (requestsResponse.success) setRequestData(requestsResponse.data)
        if (threatsResponse.success) setThreatData(threatsResponse.data)
        if (performanceResponse.success) setPerformanceData(performanceResponse.data)

      } catch (err) {
        console.error('[ChartsSection] Failed to fetch chart data:', err)
        setError('Failed to load chart data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchChartData()

    // Subscribe to real-time chart updates
    const handleChartUpdate = (data: any) => {
      // Update specific chart data based on type
      if (data.type === 'requests') {
        setRequestData(prev => [...prev.slice(-19), data.data]) // Keep last 20 points
      } else if (data.type === 'threats') {
        setThreatData(prev => [...prev.slice(-19), data.data])
      } else if (data.type === 'performance') {
        setPerformanceData(prev => [...prev.slice(-19), data.data])
      }
    }

    wsManager.subscribe('chart', handleChartUpdate)

    return () => {
      wsManager.unsubscribe('chart')
    }
  }, [timeRange])

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
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 md:mb-6 gap-2">
            <h2 className="text-lg font-semibold text-foreground security-text-metric">Request Volume & Threats</h2>
            <div className="flex gap-2">
              <span className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-md font-medium">
                Live
              </span>
              <button className="px-3 py-1 text-xs bg-muted text-muted-foreground rounded-md hover:bg-muted/80">
                Export
              </button>
            </div>
          </div>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={requestData}>
            <defs>
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
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip content={<CustomTooltip />} />
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
          <BarChart data={threatData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar dataKey="sql" stackId="a" fill="#dc2626" name="SQL Injection" />
            <Bar dataKey="xss" stackId="a" fill="#ea580c" name="XSS Attacks" />
            <Bar dataKey="ddos" stackId="a" fill="#f59e0b" name="DDoS Attacks" />
            <Bar dataKey="other" stackId="a" fill="#7c3aed" name="Other Threats" />
          </BarChart>
        </ResponsiveContainer>
      </div>

        {/* Performance Metrics */}
        <div className="bg-card rounded-lg border border-border p-4 md:p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 md:mb-6 gap-2">
            <h2 className="text-lg font-semibold text-foreground security-text-metric">System Performance</h2>
            <button className="px-3 py-1 text-xs bg-muted text-muted-foreground rounded-md hover:bg-muted/80">
              Settings
            </button>
          </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="latency"
              stroke="#2563eb"
              strokeWidth={2}
              name="Response Time (ms)"
              dot={{ fill: '#2563eb', strokeWidth: 1, r: 3 }}
              activeDot={{ r: 5, stroke: '#2563eb', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="cpu"
              stroke="#16a34a"
              strokeWidth={2}
              name="CPU Usage %"
              dot={{ fill: '#16a34a', strokeWidth: 1, r: 3 }}
              activeDot={{ r: 5, stroke: '#16a34a', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="memory"
              stroke="#f59e0b"
              strokeWidth={2}
              name="Memory Usage %"
              dot={{ fill: '#f59e0b', strokeWidth: 1, r: 3 }}
              activeDot={{ r: 5, stroke: '#f59e0b', strokeWidth: 2 }}
            />
          </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
