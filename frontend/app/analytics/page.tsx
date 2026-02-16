'use client'

import { useEffect, useState, useMemo } from 'react'
import { usePathname } from 'next/navigation'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertCircle, Download, TrendingUp, Shield, Ban } from 'lucide-react'
import { analyticsApi, chartsApi, ChartDataPoint, wsManager } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'
import { useTimezone } from '@/contexts/timezone-context'
import { formatTimeLocal } from '@/lib/chart-utils'

// Positivus theme chart colors
const CHART_COLORS = {
  requests: '#C5E246',
  blocked: '#dc2626',
  allowed: '#22c55e',
  pie: ['#dc2626', '#ea580c', '#f59e0b', '#7c3aed', '#06b6d4', '#22c55e'],
}

export default function AnalyticsPage() {
  const pathname = usePathname()
  const { timezone } = useTimezone()
  const [timeRange, setTimeRange] = useState('24h')
  const [analyticsData, setAnalyticsData] = useState<ChartDataPoint[]>([])
  const [summary, setSummary] = useState<Record<string, any>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch analytics data
  useEffect(() => {
    const fetchAnalyticsData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        // Clear previous data when navigating to this page
        setAnalyticsData([])
        setSummary({})

        let chartData: ChartDataPoint[] = []
        let summaryData: Record<string, any> = {}

        const [overviewResponse, summaryResponse] = await Promise.all([
          analyticsApi.getOverview(timeRange),
          analyticsApi.getSummary(timeRange)
        ])
        if (overviewResponse.success && overviewResponse.data?.length) {
          chartData = overviewResponse.data
        } else {
          const chartsRes = await chartsApi.getRequests(timeRange)
          if (chartsRes.success && chartsRes.data?.length) chartData = chartsRes.data
        }
        setAnalyticsData(chartData)

        if (summaryResponse.success && summaryResponse.data) {
          const s = summaryResponse.data
          summaryData = { ...s, attack_rate: s.attack_rate ?? s.block_rate ?? 0 }
        } else if (chartData.length) {
          const total = chartData.reduce((sum, d) => sum + (d.requests || 0), 0)
          const blocked = chartData.reduce((sum, d) => sum + (d.blocked || 0), 0)
          const allowed = chartData.reduce((sum, d) => sum + (d.allowed || 0), 0)
          summaryData = { total_requests: total, blocked_requests: blocked, allowed_requests: allowed, attack_rate: total ? (blocked / total) * 100 : 0 }
        }
        setSummary(summaryData)
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[AnalyticsPage] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[AnalyticsPage] Failed to fetch analytics data:', err)
          setError('Failed to load analytics data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchAnalyticsData()

    // Subscribe to real-time updates
    const handleMetricsUpdate = (data: any) => {
      const now = new Date().toISOString()
      setAnalyticsData(prev => {
        const newPoint: ChartDataPoint = {
          time: now,
          requests: data.total_requests || 0,
          blocked: data.blocked_requests || 0,
          allowed: data.allowed_requests || 0,
        }
        return [...prev.slice(-59), newPoint]
      })
    }

    wsManager.subscribe('metrics', handleMetricsUpdate)

    return () => {
      wsManager.unsubscribe('metrics')
    }
  }, [pathname, timeRange])

  const formattedData = useMemo(
    () => analyticsData.map((d) => ({ ...d, timeFormatted: formatTimeLocal(d.time || '', timezone) })),
    [analyticsData, timezone]
  )

  const chartTooltipStyle = {
    backgroundColor: 'var(--positivus-white)',
    border: '2px solid var(--positivus-gray)',
    borderRadius: 0,
  }

  // Prepare pie chart data for attack types
  const attackTypeData = [
    { name: 'SQL Injection', value: summary.sql_injection || 0 },
    { name: 'XSS', value: summary.xss || 0 },
    { name: 'DDoS', value: summary.ddos || 0 },
    { name: 'Other', value: summary.other || 0 },
  ].filter(item => item.value > 0)

  const handleExport = () => {
    const csv = [
      ['Time', 'Requests', 'Blocked', 'Allowed'].join(','),
      ...analyticsData.map(d => [
        d.time || d.date || '',
        d.requests || 0,
        d.blocked || 0,
        d.allowed || 0
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `analytics-${new Date().toISOString()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen" style={{ backgroundColor: 'var(--positivus-gray)' }}>
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Analytics</h2>
                <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>Detailed analytics and statistics for your WAF</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <Loader2 className="animate-spin h-8 w-8" style={{ color: 'var(--positivus-green)' }} />
                <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>Loading analytics data...</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-screen" style={{ backgroundColor: 'var(--positivus-gray)' }}>
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Analytics</h2>
                <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>Detailed analytics and statistics for your WAF</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <span className="ml-2 text-destructive">{error}</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  const chartCardClass = 'p-6 border-2 rounded-md'
  const chartCardStyle = { backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }
  const emptyState = (
    <div className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md" style={{ borderColor: 'var(--positivus-gray)' }}>
      <TrendingUp className="h-12 w-12 mb-3" style={{ color: 'var(--positivus-gray)' }} />
      <p className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>No data available</p>
      <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Waiting for traffic data...</p>
    </div>
  )

  return (
    <div className="flex h-screen" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Analytics</h2>
                  <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>Detailed analytics and statistics for your WAF</p>
                </div>
                <Button onClick={handleExport} variant="outline" className="rounded-none border-2 hover:bg-[#E8F5B8]" style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}>
                  <Download className="mr-2 h-4 w-4" />
                  Export Data
                </Button>
              </div>

              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                  { label: 'Total Requests', value: summary.total_requests?.toLocaleString() || '0', Icon: TrendingUp },
                  { label: 'Blocked', value: summary.blocked_requests?.toLocaleString() || '0', Icon: Ban },
                  { label: 'Allowed', value: summary.allowed_requests?.toLocaleString() || '0', Icon: Shield },
                  { label: 'Attack Rate', value: summary.attack_rate != null ? `${Number(summary.attack_rate).toFixed(1)}%` : '0%', Icon: AlertCircle },
                ].map(({ label, value, Icon }) => (
                  <Card key={label} className={chartCardClass} style={chartCardStyle}>
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                        <Icon size={20} style={{ color: 'var(--positivus-green)' }} />
                      </div>
                      <div>
                        <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>{label}</p>
                        <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>{value}</p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>

              {/* Charts Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className={chartCardClass} style={chartCardStyle}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Request Trends</h3>
                  {formattedData.length === 0 ? emptyState : (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={formattedData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--positivus-gray)" vertical={false} />
                        <XAxis dataKey="timeFormatted" stroke="var(--positivus-gray-dark)" fontSize={11} angle={-45} textAnchor="end" height={70} interval="preserveStartEnd" />
                        <YAxis stroke="var(--positivus-gray-dark)" fontSize={12} allowDecimals={false} />
                        <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: 'var(--positivus-black)' }} labelFormatter={(v) => `Time: ${v}`} />
                        <Legend wrapperStyle={{ paddingTop: 8 }} />
                        <Line type="monotone" dataKey="requests" stroke={CHART_COLORS.requests} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name="Total Requests" />
                        <Line type="monotone" dataKey="blocked" stroke={CHART_COLORS.blocked} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name="Blocked" />
                        <Line type="monotone" dataKey="allowed" stroke={CHART_COLORS.allowed} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name="Allowed" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </Card>

                <Card className={chartCardClass} style={chartCardStyle}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Blocked vs Allowed</h3>
                  {formattedData.length === 0 ? emptyState : (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={formattedData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--positivus-gray)" vertical={false} />
                        <XAxis dataKey="timeFormatted" stroke="var(--positivus-gray-dark)" fontSize={11} angle={-45} textAnchor="end" height={70} interval="preserveStartEnd" />
                        <YAxis stroke="var(--positivus-gray-dark)" fontSize={12} allowDecimals={false} />
                        <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: 'var(--positivus-black)' }} labelFormatter={(v) => `Time: ${v}`} />
                        <Legend wrapperStyle={{ paddingTop: 8 }} />
                        <Bar dataKey="blocked" fill={CHART_COLORS.blocked} radius={[4, 4, 0, 0]} name="Blocked" />
                        <Bar dataKey="allowed" fill={CHART_COLORS.allowed} radius={[4, 4, 0, 0]} name="Allowed" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </Card>

                {attackTypeData.length > 0 && (
                  <Card className={chartCardClass} style={chartCardStyle}>
                    <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Attack Type Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={attackTypeData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={90}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {attackTypeData.map((_, i) => (
                            <Cell key={i} fill={CHART_COLORS.pie[i % CHART_COLORS.pie.length]} stroke="var(--positivus-white)" strokeWidth={2} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={chartTooltipStyle} formatter={(v: number) => [v, 'Count']} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                )}

                <Card className={chartCardClass} style={chartCardStyle}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Request Volume Over Time</h3>
                  {formattedData.length === 0 ? emptyState : (
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={formattedData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                        <defs>
                          <linearGradient id="areaRequestsAnalytics" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={CHART_COLORS.requests} stopOpacity={0.4} />
                            <stop offset="100%" stopColor={CHART_COLORS.requests} stopOpacity={0.05} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--positivus-gray)" vertical={false} />
                        <XAxis dataKey="timeFormatted" stroke="var(--positivus-gray-dark)" fontSize={11} angle={-45} textAnchor="end" height={70} interval="preserveStartEnd" />
                        <YAxis stroke="var(--positivus-gray-dark)" fontSize={12} allowDecimals={false} />
                        <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: 'var(--positivus-black)' }} labelFormatter={(v) => `Time: ${v}`} />
                        <Legend wrapperStyle={{ paddingTop: 8 }} />
                        <Area type="monotone" dataKey="requests" stroke={CHART_COLORS.requests} strokeWidth={2} fill="url(#areaRequestsAnalytics)" name="Requests" />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </Card>
              </div>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
